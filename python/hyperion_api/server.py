"""gRPC сервер — мост между Python-мозгами и Java-мускулами.

Два режима:
  1. gRPC (proto-compiled) — для связки с Java orchestrator
  2. Standalone — run_inference() напрямую, без proto

Model registry: зарегистрируй модель по имени, дальше дёргай через API.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent import futures
from typing import Any, Optional

import grpc
import jax.random as jrandom
import numpy as np

from hyperion_backends.jax_backend import JAXBackend
from hyperion_inference.base import InferenceResult

logger = logging.getLogger(__name__)


_MODEL_REGISTRY: dict[str, Any] = {}


def register_model(name: str, model_fn: Any) -> None:
    """Регистрируем модель в глобальном реестре по имени."""
    _MODEL_REGISTRY[name] = model_fn
    logger.info("Model '%s' registered", name)


def list_models() -> list[str]:
    """Возвращаем список зарегистрированных моделей."""
    return list(_MODEL_REGISTRY.keys())


def tensor_to_proto(arr: Any) -> dict[str, Any]:
    """Конвертим numpy array в TensorProto-подобный dict для сериализации."""
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "data": arr.ravel().tolist(),
        "shape": list(arr.shape),
        "dtype": "float32",
    }


def proto_to_tensor(proto_dict: dict[str, Any]) -> np.ndarray:
    """Конвертим TensorProto-подобный dict обратно в numpy array."""
    data = np.array(proto_dict["data"], dtype=np.float32)
    shape = proto_dict.get("shape", [len(data)])
    return data.reshape(shape)


class InferenceServiceImpl:
    """Реализация InferenceService. Принимает запрос, гоняет инференс, отдаёт сэмплы."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def run_inference(self, request: dict[str, Any]) -> dict[str, Any]:
        """Запускаем инференс по зарегистрированной модели.

        Args:
            request: dict с полями model_name, data, config, seed

        Returns:
            dict с job_id, status, samples, elapsed_seconds
        """
        import jax.numpy as jnp

        job_id = request.get("job_id", str(uuid.uuid4()))
        model_name = request.get("model_name", "")

        if model_name not in _MODEL_REGISTRY:
            return {
                "job_id": job_id,
                "status": "FAILED",
                "error_message": f"Model '{model_name}' not registered. "
                                 f"Available: {list_models()}",
            }

        model_fn = _MODEL_REGISTRY[model_name]

        try:
            self._jobs[job_id] = {"status": "RUNNING"}
            start = time.perf_counter()

            data = {}
            for key, tensor in request.get("data", {}).items():
                data[key] = proto_to_tensor(tensor) if isinstance(tensor, dict) else tensor

            config = request.get("config", {})
            seed = request.get("seed", 42)
            method = config.get("method", "nuts")
            num_samples = config.get("num_samples", 1000)
            num_warmup = config.get("num_warmup", 500)

            rng_key = jrandom.PRNGKey(seed)
            init_key, sample_key = jrandom.split(rng_key)

            backend = JAXBackend()
            backend.initialize(model_fn, data, init_key)
            init_pos = backend.flatten_latents(backend.sample_prior(init_key))

            if method == "hmc":
                from hyperion_inference.hmc import hmc_sample
                flat_samples, log_probs, accept_probs, info = hmc_sample(
                    backend.potential_fn, sample_key, init_pos,
                    num_samples=num_samples, num_warmup=num_warmup,
                    num_leapfrog=config.get("num_leapfrog", 10),
                )
            elif method == "nuts":
                from hyperion_inference.nuts import nuts_sample
                flat_samples, log_probs, accept_probs, info = nuts_sample(
                    backend.potential_fn, sample_key, init_pos,
                    num_samples=num_samples, num_warmup=num_warmup,
                    max_tree_depth=config.get("max_tree_depth", 10),
                )
            else:
                return {
                    "job_id": job_id,
                    "status": "FAILED",
                    "error_message": f"Unsupported method: {method}. Use 'hmc' or 'nuts'.",
                }

            import jax
            unflat = jax.vmap(backend.unflatten_latents)(flat_samples)
            samples_out = {
                name: tensor_to_proto(np.asarray(arr))
                for name, arr in unflat.items()
            }

            elapsed = time.perf_counter() - start
            self._jobs[job_id] = {"status": "COMPLETED"}

            return {
                "job_id": job_id,
                "status": "COMPLETED",
                "samples": samples_out,
                "elapsed_seconds": elapsed,
                "diagnostics": {
                    "mean_accept_prob": float(jnp.mean(accept_probs)),
                    "num_samples": num_samples,
                },
            }

        except Exception as e:
            logger.exception("Inference failed for job %s", job_id)
            self._jobs[job_id] = {"status": "FAILED", "error": str(e)}
            return {
                "job_id": job_id,
                "status": "FAILED",
                "error_message": str(e),
            }

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Статус задачи по ID."""
        if job_id not in self._jobs:
            return {"job_id": job_id, "status": "NOT_FOUND"}
        return {"job_id": job_id, **self._jobs[job_id]}


class ModelServiceImpl:
    """Реализация ModelService. Валидация моделей и экспорт IR."""

    def validate_model(self, request: dict[str, Any]) -> dict[str, Any]:
        """Валидируем исходник модели — ищем sample() и прочие признаки жизни."""
        model_source = request.get("model_source", "")
        errors: list[str] = []
        warnings: list[str] = []

        if not model_source.strip():
            errors.append("Empty model source")

        if "sample" not in model_source:
            warnings.append("No sample() calls found in model")

        if "@model" not in model_source:
            warnings.append("Missing @model decorator")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def export_ir(self, request: dict[str, Any]) -> dict[str, Any]:
        """Экспортируем IR модели в JSON."""
        model_name = request.get("model_name", "")

        if model_name not in _MODEL_REGISTRY:
            return {"error": f"Model '{model_name}' not registered"}

        model_fn = _MODEL_REGISTRY[model_name]

        try:
            from hyperion_ir.compiler import ModelCompiler
            compiler = ModelCompiler()
            data = {}
            for key, tensor in request.get("data", {}).items():
                data[key] = proto_to_tensor(tensor) if isinstance(tensor, dict) else tensor

            ir_graph = compiler.compile(model_fn, data, jrandom.PRNGKey(0))

            return {
                "ir_json": json.dumps(ir_graph.to_dict(), default=str),
                "num_nodes": len(ir_graph.nodes),
                "optimization_passes": [
                    "dead_node_elimination",
                    "common_subexpression_elimination",
                    "topological_reorder",
                ],
            }
        except Exception as e:
            return {"error": str(e)}


class HyperionServer:
    """Главный gRPC сервер. Точка входа для standalone и orchestrator режимов."""

    def __init__(self, port: int = 50051, max_workers: int = 4):
        self.port = port
        self.max_workers = max_workers
        self._server: Optional[grpc.Server] = None
        self.inference_service = InferenceServiceImpl()
        self.model_service = ModelServiceImpl()

    def start(self) -> None:
        """Поднимаем gRPC сервер на указанном порту."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers)
        )
        self._server.add_insecure_port(f"[::]:{self.port}")
        self._server.start()
        logger.info("HYPERION gRPC server started on port %d", self.port)

    def stop(self, grace: float = 5.0) -> None:
        """Останавливаем сервер с grace period."""
        if self._server:
            self._server.stop(grace)
            logger.info("HYPERION gRPC server stopped")

    def wait_for_termination(self) -> None:
        """Блокируемся до завершения сервера."""
        if self._server:
            self._server.wait_for_termination()


def serve(port: int = 50051) -> None:
    """Запускаем HYPERION сервер. Точка входа для standalone режима."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    server = HyperionServer(port=port)
    server.start()
    logger.info("Server listening on port %d, models: %s", port, list_models())
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
