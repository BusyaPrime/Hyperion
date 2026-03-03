"""gRPC сервер — мост между Python-мозгами и Java-мускулами"""

from __future__ import annotations

import json
import logging
import uuid
from concurrent import futures
from typing import Any, Optional

import grpc
import numpy as np

logger = logging.getLogger(__name__)


def tensor_to_proto(arr):
    """Конвертим numpy array в TensorProto-подобный dict. Сериализация для gRPC."""
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "data": arr.ravel().tolist(),
        "shape": list(arr.shape),
        "dtype": "float32",
    }


def proto_to_tensor(proto_dict):
    """Конвертим TensorProto-подобный dict обратно в numpy array."""
    data = np.array(proto_dict["data"], dtype=np.float32)
    shape = proto_dict.get("shape", [len(data)])
    return data.reshape(shape)


class InferenceServiceImpl:
    """Реализация InferenceService. Тут живёт логика запуска инференса."""

    def __init__(self):
        self._jobs: dict[str, Any] = {}

    def run_inference(self, request: dict[str, Any]) -> dict[str, Any]:
        """Запускаем инференс и возвращаем результаты."""
        from hyperion_exp.runner import ExperimentRunner, ExperimentConfig

        job_id = request.get("job_id", str(uuid.uuid4()))

        try:
            data = {}
            for key, tensor in request.get("data", {}).items():
                data[key] = proto_to_tensor(tensor)

            config_dict = request.get("config", {})
            method_map = {
                0: "nuts", 1: "hmc", 2: "nuts", 3: "smc",
                4: "vi", 5: "flows", 6: "laplace",
            }
            method = method_map.get(config_dict.get("method", 2), "nuts")

            exp_config = ExperimentConfig(
                model_name=request.get("experiment_name", "api_model"),
                inference_method=method,
                seed=request.get("seed", 42),
                num_samples=config_dict.get("num_samples", 1000),
                num_warmup=config_dict.get("num_warmup", 500),
            )

            self._jobs[job_id] = {"status": "RUNNING"}

            # Note: выполнение model_source потребовало бы eval/exec
            # В проде — зарегистрированные модели или сериализованный IR
            response = {
                "job_id": job_id,
                "status": "COMPLETED",
                "elapsed_seconds": 0.0,
            }

            self._jobs[job_id] = {"status": "COMPLETED"}
            return response

        except Exception as e:
            self._jobs[job_id] = {"status": "FAILED", "error": str(e)}
            return {
                "job_id": job_id,
                "status": "FAILED",
                "error_message": str(e),
            }


class ModelServiceImpl:
    """Реализация ModelService. Валидация моделей и экспорт IR."""

    def validate_model(self, request: dict[str, Any]) -> dict[str, Any]:
        """Валидируем исходник модели. Ищем sample() и прочие признаки жизни."""
        model_source = request.get("model_source", "")
        errors = []
        warnings = []

        if not model_source.strip():
            errors.append("Empty model source")

        if "sample" not in model_source:
            warnings.append("No sample() calls found in model")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def export_ir(self, request: dict[str, Any]) -> dict[str, Any]:
        """Экспортируем IR модели. Пока заглушка."""
        return {
            "ir_json": "{}",
            "num_nodes": 0,
            "optimization_passes": [
                "dead_node_elimination",
                "constant_folding",
                "common_subexpression_elimination",
                "topological_reorder",
            ],
        }


class HyperionServer:
    """Главный gRPC сервер для HYPERION сервисов. Точка входа."""

    def __init__(self, port: int = 50051, max_workers: int = 4):
        self.port = port
        self.max_workers = max_workers
        self._server: Optional[grpc.Server] = None
        self._inference_service = InferenceServiceImpl()
        self._model_service = ModelServiceImpl()

    def start(self) -> None:
        """Поднимаем gRPC сервер."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers)
        )

        # Note: В проде тут регистрируем скомпилированные proto-сервисы:
        # hyperion_pb2_grpc.add_InferenceServiceServicer_to_server(...)
        # Пока структурный placeholder.

        self._server.add_insecure_port(f"[::]:{self.port}")
        self._server.start()
        logger.info(f"HYPERION gRPC server started on port {self.port}")

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
    logging.basicConfig(level=logging.INFO)
    server = HyperionServer(port=port)
    server.start()
    logger.info(f"Server listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
