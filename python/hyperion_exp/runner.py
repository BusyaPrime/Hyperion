"""Experiment Runner — запуск и сравнение инференс-движков.

Берёт модель, прогоняет через N движков, сравнивает результаты.
Выбирает лучший по ELBO/ESS/acceptance rate. Логирует всё.

Для тех кто не хочет руками запускать HMC, потом NUTS, потом VI
и вручную сравнивать — автоматизация спасает.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from pydantic import BaseModel

from hyperion_inference.base import InferenceResult

logger = logging.getLogger(__name__)


class ExperimentConfig(BaseModel):
    """Конфиг для инференс-эксперимента. Всё, что нужно, чтобы воспроизвести запуск."""
    model_name: str = "unnamed"
    inference_method: str = "nuts"
    seed: int = 42
    output_dir: str = "./experiments"

    # Конфиг инференса (пробрасываем в движок)
    num_samples: int = 1000
    num_warmup: int = 500
    step_size: float = 0.01
    num_leapfrog_steps: int = 10
    max_tree_depth: int = 10
    target_accept_rate: float = 0.8

    # VI-специфичное
    num_steps: int = 5000
    learning_rate: float = 0.01
    covariance_type: str = "diagonal"

    # SMC-специфичное
    num_particles: int = 1000
    num_tempering_steps: int = 20

    # Flows-специфичное
    num_layers: int = 6
    hidden_dim: int = 64
    flow_type: str = "realnvp"

    model_config = {"extra": "allow"}

    def to_inference_config(self) -> dict[str, Any]:
        return self.model_dump(exclude={"model_name", "inference_method", "seed", "output_dir"})


@dataclass
class ExperimentResult:
    """Результаты завершённого эксперимента. Всё, что наваяли — samples, отчёты, артифакты."""
    config: ExperimentConfig
    inference_result: Any = None
    diagnostics_report: Any = None
    elapsed_seconds: float = 0.0
    output_dir: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)

    def save(self, path: Optional[str] = None) -> str:
        save_dir = path or self.output_dir
        os.makedirs(save_dir, exist_ok=True)

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            f.write(self.config.model_dump_json(indent=2))
        self.artifacts["config"] = config_path

        if self.inference_result is not None and self.inference_result.samples:
            samples_path = os.path.join(save_dir, "samples.npz")
            np.savez(
                samples_path,
                **{k: np.asarray(v) for k, v in self.inference_result.samples.items()},
            )
            self.artifacts["samples"] = samples_path

        if self.diagnostics_report is not None:
            report_path = os.path.join(save_dir, "report.md")
            with open(report_path, "w") as f:
                f.write(self.diagnostics_report.to_markdown())
            self.artifacts["report"] = report_path

            report_json_path = os.path.join(save_dir, "report.json")
            with open(report_json_path, "w") as f:
                f.write(self.diagnostics_report.to_json())
            self.artifacts["report_json"] = report_json_path

        manifest = {
            "model_name": self.config.model_name,
            "inference_method": self.config.inference_method,
            "seed": self.config.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "artifacts": self.artifacts,
            "timestamp": datetime.now().isoformat(),
        }
        manifest_path = os.path.join(save_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return save_dir


@dataclass
class RunResult:
    """Результат одного запуска: engine name, результат, время, метрики."""

    engine_name: str
    result: InferenceResult
    elapsed_seconds: float
    summary: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Запускает инференс-эксперименты с трекингом и воспроизводимостью. Главный оркестратор.

    Поддерживает два режима:
    1) run() / run_comparison() — высокоуровневый API с model_fn, data, артифактами.
    2) add_engine() + run_all() + best_result() — низкоуровневый API для сравнения
       нескольких движков на одном backend, выбор лучшего по метрике, print_comparison().
    """

    def __init__(self, base_output_dir: str = "./experiments"):
        self._base_output_dir = base_output_dir
        self._engines: list[tuple[str, Any, dict]] = []

    def add_engine(self, name: str, engine: Any, config: dict[str, Any]) -> None:
        """Добавить движок для сравнения."""
        self._engines.append((name, engine, config))

    def run_single(
        self,
        name: str,
        engine: Any,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> RunResult:
        """Запуск одного движка с замером времени."""
        start = time.perf_counter()
        result = engine.run(backend, rng_key, config)
        elapsed = time.perf_counter() - start

        summary = self._compute_summary(result)
        summary["elapsed_seconds"] = elapsed

        return RunResult(
            engine_name=name,
            result=result,
            elapsed_seconds=elapsed,
            summary=summary,
        )

    def run_all(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
    ) -> list[RunResult]:
        """Запуск всех добавленных движков последовательно."""
        results = []
        for i, (name, engine, config) in enumerate(self._engines):
            key_i = jrandom.fold_in(rng_key, i)
            run_result = self.run_single(name, engine, backend, key_i, config)
            results.append(run_result)
        return results

    def best_result(
        self,
        results: list[RunResult],
        metric: str = "accept_rate",
    ) -> RunResult:
        """Выбрать лучший результат по метрике.

        Метрики: accept_rate, ess_per_second, elbo.
        Если метрика не найдена — возвращаем первый результат.
        """
        scored = []
        for r in results:
            score = r.summary.get(metric, float("-inf"))
            if score is None:
                score = float("-inf")
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else results[0]

    def _compute_summary(self, result: InferenceResult) -> dict[str, Any]:
        """Вычислить сводные метрики из результата."""
        summary: dict[str, Any] = {}

        if result.diagnostics:
            diag = result.diagnostics
            if "sample_accept_rate" in diag:
                summary["accept_rate"] = float(diag["sample_accept_rate"])
            elif "warmup_accept_rate" in diag:
                summary["accept_rate"] = float(diag["warmup_accept_rate"])
            if "best_elbo" in diag:
                summary["elbo"] = float(diag["best_elbo"])
            if "num_divergences" in diag:
                summary["num_divergences"] = int(diag["num_divergences"])
            if "log_evidence" in diag:
                summary["log_evidence"] = float(diag["log_evidence"])

        if result.samples:
            total_samples = 0
            for name, arr in result.samples.items():
                if hasattr(arr, "shape") and arr.ndim >= 1:
                    total_samples = arr.shape[0]
                    break
            summary["num_samples"] = total_samples

        return summary

    def print_comparison(self, results: list[RunResult]) -> str:
        """Человекочитаемая таблица сравнения."""
        lines = [
            "=" * 70,
            f"{'Движок':<15} {'Время (с)':<12} {'Accept Rate':<15} {'Samples':<10} {'Доп. метрики'}",
            "-" * 70,
        ]
        for r in results:
            accept = r.summary.get("accept_rate", "—")
            if isinstance(accept, float):
                accept = f"{accept:.4f}"
            n_samples = r.summary.get("num_samples", "—")
            extras = []
            if "elbo" in r.summary:
                extras.append(f"ELBO={r.summary['elbo']:.2f}")
            if "num_divergences" in r.summary:
                extras.append(f"div={r.summary['num_divergences']}")
            if "log_evidence" in r.summary:
                extras.append(f"logZ={r.summary['log_evidence']:.2f}")
            extra_str = ", ".join(extras) if extras else "—"
            lines.append(
                f"{r.engine_name:<15} {r.elapsed_seconds:<12.3f} {str(accept):<15} {str(n_samples):<10} {extra_str}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    def _get_engine(self, method: str):
        from hyperion_inference.hmc import HMCKernel
        from hyperion_inference.nuts import NUTSKernel
        from hyperion_inference.smc import SMCEngine
        from hyperion_inference.vi import VIEngine
        from hyperion_inference.flows import FlowsEngine
        from hyperion_inference.laplace import LaplaceApproximation

        engines = {
            "hmc": HMCKernel,
            "nuts": NUTSKernel,
            "smc": SMCEngine,
            "vi": VIEngine,
            "flows": FlowsEngine,
            "laplace": LaplaceApproximation,
        }
        engine_cls = engines.get(method.lower())
        if engine_cls is None:
            raise ValueError(f"Unknown inference method: {method}. Available: {list(engines.keys())}")
        return engine_cls()

    def run(
        self,
        model_fn: Callable,
        data: dict[str, Any],
        config: Optional[ExperimentConfig] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Запускает полный инференс-эксперимент.

        Args:
            model_fn: Функция вероятностной модели.
            data: Словарь с наблюдаемыми данными.
            config: Конфиг эксперимента.
            **kwargs: Переопределение полей конфига.

        Returns:
            ExperimentResult с samples, диагностикой и артифактами.
        """
        if config is None:
            config = ExperimentConfig(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        # Создаём выходную директорию
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = hashlib.sha256(
            f"{config.model_name}_{config.seed}_{timestamp}".encode()
        ).hexdigest()[:8]
        output_dir = os.path.join(
            self._base_output_dir,
            config.model_name,
            f"{config.inference_method}_{run_id}",
        )
        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            f"Starting experiment: model={config.model_name}, "
            f"method={config.inference_method}, seed={config.seed}"
        )

        # Инициализируем бэкенд
        from hyperion_backends.jax_backend import JAXBackend

        rng_key = jrandom.PRNGKey(config.seed)
        backend = JAXBackend()
        init_key, run_key = jrandom.split(rng_key)

        jax_data = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in data.items()}
        backend.initialize(model_fn, jax_data, init_key)

        # Запускаем инференс
        engine = self._get_engine(config.inference_method)
        inference_config = config.to_inference_config()

        start_time = time.time()
        inference_result = engine.run(backend, run_key, inference_config)
        elapsed = time.time() - start_time

        logger.info(f"Inference completed in {elapsed:.2f}s")

        # Генерим диагностический отчёт
        from hyperion_diagnostics.report import generate_report

        report = generate_report(
            inference_result,
            model_name=config.model_name,
            inference_method=config.inference_method,
            config=inference_config,
        )

        result = ExperimentResult(
            config=config,
            inference_result=inference_result,
            diagnostics_report=report,
            elapsed_seconds=elapsed,
            output_dir=output_dir,
        )

        result.save()

        logger.info(f"Artifacts saved to {output_dir}")
        return result

    def run_comparison(
        self,
        model_fn: Callable,
        data: dict[str, Any],
        methods: list[str],
        base_config: Optional[ExperimentConfig] = None,
    ) -> dict[str, ExperimentResult]:
        """Запускает одну и ту же модель разными методами инференса для сравнения.

        Args:
            model_fn: Функция модели.
            data: Данные.
            methods: Список методов (nuts, hmc, vi, ...).
            base_config: Базовый конфиг.

        Returns:
            Словарь метод -> ExperimentResult.
        """
        results = {}
        for method in methods:
            cfg = base_config or ExperimentConfig()
            cfg.inference_method = method
            try:
                results[method] = self.run(model_fn, data, config=cfg)
            except Exception as e:
                logger.error(f"Method {method} failed: {e}")
                results[method] = None
        return results
