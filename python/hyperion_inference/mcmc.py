"""MCMC — high-level API для запуска MCMC в HYPERION.

Аналог numpyro.infer.MCMC: оборачивает kernel (HMC/NUTS) в удобный
интерфейс с multi-chain, warmup, диагностикой.

    kernel = NUTS()
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=4)
    mcmc.run(rng_key, model, data)
    samples = mcmc.get_samples()
    mcmc.print_summary()
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from hyperion_backends.jax_backend import JAXBackend
from hyperion_inference.base import InferenceResult


class MCMC:
    """High-level MCMC runner.

    Скрывает backend initialization, config assembly, multi-chain logic.
    После run() — get_samples(), print_summary(), diagnostics.
    """

    def __init__(
        self,
        kernel: Any,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        dense_mass: bool = False,
        target_accept_rate: float = 0.8,
        max_tree_depth: int = 10,
        step_size: float = 0.01,
        num_leapfrog: int = 10,
        **extra_config,
    ):
        """Инициализация MCMC-раннера.

        Args:
            kernel: HMC/NUTS или другой kernel, реализующий InferenceEngine
            num_warmup: число шагов warmup (адаптация step_size)
            num_samples: число сэмплов после warmup
            num_chains: число независимых цепей
            dense_mass: использовать dense mass matrix (для сильно коррелированных параметров)
            target_accept_rate: целевая вероятность принятия для адаптации
            max_tree_depth: макс. глубина дерева NUTS
            step_size: начальный шаг (перезаписывается адаптацией)
            num_leapfrog: шагов leapfrog для HMC
            **extra_config: доп. параметры в config для kernel
        """
        self.kernel = kernel
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self._config = {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
            "dense_mass": dense_mass,
            "target_accept_rate": target_accept_rate,
            "max_tree_depth": max_tree_depth,
            "step_size": step_size,
            "num_leapfrog": num_leapfrog,
            **extra_config,
        }
        self._result: Optional[InferenceResult] = None
        self._backend: Optional[JAXBackend] = None

    def run(
        self,
        rng_key: jax.random.PRNGKey,
        model_fn: Any,
        data: Optional[dict[str, jnp.ndarray]] = None,
    ) -> None:
        """Запуск MCMC: warmup + sampling.

        Инициализирует backend, прогоняет warmup для адаптации, затем собирает сэмплы.
        Результат сохраняется внутри — дальше get_samples(), print_summary().

        Args:
            rng_key: JAX PRNG ключ
            model_fn: модель, обёрнутая в @model
            data: dict наблюдений {"obs_name": jnp.array}, можно пустой
        """
        if data is None:
            data = {}

        self._backend = JAXBackend()
        init_key, sample_key = jrandom.split(rng_key)
        self._backend.initialize(model_fn, data, init_key)
        self._result = self.kernel.run(self._backend, sample_key, self._config)

    def get_samples(self, group: str = "posterior") -> dict[str, np.ndarray]:
        """Все сэмплы, склеенные по цепям в один массив.

        Args:
            group: пока не используется, оставлен для совместимости с numpyro

        Returns:
            dict[str, np.ndarray] — имя параметра -> массив (num_chains*num_samples, ...)
        """
        if self._result is None:
            raise RuntimeError("Сначала вызови run()")
        return {name: np.asarray(v) for name, v in self._result.samples.items()}

    def get_samples_by_chain(self) -> Optional[dict[str, np.ndarray]]:
        """Сэмплы с сохранённой структурой по цепям.

        Returns:
            dict[str, np.ndarray] — имя -> (num_chains, num_samples, ...), или None,
            если kernel не вернул samples_by_chain
        """
        if self._result is None:
            raise RuntimeError("Сначала вызови run()")
        if self._result.samples_by_chain is None:
            return None
        return {name: np.asarray(v) for name, v in self._result.samples_by_chain.items()}

    @property
    def result(self) -> InferenceResult:
        if self._result is None:
            raise RuntimeError("Сначала вызови run()")
        return self._result

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self.result.diagnostics

    def print_summary(self, prob: float = 0.9) -> None:
        """Печатает summary-таблицу: mean, std, median, CI, ESS, R-hat.

        Args:
            prob: уровень для доверительного интервала (по умолчанию 90%)
        """
        from hyperion_diagnostics.metrics import (
            summary_table, effective_sample_size_multichain, r_hat,
        )

        samples = self.get_samples()
        by_chain = self.get_samples_by_chain()
        table = summary_table(samples, prob=prob, samples_by_chain=by_chain)

        alpha = (1.0 - prob) / 2.0
        ci_lo_key = f"ci_{alpha:.1%}"
        ci_hi_key = f"ci_{1 - alpha:.1%}"

        header = f"{'param':>20s} {'mean':>10s} {'std':>10s} {'median':>10s}"
        header += f" {ci_lo_key:>10s} {ci_hi_key:>10s} {'ess':>8s}"
        if by_chain is not None:
            header += f" {'r_hat':>8s}"
        print(header)
        print("-" * len(header))

        for name, stats in table.items():
            row = f"{name:>20s}"
            row += f" {stats['mean']:10.3f}"
            row += f" {stats['std']:10.3f}"
            row += f" {stats['median']:10.3f}"
            row += f" {stats.get(ci_lo_key, float('nan')):10.3f}"
            row += f" {stats.get(ci_hi_key, float('nan')):10.3f}"
            row += f" {stats['ess']:8.0f}"
            if by_chain is not None:
                base_name = name.split("[")[0]
                dim_suffix = name[len(base_name):] if "[" in name else ""
                if base_name in by_chain:
                    arr = np.asarray(by_chain[base_name])
                    if arr.ndim == 2:
                        rh = r_hat(arr)
                    elif arr.ndim == 3 and dim_suffix:
                        d = int(dim_suffix.strip("[]"))
                        rh = r_hat(arr[:, :, d])
                    else:
                        rh = float("nan")
                    row += f" {rh:8.3f}"
                else:
                    row += f" {'N/A':>8s}"
            print(row)

        diag = self.diagnostics
        print()

        def _scalar(v):
            v = np.asarray(v)
            return float(np.mean(v)) if v.ndim > 0 else float(v)

        def _count(v):
            v = np.asarray(v)
            return int(np.sum(v)) if v.ndim > 0 else int(v)

        if "mean_accept_prob" in diag:
            print(f"Mean accept prob: {_scalar(diag['mean_accept_prob']):.3f}")
        if "num_divergences" in diag:
            print(f"Divergences: {_count(diag['num_divergences'])}")
        if "mean_tree_depth" in diag:
            print(f"Mean tree depth: {_scalar(diag['mean_tree_depth']):.1f}")
        if "num_max_treedepth" in diag:
            print(f"Max treedepth hits: {_count(diag['num_max_treedepth'])}")
