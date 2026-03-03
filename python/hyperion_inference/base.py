"""Базовые классы для inference-движков.

Здесь живут InferenceState, InferenceResult и абстрактный InferenceEngine.

InferenceEngine — единый контракт: run(backend, rng_key, config) -> InferenceResult.
initialize/step/get_samples/get_metrics — опционально, для пошагового API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import jax
import jax.numpy as jnp


@dataclass
class InferenceState:
    """Базовое состояние инференса — шаг, rng. Наследники добавляют своё."""
    step: int = 0
    rng_key: Any = None


@dataclass
class InferenceResult:
    """Результат инференса: сэмплы, log_probs, диагностика.

    Возвращается из kernel.run(). MCMC.get_samples() — это просто обёртка над
    result.samples. samples_by_chain — если kernel сохранил структуру по цепям,
    иначе None. diagnostics — mean_accept_prob, num_divergences и т.п.
    """

    samples: dict[str, jnp.ndarray] = field(default_factory=dict)
    log_probs: Optional[jnp.ndarray] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    num_chains: int = 1
    samples_by_chain: Optional[dict[str, jnp.ndarray]] = None

    @property
    def num_samples(self) -> int:
        if self.samples:
            first_key = next(iter(self.samples))
            return self.samples[first_key].shape[0]
        return 0


class InferenceEngine(ABC):
    """Абстрактный базовый класс для всех inference-движков.

    Единственный обязательный метод — run().
    initialize/step/get_samples/get_metrics — для движков с пошаговым API (SMC, VI).
    HMC/NUTS/Laplace используют только run().
    """

    @abstractmethod
    def run(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> InferenceResult:
        """Полный прогон инференса: warmup + sampling. Единая точка входа."""
        ...

    def initialize(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> InferenceState:
        """Инициализация состояния. Опционально для пошагового API."""
        raise NotImplementedError(
            f"{type(self).__name__} не поддерживает пошаговый API — используйте run()"
        )

    def step(self, state: InferenceState) -> InferenceState:
        """Один шаг инференса. Опционально для пошагового API."""
        raise NotImplementedError(
            f"{type(self).__name__} не поддерживает пошаговый API — используйте run()"
        )

    def get_samples(self, state: InferenceState) -> dict[str, jnp.ndarray]:
        """Вытаскиваем сэмплы из state. Опционально для пошагового API."""
        raise NotImplementedError(
            f"{type(self).__name__} не поддерживает get_samples() — используйте run()"
        )

    def get_metrics(self, state: InferenceState) -> dict[str, Any]:
        """Диагностика. Опционально для пошагового API."""
        raise NotImplementedError(
            f"{type(self).__name__} не поддерживает get_metrics() — используйте run()"
        )
