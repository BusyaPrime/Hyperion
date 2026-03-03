"""Базовый интерфейс бэкенда — абстракция над вычислениями.

Бэкенд знает как считать log_joint, градиенты, семплить из приора.
Инференс-движки работают ТОЛЬКО через этот интерфейс — им плевать
что внутри: JAX, PyTorch, нумпай на коленке.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


class Backend(ABC):
    """Абстрактный бэкенд. Инференс-движки видят только этот интерфейс."""

    @abstractmethod
    def initialize(
        self,
        model_fn: Any,
        data: dict[str, jnp.ndarray],
        rng_key: Any,
    ) -> None:
        """Инициализация: трассировка модели, построение вычислительного графа."""
        ...

    @abstractmethod
    def sample_prior(self, rng_key: Any) -> dict[str, jnp.ndarray]:
        """Засемплить латенты из приора."""
        ...

    @abstractmethod
    def flatten_latents(self, latent_values: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """dict латентов → плоский вектор R^d."""
        ...

    @abstractmethod
    def unflatten_latents(self, flat: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Плоский вектор R^d → dict латентов."""
        ...

    @property
    @abstractmethod
    def potential_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """JIT-скомпилированная log p(data, z) как функция от плоского вектора z.
        
        Это ГЛАВНЫЙ интерфейс для инференс-движков. Один вызов — один скаляр.
        """
        ...

    @property
    @abstractmethod
    def potential_and_grad_fn(self) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
        """JIT-скомпилированная (log_p, grad_log_p). Для HMC/NUTS — основная рабочая лошадка."""
        ...

    @property
    @abstractmethod
    def grad_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """JIT-скомпилированный градиент log p(data, z)."""
        ...

    @property
    @abstractmethod
    def total_dim(self) -> int:
        """Суммарная размерность латентного пространства (в unconstrained)."""
        ...

    @abstractmethod
    def get_latent_shapes(self) -> dict[str, tuple[int, ...]]:
        """Словарь: имя латента → его shape."""
        ...
