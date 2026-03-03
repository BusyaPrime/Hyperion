"""Ограничения (constraints) для случайных величин и параметров.

Каждый constraint умеет:
- check() — проверить что значение в допустимой области
- feasible_like() — выдать "безопасную" точку нужной формы

Используются bijectors (transforms.py) чтобы отображать
неограниченное пространство R^n в нужную область. Курс теории вероятностей НИУ ИТМО.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax.numpy as jnp


class Constraint(ABC):
    """Базовый класс ограничений. Все ограничения наследуются от него."""

    @abstractmethod
    def check(self, value: jnp.ndarray) -> jnp.ndarray:
        """Вернуть булев массив: True где значение удовлетворяет ограничению."""
        ...

    @abstractmethod
    def feasible_like(self, value: jnp.ndarray) -> jnp.ndarray:
        """Вернуть допустимую точку такой же формы (для инициализации)."""
        ...


class Positive(Constraint):
    """x > 0. Классика: дисперсии, scale-параметры, всё что не может быть отрицательным."""

    def check(self, value: jnp.ndarray) -> jnp.ndarray:
        return value > 0

    def feasible_like(self, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(value)

    def __repr__(self) -> str:
        return "Positive()"


class Bounded(Constraint):
    """lower <= x <= upper. Когда параметр живёт в ящике."""

    def __init__(self, lower: float = float("-inf"), upper: float = float("inf")):
        if lower >= upper:
            raise ValueError(f"lower ({lower}) должен быть меньше upper ({upper}), это же очевидно")
        self.lower = lower
        self.upper = upper

    def check(self, value: jnp.ndarray) -> jnp.ndarray:
        return (value >= self.lower) & (value <= self.upper)

    def feasible_like(self, value: jnp.ndarray) -> jnp.ndarray:
        if jnp.isfinite(self.lower + self.upper):
            mid = (self.lower + self.upper) / 2.0
        else:
            mid = 0.0
        return jnp.full_like(value, mid)

    def __repr__(self) -> str:
        return f"Bounded(lower={self.lower}, upper={self.upper})"


class Simplex(Constraint):
    """x >= 0, sum(x) = 1. Симплекс — для категориальных вероятностей и Dirichlet."""

    def check(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.atleast_1d(value)
        return (value >= 0).all(axis=-1) & jnp.isclose(value.sum(axis=-1), 1.0, atol=1e-5)

    def feasible_like(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.atleast_1d(value)
        k = value.shape[-1]
        return jnp.broadcast_to(jnp.full_like(value, 1.0 / k), value.shape)

    def __repr__(self) -> str:
        return "Simplex()"


class CorrCholesky(Constraint):
    """Нижнетреугольная матрица — Холецки-фактор корреляционной матрицы.
    Диагональ положительная, строки нормированы. Для LKJ и прочей красоты."""

    def __init__(self, dim: int | None = None):
        self._dim = dim

    def check(self, value: jnp.ndarray) -> jnp.ndarray:
        diag_positive = (jnp.diagonal(value, axis1=-2, axis2=-1) > 0).all(axis=-1)
        lower = jnp.all(jnp.isclose(value, jnp.tril(value)), axis=(-2, -1))
        row_norms_sq = jnp.sum(value ** 2, axis=-1)
        unit_norm = jnp.all(jnp.isclose(row_norms_sq, 1.0, atol=1e-4), axis=-1)
        return diag_positive & lower & unit_norm

    def feasible_like(self, value: jnp.ndarray) -> jnp.ndarray:
        eye = jnp.eye(value.shape[-1], dtype=value.dtype)
        return jnp.broadcast_to(eye, value.shape)

    def __repr__(self) -> str:
        dim_str = f", dim={self._dim}" if self._dim is not None else ""
        return f"CorrCholesky({dim_str})"


class UnitInterval(Constraint):
    """0 <= x <= 1. Для вероятностей, Beta-параметров и прочего в [0, 1]."""

    def check(self, value: jnp.ndarray) -> jnp.ndarray:
        return (value >= 0.0) & (value <= 1.0)

    def feasible_like(self, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.full_like(value, 0.5)

    def __repr__(self) -> str:
        return "UnitInterval()"


class Real(Constraint):
    """Вся вещественная прямая. Без ограничений (ну почти — NaN не считается)."""

    def check(self, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.isfinite(value)

    def feasible_like(self, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(value)

    def __repr__(self) -> str:
        return "Real()"


class PositiveDefinite(Constraint):
    """Положительно-определённая матрица. Для ковариационных матриц.
    Проверяем через собственные значения — дорого, но надёжно."""

    def check(self, value: jnp.ndarray) -> jnp.ndarray:
        eigvals = jnp.linalg.eigvalsh(value)
        return (eigvals > 0).all(axis=-1)

    def feasible_like(self, value: jnp.ndarray) -> jnp.ndarray:
        n = value.shape[-1]
        return jnp.broadcast_to(jnp.eye(n, dtype=value.dtype), value.shape)

    def __repr__(self) -> str:
        return "PositiveDefinite()"


class LowerTriangular(Constraint):
    """Нижнетреугольная матрица (без ограничений на диагональ)."""

    def check(self, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.all(jnp.isclose(value, jnp.tril(value)), axis=(-2, -1))

    def feasible_like(self, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.broadcast_to(jnp.eye(value.shape[-1], dtype=value.dtype), value.shape)

    def __repr__(self) -> str:
        return "LowerTriangular()"


# Синглтоны — используй их вместо создания новых экземпляров каждый раз.
# Это и быстрее, и isinstance проверки работают корректно.
positive = Positive()
simplex = Simplex()
corr_cholesky = CorrCholesky()
unit_interval = UnitInterval()
real = Real()
positive_definite = PositiveDefinite()
lower_triangular = LowerTriangular()
bounded = Bounded


def corr_cholesky_constraint(dim: int) -> CorrCholesky:
    """CorrCholesky с известной размерностью — для автоматического biject_to."""
    return CorrCholesky(dim=dim)
