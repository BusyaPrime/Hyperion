"""Биективные трансформации (bijectors) для HYPERION PPL.

Каждый transform умеет:
  - forward(x): из неограниченного пространства в ограниченное
  - inverse(y): обратно
  - log_abs_det_jacobian(x, y): лог|det(df/dx)| для change-of-variables

Это критически важный модуль: без корректных якобианов
инференс будет выдавать чушь (а ты даже не заметишь).

Правило: если сомневаешься в якобиане — проверь автодиффом.
jax.jacobian(transform.forward)(x) vs transform.log_abs_det_jacobian(x, y)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp

# Магическая константа: меньше — крашнёмся на нулях, больше — теряем точность
_EPS = 1e-8


class Transform(ABC):
    """Базовый класс для всех биективных трансформаций.

    forward: unconstrained -> constrained
    inverse: constrained -> unconstrained
    log_abs_det_jacobian: для поправки плотности при замене переменных
    """

    @abstractmethod
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Прямое преобразование: из R^n в целевое пространство."""
        ...

    @abstractmethod
    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        """Обратное преобразование: из целевого пространства в R^n."""
        ...

    @abstractmethod
    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Лог абсолютного определителя якобиана df/dx.

        Это слагаемое в log_prob при замене переменных.
        Забудешь его — получишь неправильный постериор.
        """
        ...

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)

    def __eq__(self, other: object) -> bool:
        if type(self) != type(other):
            return False
        return self._eq_impl(other)

    def _eq_impl(self, other: Transform) -> bool:
        """Переопредели в наследниках для сравнения атрибутов."""
        return True


class IdentityTransform(Transform):
    """Тождественное отображение. y = x, якобиан = 0.

    Используется для неограниченных параметров (constraint=Real).
    Максимально скучная трансформация, но кто-то должен делать и это.
    """

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return y

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)

    def __repr__(self) -> str:
        return "IdentityTransform()"


class ExpTransform(Transform):
    """z = exp(u). Отображает R -> R+.

    Якобиан: d/du exp(u) = exp(u), значит log|det J| = u.
    Классика для positive constraint. Просто и честно.
    """

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(x)

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        # Клипаем чтобы не крашнуться на log(0)
        return jnp.log(jnp.maximum(y, _EPS))

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x

    def __repr__(self) -> str:
        return "ExpTransform()"


class SoftplusTransform(Transform):
    """z = log(1 + exp(u)). Гладкая альтернатива exp для R -> R+.

    Более численно стабильная чем exp при больших u,
    но якобиан чуть сложнее: log(sigmoid(u)).
    """

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.softplus(x)

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        y = jnp.maximum(y, _EPS)
        return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(y) + _EPS))

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return -jax.nn.softplus(-x)

    def __repr__(self) -> str:
        return "SoftplusTransform()"


class SigmoidTransform(Transform):
    """z = sigmoid(u). Отображает R -> (0, 1).

    Якобиан: sigmoid(u) * (1 - sigmoid(u))
    В лог-домене: -softplus(-u) - softplus(u)

    Используется для UnitInterval и как часть BoundedTransform.
    """

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.sigmoid(x)

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        # logit(y) = log(y/(1-y)) — клипаем чтобы не крашнуться на границах
        y_safe = jnp.clip(y, _EPS, 1.0 - _EPS)
        return jnp.log(y_safe) - jnp.log(1.0 - y_safe)

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return -jax.nn.softplus(-x) - jax.nn.softplus(x)

    def __repr__(self) -> str:
        return "SigmoidTransform()"


class AffineTransform(Transform):
    """z = loc + scale * u. Сдвиг и масштабирование.

    Якобиан: |scale| (константа, не зависит от u).
    Простейшая параметрическая трансформация.
    """

    def __init__(self, loc: Any = 0.0, scale: Any = 1.0):
        self.loc = jnp.asarray(loc, dtype=jnp.float32)
        self.scale = jnp.asarray(scale, dtype=jnp.float32)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.loc + self.scale * x

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return (y - self.loc) / (self.scale + _EPS)

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.full_like(x, jnp.log(jnp.abs(self.scale) + _EPS))

    def _eq_impl(self, other: Transform) -> bool:
        return (
            bool(jnp.allclose(self.loc, other.loc))
            and bool(jnp.allclose(self.scale, other.scale))
        )

    def __repr__(self) -> str:
        return f"AffineTransform(loc={self.loc}, scale={self.scale})"


class BoundedTransform(Transform):
    """Отображает R -> (lower, upper) через sigmoid + affine.

    По сути это: y = lower + (upper - lower) * sigmoid(x)
    Якобиан складывается из sigmoid-части и масштабирования.
    """

    def __init__(self, lower: float, upper: float):
        self.lower = jnp.float32(lower)
        self.upper = jnp.float32(upper)
        self._sigmoid = SigmoidTransform()

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        s = self._sigmoid.forward(x)
        return self.lower + (self.upper - self.lower) * s

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        # Клипаем чтобы не вылететь за границы
        y_safe = jnp.clip(y, self.lower + _EPS, self.upper - _EPS)
        s = (y_safe - self.lower) / (self.upper - self.lower + _EPS)
        return self._sigmoid.inverse(s)

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        s = self._sigmoid.forward(x)
        return (
            self._sigmoid.log_abs_det_jacobian(x, s)
            + jnp.log(self.upper - self.lower + _EPS)
        )

    def _eq_impl(self, other: Transform) -> bool:
        return (
            bool(jnp.allclose(self.lower, other.lower))
            and bool(jnp.allclose(self.upper, other.upper))
        )

    def __repr__(self) -> str:
        return f"BoundedTransform(lower={self.lower}, upper={self.upper})"


class StickBreakingTransform(Transform):
    """Отображает R^{K-1} -> simplex(K) через stick-breaking.

    Представь что у тебя палка длины 1. Ты отламываешь кусок z_1,
    от остатка отламываешь z_2, и так далее. Последний кусок — сколько осталось.

    Математически: x_k = z_k * prod_{j<k}(1 - z_j), z_k = sigmoid(u_k - offset)
    Offset нужен чтобы центрировать на равномерном распределении.
    """

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        K_minus_1 = x.shape[-1]
        # Оффсет для центрирования: log(K-1), log(K-2), ..., log(1)
        offset = jnp.log(jnp.arange(K_minus_1, 0, -1, dtype=x.dtype))
        z = jax.nn.sigmoid(x - offset)
        z_padded = jnp.concatenate(
            [z, jnp.ones(x.shape[:-1] + (1,), dtype=x.dtype)], axis=-1
        )
        cumprod = jnp.concatenate(
            [
                jnp.ones(x.shape[:-1] + (1,), dtype=x.dtype),
                jnp.cumprod(1.0 - z, axis=-1),
            ],
            axis=-1,
        )
        return z_padded * cumprod

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        y_trimmed = y[..., :-1]
        remainder = 1.0 - jnp.cumsum(y_trimmed, axis=-1) + y_trimmed
        z = y_trimmed / jnp.maximum(remainder, _EPS)
        z = jnp.clip(z, _EPS, 1.0 - _EPS)
        offset = jnp.log(
            jnp.arange(y_trimmed.shape[-1], 0, -1, dtype=y.dtype)
        )
        return jnp.log(z) - jnp.log(1.0 - z) + offset

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Якобиан stick-breaking — формула из Stan math library."""
        n = x.shape[-1]
        offset = jnp.log(jnp.arange(n, 0, -1, dtype=x.dtype))
        s = jax.nn.sigmoid(x - offset)
        log_jac = (
            jnp.log(s + _EPS)
            + jnp.log(1.0 - s + _EPS)
            + jnp.concatenate(
                [
                    jnp.zeros(x.shape[:-1] + (1,), dtype=x.dtype),
                    jnp.cumsum(jnp.log(1.0 - s[..., :-1] + _EPS), axis=-1),
                ],
                axis=-1,
            )
        )
        return jnp.sum(log_jac, axis=-1)

    def __repr__(self) -> str:
        return "StickBreakingTransform()"


class CholeskyTransform(Transform):
    """Отображает вектор R^{n*(n+1)/2} в нижнетреугольную матрицу с положительной диагональю.

    Диагональные элементы — через exp() (чтобы гарантировать > 0).
    Вне-диагональные — без ограничений.

    Якобиан = сумма диагональных элементов unconstrained вектора
    (потому что exp на диагонали вносит якобиан = значение).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._n_tril = dim * (dim + 1) // 2
        # Diagonal (i,i) sits at packed index i*(i+1)//2 + i in row-major tril order
        self._diag_indices = jnp.array(
            [i * (i + 1) // 2 + i for i in range(dim)], dtype=jnp.int32
        )

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        L = jnp.zeros(x.shape[:-1] + (self.dim, self.dim), dtype=x.dtype)
        idx = jnp.tril_indices(self.dim)
        L = L.at[..., idx[0], idx[1]].set(x[..., : self._n_tril])
        diag_idx = jnp.diag_indices(self.dim)
        L = L.at[..., diag_idx[0], diag_idx[1]].set(
            jnp.exp(L[..., diag_idx[0], diag_idx[1]])
        )
        return L

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        L = y.copy()
        diag_idx = jnp.diag_indices(self.dim)
        L = L.at[..., diag_idx[0], diag_idx[1]].set(
            jnp.log(jnp.maximum(L[..., diag_idx[0], diag_idx[1]], _EPS))
        )
        idx = jnp.tril_indices(self.dim)
        return L[..., idx[0], idx[1]]

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        diag_x = x[..., self._diag_indices]
        return jnp.sum(diag_x, axis=-1)

    def _eq_impl(self, other: Transform) -> bool:
        return self.dim == other.dim

    def __repr__(self) -> str:
        return f"CholeskyTransform(dim={self.dim})"


class CorrCholeskyTransform(Transform):
    """Отображает R^{n*(n-1)/2} в Cholesky-фактор корреляционной матрицы.

    Через tanh получаем частичные корреляции в (-1, 1),
    потом stick-breaking-подобная конструкция строит L.

    Тут водятся драконы: якобиан этой штуки — отдельная история.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._n_offdiag = dim * (dim - 1) // 2

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        z = jnp.tanh(x)
        L = jnp.zeros(x.shape[:-1] + (self.dim, self.dim), dtype=x.dtype)
        L = L.at[..., 0, 0].set(1.0)
        k = 0
        for i in range(1, self.dim):
            remaining = jnp.ones(x.shape[:-1], dtype=x.dtype)
            for j in range(i):
                L = L.at[..., i, j].set(
                    z[..., k] * jnp.sqrt(jnp.maximum(remaining, _EPS))
                )
                remaining = remaining * (1.0 - z[..., k] ** 2)
                k += 1
            L = L.at[..., i, i].set(jnp.sqrt(jnp.maximum(remaining, _EPS)))
        return L

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        x_list = []
        for i in range(1, self.dim):
            remaining = jnp.ones(y.shape[:-2], dtype=y.dtype)
            for j in range(i):
                z_ij = y[..., i, j] / jnp.sqrt(jnp.maximum(remaining, _EPS))
                z_ij = jnp.clip(z_ij, -1.0 + _EPS, 1.0 - _EPS)
                x_list.append(jnp.arctanh(z_ij))
                remaining = remaining * (1.0 - z_ij**2)
        return jnp.stack(x_list, axis=-1)

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        z = jnp.tanh(x)
        # tanh вносит log(1 - z^2) на каждый параметр
        log_det = jnp.sum(jnp.log(jnp.maximum(1.0 - z**2, _EPS)), axis=-1)
        # Плюс sqrt(remaining) вносит 0.5 * log(remaining)
        k = 0
        for i in range(1, self.dim):
            remaining = jnp.ones(x.shape[:-1], dtype=x.dtype)
            for j in range(i):
                log_det = log_det + 0.5 * jnp.log(jnp.maximum(remaining, _EPS))
                remaining = remaining * (1.0 - z[..., k] ** 2)
                k += 1
        return log_det

    def _eq_impl(self, other: Transform) -> bool:
        return self.dim == other.dim

    def __repr__(self) -> str:
        return f"CorrCholeskyTransform(dim={self.dim})"


class ComposeTransform(Transform):
    """Композиция нескольких трансформаций: f_n(...(f_2(f_1(x)))).

    Якобиан: сумма log|det J_i| по цепочке (chain rule).
    Порядок: parts[0] применяется первым.
    """

    def __init__(self, parts: list[Transform]):
        self.parts = parts

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        for part in self.parts:
            x = part.forward(x)
        return x

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        for part in reversed(self.parts):
            y = part.inverse(y)
        return y

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        result = jnp.zeros(x.shape[:-1] if x.ndim > 0 else (), dtype=x.dtype)
        current = x
        for part in self.parts:
            next_val = part.forward(current)
            result = result + part.log_abs_det_jacobian(current, next_val)
            current = next_val
        return result

    def _eq_impl(self, other: Transform) -> bool:
        if len(self.parts) != len(other.parts):
            return False
        return all(p == o for p, o in zip(self.parts, other.parts))

    def __repr__(self) -> str:
        parts_str = ", ".join(repr(p) for p in self.parts)
        return f"ComposeTransform([{parts_str}])"


class PowerTransform(Transform):
    """z = x^power. Для положительных x.

    power=0.5 — корень, power=2 — квадрат.
    Якобиан: |power| * x^(power-1).
    """

    def __init__(self, power: float):
        if power == 0:
            raise ValueError("PowerTransform: power не может быть 0 (деление на ноль в inverse)")
        self.power = jnp.float32(power)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.power(jnp.maximum(x, _EPS), self.power)

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.power(jnp.maximum(y, _EPS), 1.0 / self.power)

    def log_abs_det_jacobian(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return (
            jnp.log(jnp.abs(self.power) + _EPS)
            + (self.power - 1.0) * jnp.log(jnp.abs(x) + _EPS)
        )

    def _eq_impl(self, other: Transform) -> bool:
        return bool(jnp.allclose(self.power, other.power))

    def __repr__(self) -> str:
        return f"PowerTransform(power={self.power})"


def biject_to(constraint: Any) -> Transform:
    """Подобрать трансформацию для данного constraint.

    Real -> Identity, Positive -> Exp, UnitInterval -> Sigmoid, и т.д.

    Если для твоего constraint нет дефолтной трансформации —
    напиши свою и не жалуйся.
    """
    from hyperion_dsl.constraints import (
        Bounded,
        CorrCholesky,
        Positive,
        Real,
        Simplex,
        UnitInterval,
    )

    if isinstance(constraint, Real):
        return IdentityTransform()
    if isinstance(constraint, Positive):
        return ExpTransform()
    if isinstance(constraint, UnitInterval):
        return SigmoidTransform()
    if isinstance(constraint, Bounded):
        return BoundedTransform(constraint.lower, constraint.upper)
    if isinstance(constraint, Simplex):
        return StickBreakingTransform()
    if isinstance(constraint, CorrCholesky):
        if hasattr(constraint, '_dim') and constraint._dim is not None:
            return CorrCholeskyTransform(constraint._dim)
        raise ValueError(
            "CorrCholesky constraint needs dimension. "
            "Use corr_cholesky_constraint(dim) or CorrCholeskyTransform(dim) directly."
        )
    raise ValueError(f"Нет дефолтной трансформации для {constraint}. Пиши свою.")
