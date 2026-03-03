"""Библиотека вероятностных распределений HYPERION.

Тут живут все распределения — от банальной нормалки до LKJ Cholesky.
Каждое распределение умеет:
  - log_prob(value) — посчитать лог-плотность (это главная рабочая лошадка)
  - sample(key, shape) — засемплить через JAX PRNG
  - support — вернуть constraint (область определения)
  - batch_shape / event_shape — формы тензоров

Числовая стабильность — наш бог. Везде где можно — log-domain,
log1p, clipping, eps-гварды. Потому что float32 — это боль.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.special import gammaln, betaln

from hyperion_dsl.constraints import (
    Constraint, Real, Positive, Bounded, Simplex, CorrCholesky,
    real, positive, unit_interval, corr_cholesky,
)

Shape = Tuple[int, ...]

# Минимальный эпсилон для числовой стабильности.
# Если ты думаешь что можно обойтись без него — попробуй, посмотрим на твои NaN-ы.
_EPS = 1e-8
_LOG_2PI = jnp.log(2.0 * jnp.pi)


def _check_positive(value, name: str) -> None:
    try:
        if jnp.any(value <= 0):
            raise ValueError(f"{name} must be positive, got {value}")
    except jax.errors.TracerBoolConversionError:
        pass


class Distribution(ABC):
    """Базовый класс для всех вероятностных распределений HYPERION.

    Каждое распределение обязано реализовать:
    - support: Constraint — область определения (positive, real, simplex и т.д.)
    - log_prob(value): лог-плотность в точке value
    - sample(key, sample_shape): семплирование через JAX PRNG

    batch_shape — сколько независимых распределений в батче.
    event_shape — форма одного события (() для скаляра, (K,) для вектора).

    Args:
        batch_shape: форма батча, по умолчанию ().
        event_shape: форма события, по умолчанию ().
    """

    def __init__(self, batch_shape: Shape = (), event_shape: Shape = ()):
        self._batch_shape = batch_shape
        self._event_shape = event_shape

    @property
    def batch_shape(self) -> Shape:
        """Форма батча — "сколько независимых распределений мы несём"."""
        return self._batch_shape

    @property
    def event_shape(self) -> Shape:
        """Форма одного события — "()" для скаляров, "(K,)" для многомерных."""
        return self._event_shape

    @property
    def shape(self) -> Shape:
        """Полная форма = batch + event. Для отладки и shape inference."""
        return self._batch_shape + self._event_shape

    @property
    @abstractmethod
    def support(self) -> Constraint:
        """На чём живёт распределение (positive, real, simplex, etc.)."""
        ...

    @abstractmethod
    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """Лог-плотность. Это самый важный метод — от него кормятся все инференсы."""
        ...

    @abstractmethod
    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        """Засемплить значение. key — JAX PRNG ключ (не numpy.random!)."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normal(Distribution):
    """Нормальное распределение N(loc, scale).

    Самое используемое распределение во вселенной.
    Если сомневаешься какой prior ставить — ставь нормалку. Классика НИУ ИТМО.
    """

    def __init__(self, loc: Any = 0.0, scale: Any = 1.0):
        self.loc = jnp.asarray(loc, dtype=jnp.float32)
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        _check_positive(self.scale, "Normal.scale")
        batch_shape = jnp.broadcast_shapes(self.loc.shape, self.scale.shape)
        super().__init__(batch_shape=batch_shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return real

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        var = jnp.maximum(self.scale ** 2, _EPS)
        return -0.5 * (_LOG_2PI + jnp.log(var) + (value - self.loc) ** 2 / var)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape + self.event_shape
        eps = jrandom.normal(key, shape=shape)
        return self.loc + self.scale * eps

    @property
    def mean(self) -> jnp.ndarray:
        return self.loc

    @property
    def variance(self) -> jnp.ndarray:
        return self.scale ** 2

    @property
    def entropy(self) -> jnp.ndarray:
        """Энтропия нормалки: 0.5 * log(2*pi*e*sigma^2). Формула красивая, жизнь — нет."""
        return 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * self.scale ** 2)


class HalfNormal(Distribution):
    """Полу-нормальное распределение |N(0, scale)|.

    Живёт на (0, +inf). Классический prior для дисперсий и scale-параметров.
    Легче чем InverseGamma, честнее чем Uniform.
    """

    def __init__(self, scale: Any = 1.0):
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        _check_positive(self.scale, "HalfNormal.scale")
        super().__init__(batch_shape=self.scale.shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return positive

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        safe_scale = jnp.maximum(self.scale, _EPS)
        # log(2) + log_prob_normal(value; 0, scale)
        result = (jnp.log(2.0) - 0.5 * _LOG_2PI - jnp.log(safe_scale)
                - 0.5 * (value / safe_scale) ** 2)
        return jnp.where(value >= 0, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return jnp.abs(jrandom.normal(key, shape=shape)) * self.scale


class HalfCauchy(Distribution):
    """Полу-Коши |Cauchy(0, scale)|.

    Тяжёлые хвосты, живёт на (0, +inf). Канонический prior для horseshoe.
    Если тебе нужно что-то "более размытое чем HalfNormal" — бери это.
    """

    def __init__(self, scale: Any = 1.0):
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        _check_positive(self.scale, "HalfCauchy.scale")
        super().__init__(batch_shape=self.scale.shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return positive

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        safe_scale = jnp.maximum(self.scale, _EPS)
        result = (jnp.log(2.0) - jnp.log(jnp.pi) - jnp.log(safe_scale)
                - jnp.log1p((value / safe_scale) ** 2))
        return jnp.where(value >= 0, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return jnp.abs(self.scale * jrandom.cauchy(key, shape=shape))


class LogNormal(Distribution):
    """Лог-нормальное: если X ~ N(loc, scale), то exp(X) ~ LogNormal.

    Для моделирования величин, которые точно положительные
    и у которых правый хвост тяжелее левого (доходы, размеры, цены).
    """

    def __init__(self, loc: Any = 0.0, scale: Any = 1.0):
        self.loc = jnp.asarray(loc, dtype=jnp.float32)
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        _check_positive(self.scale, "LogNormal.scale")
        batch_shape = jnp.broadcast_shapes(self.loc.shape, self.scale.shape)
        super().__init__(batch_shape=batch_shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return positive

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        # Аккуратно: log(value) может быть -inf при value=0
        safe_value = jnp.maximum(value, _EPS)
        var = jnp.maximum(self.scale ** 2, _EPS)
        result = (-jnp.log(safe_value) - 0.5 * _LOG_2PI - 0.5 * jnp.log(var)
                - 0.5 * (jnp.log(safe_value) - self.loc) ** 2 / var)
        return jnp.where(value > 0, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return jnp.exp(self.loc + self.scale * jrandom.normal(key, shape=shape))


class Gamma(Distribution):
    """Гамма-распределение Gamma(concentration, rate).

    Для положительных величин с правым хвостом.
    concentration (она же alpha/shape) и rate (она же beta/1/scale).
    Не путай rate и scale — это классический источник багов.
    """

    def __init__(self, concentration: Any, rate: Any):
        self.concentration = jnp.asarray(concentration, dtype=jnp.float32)
        self.rate = jnp.asarray(rate, dtype=jnp.float32)
        _check_positive(self.concentration, "Gamma.concentration")
        _check_positive(self.rate, "Gamma.rate")
        batch_shape = jnp.broadcast_shapes(self.concentration.shape, self.rate.shape)
        super().__init__(batch_shape=batch_shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return positive

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        safe_value = jnp.maximum(value, _EPS)
        safe_rate = jnp.maximum(self.rate, _EPS)
        result = (self.concentration * jnp.log(safe_rate)
                - gammaln(self.concentration)
                + (self.concentration - 1.0) * jnp.log(safe_value)
                - self.rate * value)
        return jnp.where(value > 0, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        safe_rate = jnp.maximum(self.rate, _EPS)
        return jrandom.gamma(key, self.concentration, shape=shape) / safe_rate

    @property
    def mean(self) -> jnp.ndarray:
        return self.concentration / self.rate

    @property
    def variance(self) -> jnp.ndarray:
        return self.concentration / (self.rate ** 2)

    @property
    def entropy(self) -> jnp.ndarray:
        """Энтропия Гаммы — формула из учебника, но кто их вообще читает."""
        a = self.concentration
        return (a - jnp.log(jnp.maximum(self.rate, _EPS)) + gammaln(a)
                + (1.0 - a) * jax.scipy.special.digamma(a))


class Beta(Distribution):
    """Бета-распределение Beta(a, b) на [0, 1].

    Классика для моделирования вероятностей. a=b=1 это Uniform(0,1).
    a,b < 1 — U-образная, a,b > 1 — колокольчик.
    """

    def __init__(self, concentration1: Any, concentration0: Any):
        self.concentration1 = jnp.asarray(concentration1, dtype=jnp.float32)
        self.concentration0 = jnp.asarray(concentration0, dtype=jnp.float32)
        _check_positive(self.concentration1, "Beta.concentration1")
        _check_positive(self.concentration0, "Beta.concentration0")
        batch_shape = jnp.broadcast_shapes(self.concentration1.shape, self.concentration0.shape)
        super().__init__(batch_shape=batch_shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return unit_interval

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        # Клипаем чтобы log(0) не взорвал всё к чертям
        safe_v = jnp.clip(value, _EPS, 1.0 - _EPS)
        result = ((self.concentration1 - 1.0) * jnp.log(safe_v)
                + (self.concentration0 - 1.0) * jnp.log1p(-safe_v)
                - betaln(self.concentration1, self.concentration0))
        return jnp.where((value > 0) & (value < 1), result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return jrandom.beta(key, self.concentration1, self.concentration0, shape=shape)

    @property
    def mean(self) -> jnp.ndarray:
        total = self.concentration1 + self.concentration0
        return self.concentration1 / total

    @property
    def entropy(self) -> jnp.ndarray:
        a, b = self.concentration1, self.concentration0
        return (betaln(a, b) - (a - 1) * jax.scipy.special.digamma(a)
                - (b - 1) * jax.scipy.special.digamma(b)
                + (a + b - 2) * jax.scipy.special.digamma(a + b))


class Uniform(Distribution):
    """Равномерное распределение на [low, high].

    Звучит просто, но как prior — обычно плохая идея
    (hard boundaries портят геометрию для HMC).
    """

    def __init__(self, low: Any = 0.0, high: Any = 1.0):
        self.low = jnp.asarray(low, dtype=jnp.float32)
        self.high = jnp.asarray(high, dtype=jnp.float32)
        try:
            if jnp.any(self.low >= self.high):
                raise ValueError(
                    f"Uniform requires low < high, got low={self.low}, high={self.high}"
                )
        except jax.errors.TracerBoolConversionError:
            pass
        batch_shape = jnp.broadcast_shapes(self.low.shape, self.high.shape)
        super().__init__(batch_shape=batch_shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return Bounded(self.low, self.high)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        in_support = (value >= self.low) & (value <= self.high)
        return jnp.where(in_support, -jnp.log(self.high - self.low + _EPS), -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return jrandom.uniform(key, shape=shape, minval=self.low, maxval=self.high)


class Exponential(Distribution):
    """Экспоненциальное распределение Exp(rate).

    Время до следующего события в пуассоновском процессе.
    rate=0 — бесконечное ожидание. rate=inf — мгновение.
    """

    def __init__(self, rate: Any = 1.0):
        self.rate = jnp.asarray(rate, dtype=jnp.float32)
        _check_positive(self.rate, "Exponential.rate")
        super().__init__(batch_shape=self.rate.shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return positive

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        result = jnp.log(jnp.maximum(self.rate, _EPS)) - self.rate * value
        return jnp.where(value >= 0, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        safe_rate = jnp.maximum(self.rate, _EPS)
        return jrandom.exponential(key, shape=shape) / safe_rate

    @property
    def mean(self) -> jnp.ndarray:
        return 1.0 / self.rate


class Cauchy(Distribution):
    """Распределение Коши — нормалка на стероидах (бесконечная дисперсия).

    Тяжёлые хвосты такие тяжёлые, что среднее не определено.
    Хорошо для робастных моделей и когда данные с выбросами.
    """

    def __init__(self, loc: Any = 0.0, scale: Any = 1.0):
        self.loc = jnp.asarray(loc, dtype=jnp.float32)
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        _check_positive(self.scale, "Cauchy.scale")
        batch_shape = jnp.broadcast_shapes(self.loc.shape, self.scale.shape)
        super().__init__(batch_shape=batch_shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return real

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        safe_scale = jnp.maximum(self.scale, _EPS)
        # log(1/(pi*scale * (1 + ((x-loc)/scale)^2)))
        return (-jnp.log(jnp.pi) - jnp.log(safe_scale)
                - jnp.log1p(((value - self.loc) / safe_scale) ** 2))

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return self.loc + self.scale * jrandom.cauchy(key, shape=shape)


class StudentT(Distribution):
    """Распределение Стьюдента — нормалка с тяжёлыми хвостами.

    df (degrees of freedom) контролирует тяжесть хвостов:
    df=1 — это Cauchy, df->inf — это Normal. df=4-5 — золотая середина.
    """

    def __init__(self, df: Any, loc: Any = 0.0, scale: Any = 1.0):
        self.df = jnp.asarray(df, dtype=jnp.float32)
        self.loc = jnp.asarray(loc, dtype=jnp.float32)
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        _check_positive(self.df, "StudentT.df")
        _check_positive(self.scale, "StudentT.scale")
        batch_shape = jnp.broadcast_shapes(self.df.shape, self.loc.shape, self.scale.shape)
        super().__init__(batch_shape=batch_shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return real

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        safe_scale = jnp.maximum(self.scale, _EPS)
        y = (value - self.loc) / safe_scale
        z = jnp.maximum(1.0 + y ** 2 / self.df, _EPS)
        return (gammaln((self.df + 1.0) / 2.0) - gammaln(self.df / 2.0)
                - 0.5 * jnp.log(self.df * jnp.pi) - jnp.log(safe_scale)
                - (self.df + 1.0) / 2.0 * jnp.log(z))

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return self.loc + self.scale * jrandom.t(key, self.df, shape=shape)


class Dirichlet(Distribution):
    """Дирихле — обобщение Beta на многомерный случай.

    Семплы живут на симплексе (сумма = 1, все >= 0).
    concentration = [1,1,...,1] — это равномерное на симплексе.
    """

    def __init__(self, concentration: Any):
        self.concentration = jnp.asarray(concentration, dtype=jnp.float32)
        if self.concentration.ndim < 1:
            raise ValueError("concentration для Dirichlet должен быть минимум 1D, что логично")
        _check_positive(self.concentration, "Dirichlet.concentration")
        event_shape = self.concentration.shape[-1:]
        batch_shape = self.concentration.shape[:-1]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def support(self) -> Constraint:
        return Simplex()

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        safe_v = jnp.clip(value, _EPS, 1.0)
        # B(alpha) = prod(Gamma(a_i)) / Gamma(sum(a_i))
        # log_prob = sum((a_i - 1)*log(x_i)) - log(B(alpha))
        log_normalizer = (gammaln(jnp.sum(self.concentration, axis=-1))
                         - jnp.sum(gammaln(self.concentration), axis=-1))
        result = (jnp.sum((self.concentration - 1.0) * jnp.log(safe_v), axis=-1)
                + log_normalizer)
        on_simplex = (jnp.all(value > 0, axis=-1)) & (jnp.abs(jnp.sum(value, axis=-1) - 1.0) < 1e-5)
        return jnp.where(on_simplex, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return jrandom.dirichlet(key, self.concentration, shape=shape)


class MultivariateNormal(Distribution):
    """Многомерное нормальное распределение MVN(loc, Sigma).

    Можно задать ковариацию тремя способами:
    1) covariance_matrix — сама ковариационная матрица (мы сделаем Cholesky)
    2) precision_matrix — обратная к ковариации (инвертируем, потом Cholesky)
    3) scale_tril — нижнетреугольный Cholesky-фактор (самый эффективный способ)

    Внутри всегда работаем с scale_tril — потому что solve_triangular
    быстрее и стабильнее чем inv().
    """

    def __init__(
        self,
        loc: Any,
        covariance_matrix: Optional[Any] = None,
        precision_matrix: Optional[Any] = None,
        scale_tril: Optional[Any] = None,
    ):
        self.loc = jnp.atleast_1d(jnp.asarray(loc, dtype=jnp.float32))
        if scale_tril is not None:
            self.scale_tril = jnp.asarray(scale_tril, dtype=jnp.float32)
        elif covariance_matrix is not None:
            self.scale_tril = jnp.linalg.cholesky(
                jnp.asarray(covariance_matrix, dtype=jnp.float32)
            )
        elif precision_matrix is not None:
            cov = jnp.linalg.inv(jnp.asarray(precision_matrix, dtype=jnp.float32))
            self.scale_tril = jnp.linalg.cholesky(cov)
        else:
            # Если ничего не передали — единичная ковариация
            n = self.loc.shape[-1]
            self.scale_tril = jnp.eye(n, dtype=jnp.float32)

        event_shape = self.loc.shape[-1:]
        batch_shape = jnp.broadcast_shapes(
            self.loc.shape[:-1], self.scale_tril.shape[:-2]
        )
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def support(self) -> Constraint:
        return real

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        diff = value - self.loc
        k = self.loc.shape[-1]
        # Решаем L*M = diff вместо inv(L) @ diff — это O(n^2) vs O(n^3)
        M = jax.scipy.linalg.solve_triangular(
            self.scale_tril, diff[..., None], lower=True
        )
        log_det = jnp.sum(
            jnp.log(jnp.abs(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)) + _EPS),
            axis=-1
        )
        return -0.5 * (k * _LOG_2PI + 2.0 * log_det + jnp.sum(M[..., 0] ** 2, axis=-1))

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        k = self.event_shape[0]
        eps = jrandom.normal(key, shape=shape + (k,))
        # z = mu + L @ eps (через einsum для batch-совместимости)
        return self.loc + jnp.einsum("...ij,...j->...i", self.scale_tril, eps)


class LKJCholesky(Distribution):
    """LKJ распределение над Cholesky-факторами корреляционных матриц.

    Используем C-vine метод: семплим частичные корреляции из Beta,
    потом stick-breaking в Cholesky-фактор.

    concentration=1 — равномерное на корреляционных матрицах.
    concentration>1 — стягивает к единичной (диагональной).
    concentration<1 — предпочитает сильные корреляции (экзотика).
    """

    def __init__(self, dimension: int, concentration: Any = 1.0):
        if dimension < 2:
            raise ValueError("Размерность должна быть >= 2, корреляция одной переменной — это 1.0")
        self.dimension = dimension
        self.concentration = jnp.asarray(concentration, dtype=jnp.float32)
        batch_shape = self.concentration.shape
        event_shape = (dimension, dimension)
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

        # C-vine: partial correlation at row i, column j uses
        # Beta(eta + (d-1-j)/2, eta + (d-1-j)/2).
        # Build per-element offset vector in packed lower-triangular order.
        offsets = []
        for i in range(1, dimension):
            for j in range(i):
                offsets.append(0.5 * (dimension - 1 - j))
        offsets = jnp.array(offsets, dtype=jnp.float32)
        self._beta_concentration = self.concentration[..., None] + offsets

    @property
    def support(self) -> Constraint:
        return CorrCholesky(dim=self.dimension)

    def _signed_stick_breaking_tril(self, partial_corr: jnp.ndarray, n: int) -> jnp.ndarray:
        """Превращаем частичные корреляции в Cholesky-фактор.

        L_ij = p_ij * prod_{k<j} sqrt(1 - p_ik^2)
        L_ii = prod_{k<i} sqrt(1 - p_ik^2)

        Если это выглядит сложно — так оно и есть. Но оно работает.
        """
        batch_shape = partial_corr.shape[:-1]
        cholesky = jnp.zeros(batch_shape + (n, n), dtype=jnp.float32)
        cholesky = cholesky.at[..., jnp.arange(n), jnp.arange(n)].set(1.0)

        idx = 0
        for i in range(1, n):
            row_prod = jnp.ones(batch_shape, dtype=jnp.float32)
            for j in range(i):
                p = partial_corr[..., idx]
                cholesky = cholesky.at[..., i, j].set(p * row_prod)
                row_prod = row_prod * jnp.sqrt(jnp.clip(1.0 - p * p, _EPS, 1.0))
                idx += 1
            cholesky = cholesky.at[..., i, i].set(row_prod)
        return cholesky

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        n = self.dimension
        c = self.concentration
        diag = jnp.diagonal(value, axis1=-2, axis2=-1)
        exponents = n - jnp.arange(n, dtype=jnp.float32) + 2.0 * c - 3.0
        return jnp.sum(exponents * jnp.log(jnp.clip(diag, _EPS, None)), axis=-1)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        n = self.dimension
        num_tril = n * (n - 1) // 2
        shape = sample_shape + self.batch_shape + (num_tril,)

        conc = jnp.broadcast_to(self._beta_concentration, shape)
        beta_sample = jrandom.beta(key, conc, conc, shape=shape)
        partial_corr = 2.0 * beta_sample - 1.0
        return self._signed_stick_breaking_tril(partial_corr, n)


class Bernoulli(Distribution):
    """Бернулли — орёл/решка, 0/1, да/нет.

    Можно задать через probs (вероятность) или logits (лог-отношение шансов).
    Внутри всегда работаем с logits — так числено стабильнее.
    """

    def __init__(self, probs: Optional[Any] = None, logits: Optional[Any] = None):
        if probs is not None and logits is not None:
            raise ValueError("Укажи либо probs, либо logits — оба сразу крашнут логику")
        if probs is not None:
            self.probs = jnp.clip(jnp.asarray(probs, dtype=jnp.float32), _EPS, 1.0 - _EPS)
            self.logits = jnp.log(self.probs) - jnp.log1p(-self.probs)
        elif logits is not None:
            self.logits = jnp.asarray(logits, dtype=jnp.float32)
            self.probs = jax.nn.sigmoid(self.logits)
        else:
            raise ValueError("Нужен или probs, или logits. Телепатия не поддерживается")
        super().__init__(batch_shape=self.probs.shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return Bounded(0.0, 1.0)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        # BCE через logits — числено стабильная формула
        result = value * self.logits - jax.nn.softplus(self.logits)
        is_binary = (value == 0.0) | (value == 1.0)
        return jnp.where(is_binary, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return jrandom.bernoulli(key, self.probs, shape=shape).astype(jnp.float32)


class Categorical(Distribution):
    """Категориальное — бросок кости с K гранями.

    Задаётся через probs (вектор вероятностей) или logits.
    Семплы — целые числа от 0 до K-1.
    """

    def __init__(self, probs: Optional[Any] = None, logits: Optional[Any] = None):
        if probs is not None and logits is not None:
            raise ValueError("probs или logits — выбери что-нибудь одно")
        if probs is not None:
            self.probs = jnp.asarray(probs, dtype=jnp.float32)
            self.probs = self.probs / jnp.sum(self.probs, axis=-1, keepdims=True)
            self.logits = jnp.log(jnp.maximum(self.probs, _EPS))
        elif logits is not None:
            self.logits = jnp.asarray(logits, dtype=jnp.float32)
            self.probs = jax.nn.softmax(self.logits, axis=-1)
        else:
            raise ValueError("probs или logits — выбери что-нибудь одно")
        super().__init__(batch_shape=self.probs.shape[:-1], event_shape=())

    @property
    def support(self) -> Constraint:
        return Bounded(0, int(self.probs.shape[-1]) - 1)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value_f = jnp.asarray(value, dtype=jnp.float32)
        is_integer = value_f == jnp.floor(value_f)
        K = self.logits.shape[-1]
        in_range = (value_f >= 0) & (value_f < K)
        value_int = jnp.asarray(value_f, dtype=jnp.int32)
        log_normalizer = jax.scipy.special.logsumexp(self.logits, axis=-1)
        normalized_logits = self.logits - log_normalizer[..., None]
        result = jnp.take_along_axis(normalized_logits, value_int[..., None], axis=-1)[..., 0]
        return jnp.where(is_integer & in_range, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return jrandom.categorical(key, self.logits, shape=shape)


class Poisson(Distribution):
    """Пуассон — считает количество событий за интервал.

    rate — среднее число событий. Семплы — неотрицательные целые.
    Для больших rate лучше аппроксимировать нормалкой, но мы честные.
    """

    def __init__(self, rate: Any):
        self.rate = jnp.asarray(rate, dtype=jnp.float32)
        try:
            if jnp.any(self.rate < 0):
                raise ValueError(f"Poisson.rate must be non-negative, got {self.rate}")
        except jax.errors.TracerBoolConversionError:
            pass
        super().__init__(batch_shape=self.rate.shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return Bounded(0.0, float("inf"))

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        safe_rate = jnp.maximum(self.rate, _EPS)
        result = value * jnp.log(safe_rate) - self.rate - gammaln(value + 1.0)
        is_valid = (value >= 0) & (value == jnp.floor(value))
        is_zero_rate = self.rate <= _EPS
        zero_rate_lp = jnp.where(value == 0.0, jnp.zeros_like(result), -jnp.inf)
        return jnp.where(is_zero_rate, zero_rate_lp, jnp.where(is_valid, result, -jnp.inf))

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return jrandom.poisson(key, self.rate, shape=shape).astype(jnp.float32)


class InverseGamma(Distribution):
    """Обратное Гамма-распределение.

    Если X ~ Gamma(a, b), то 1/X ~ InverseGamma(a, b).
    Классический conjugate prior для дисперсии нормалки,
    но HalfNormal обычно лучше (менее информативный).
    """

    def __init__(self, concentration: Any, scale: Any):
        self.concentration = jnp.asarray(concentration, dtype=jnp.float32)
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        _check_positive(self.concentration, "InverseGamma.concentration")
        _check_positive(self._scale, "InverseGamma.scale")
        batch_shape = jnp.broadcast_shapes(self.concentration.shape, self._scale.shape)
        super().__init__(batch_shape=batch_shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return positive

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        safe_value = jnp.maximum(value, _EPS)
        result = (self.concentration * jnp.log(jnp.maximum(self._scale, _EPS))
                - gammaln(self.concentration)
                - (self.concentration + 1.0) * jnp.log(safe_value)
                - self._scale / safe_value)
        return jnp.where(value > 0, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        return self._scale / jrandom.gamma(key, self.concentration, shape=shape)


class Horseshoe(Distribution):
    """Horseshoe prior — подковообразный prior для разреженной регрессии.

    Иерархическая конструкция:
      lambda ~ HalfCauchy(1)
      tau ~ HalfCauchy(scale)
      beta ~ Normal(0, lambda * tau)

    Маргинальный log_prob в замкнутом виде не существует (поэтому NotImplementedError).
    Для инференса используй иерархическую параметризацию (раскрой через sample).
    """

    def __init__(self, scale: Any = 1.0):
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        super().__init__(batch_shape=self.scale.shape, event_shape=())

    @property
    def support(self) -> Constraint:
        return real

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(
            "Маргинальный log_prob подковы не имеет замкнутой формы. "
            "Используй иерархическую параметризацию — это не лень, это математика."
        )

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        key_lam, key_tau, key_beta = jrandom.split(key, 3)
        shape = sample_shape + self.batch_shape
        lam = jnp.abs(jrandom.cauchy(key_lam, shape=shape))
        tau = self.scale * jnp.abs(jrandom.cauchy(key_tau, shape=shape))
        scale_beta = jnp.maximum(lam * tau, _EPS)
        return scale_beta * jrandom.normal(key_beta, shape=shape)


class GaussianProcess(Distribution):
    """Гауссовский процесс (конечномерное приближение).

    Это по сути MultivariateNormal, где ковариационная матрица
    задаётся ядром (RBF, Matern и т.д.). Пока — тонкая обёртка над MVN.

    TODO: добавить полноценные ядра, mean functions, sparse GP.
    """

    def __init__(
        self,
        loc: Any = 0.0,
        covariance_matrix: Optional[Any] = None,
        scale_tril: Optional[Any] = None,
    ):
        self._mvnormal = MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix, scale_tril=scale_tril,
        )
        super().__init__(
            batch_shape=self._mvnormal.batch_shape,
            event_shape=self._mvnormal.event_shape,
        )

    @property
    def support(self) -> Constraint:
        return real

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return self._mvnormal.log_prob(value)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        return self._mvnormal.sample(key, sample_shape)


class Binomial(Distribution):
    """Биномиальное — число успехов в n независимых испытаниях Бернулли.

    total_count — число испытаний, probs или logits — вероятность успеха.
    """

    def __init__(
        self,
        total_count: Any,
        probs: Optional[Any] = None,
        logits: Optional[Any] = None,
    ):
        self.total_count = jnp.asarray(total_count, dtype=jnp.float32)
        if probs is not None and logits is not None:
            raise ValueError("probs или logits — выбери одно")
        if probs is not None:
            self.probs = jnp.clip(jnp.asarray(probs, dtype=jnp.float32), _EPS, 1.0 - _EPS)
            self.logits = jnp.log(self.probs) - jnp.log1p(-self.probs)
        elif logits is not None:
            self.logits = jnp.asarray(logits, dtype=jnp.float32)
            self.probs = jax.nn.sigmoid(self.logits)
        else:
            raise ValueError("Нужен probs или logits")
        batch = jnp.broadcast_shapes(self.total_count.shape, self.probs.shape)
        super().__init__(batch_shape=batch, event_shape=())

    @property
    def support(self) -> Constraint:
        return Bounded(0.0, float("inf"))

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        k = jnp.asarray(value, dtype=jnp.float32)
        n = self.total_count
        log_comb = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        log_p = k * jnp.log(jnp.maximum(self.probs, _EPS))
        log_1mp = (n - k) * jnp.log(jnp.maximum(1.0 - self.probs, _EPS))
        result = log_comb + log_p + log_1mp
        valid = (k >= 0) & (k <= n) & (k == jnp.floor(k))
        return jnp.where(valid, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        n = jnp.broadcast_to(self.total_count, shape)
        p = jnp.broadcast_to(self.probs, shape)
        max_n = int(np.max(np.asarray(self.total_count)))
        trials = jrandom.bernoulli(key, p[..., None], shape=shape + (max_n,))
        mask = jnp.arange(max_n) < n[..., None]
        return jnp.sum(trials * mask, axis=-1).astype(jnp.float32)


class Multinomial(Distribution):
    """Мультиномиальное — обобщение биномиального на K категорий.

    total_count — общее число испытаний, probs/logits — вероятности по категориям.
    """

    def __init__(
        self,
        total_count: Any,
        probs: Optional[Any] = None,
        logits: Optional[Any] = None,
    ):
        self.total_count = jnp.asarray(total_count, dtype=jnp.float32)
        if probs is not None and logits is not None:
            raise ValueError("probs или logits — выбери одно")
        if probs is not None:
            self.probs = jnp.asarray(probs, dtype=jnp.float32)
            self.probs = self.probs / jnp.sum(self.probs, axis=-1, keepdims=True)
            self.logits = jnp.log(jnp.maximum(self.probs, _EPS))
        elif logits is not None:
            self.logits = jnp.asarray(logits, dtype=jnp.float32)
            self.probs = jax.nn.softmax(self.logits, axis=-1)
        else:
            raise ValueError("Нужен probs или logits")
        batch = jnp.broadcast_shapes(
            self.total_count.shape, self.probs.shape[:-1]
        )
        super().__init__(
            batch_shape=batch,
            event_shape=(self.probs.shape[-1],),
        )

    @property
    def support(self) -> Constraint:
        return Bounded(0.0, float("inf"))

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        k = jnp.asarray(value, dtype=jnp.float32)
        n = self.total_count
        log_factorial_n = gammaln(n + 1)
        log_factorial_k = jnp.sum(gammaln(k + 1), axis=-1)
        log_probs = jnp.sum(k * jnp.log(jnp.maximum(self.probs, _EPS)), axis=-1)
        result = log_factorial_n - log_factorial_k + log_probs
        valid = (
            jnp.isclose(jnp.sum(k, axis=-1), n, atol=1e-4)
            & jnp.all(k >= 0, axis=-1)
            & jnp.all(k == jnp.floor(k), axis=-1)
        )
        return jnp.where(valid, result, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        shape = sample_shape + self.batch_shape
        n = jnp.broadcast_to(self.total_count, shape)
        max_n = int(np.max(np.asarray(self.total_count)))
        K = self.probs.shape[-1]
        keys = jrandom.split(key, max_n)

        def _draw_one(k):
            return jrandom.categorical(k, self.logits, shape=shape)

        cats = jax.vmap(_draw_one)(keys)
        cats = jnp.moveaxis(cats, 0, -1)
        one_hot = jax.nn.one_hot(cats, K)
        mask = (jnp.arange(max_n) < n[..., None])[..., None]
        return jnp.sum(one_hot * mask, axis=-2).astype(jnp.float32)


class Delta(Distribution):
    """Delta (точечная масса) — всегда возвращает фиксированное значение.

    Полезно для conditioning, deterministic nodes, и как компонент mixture.
    log_prob(v) = log_density если v == value, иначе -inf.
    """

    def __init__(self, value: Any, log_density: float = 0.0, event_dim: int = 0):
        self.v = jnp.asarray(value, dtype=jnp.float32)
        self.log_density = jnp.asarray(log_density, dtype=jnp.float32)
        if event_dim > self.v.ndim:
            event_dim = self.v.ndim
        event_shape = self.v.shape[self.v.ndim - event_dim:] if event_dim > 0 else ()
        batch_shape = self.v.shape[:self.v.ndim - event_dim]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def support(self) -> Constraint:
        return real

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float32)
        event_dims = tuple(range(self.v.ndim - len(self.event_shape), self.v.ndim))
        if event_dims:
            close = jnp.all(jnp.isclose(value, self.v, atol=1e-6), axis=event_dims)
        else:
            close = jnp.isclose(value, self.v, atol=1e-6)
        return jnp.where(close, self.log_density, -jnp.inf)

    def sample(self, key: jax.random.PRNGKey, sample_shape: Shape = ()) -> jnp.ndarray:
        return jnp.broadcast_to(self.v, sample_shape + self.v.shape)
