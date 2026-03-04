"""HYPERION DSL — язык описания вероятностных моделей.

Импортируй отсюда всё что нужно для описания модели:
примитивы (sample, plate, param), ограничения (positive, bounded),
распределения (Normal, Gamma, ...) и декоратор @model.

Пример минимальной модели:
    from hyperion_dsl import model, sample, Normal

    @model
    def coin_flip(data):
        p = sample("p", Beta(1, 1))
        sample("obs", Bernoulli(probs=p), obs=data)
"""

from hyperion_dsl.primitives import sample, plate, param, deterministic, factor
from hyperion_dsl.model import model
from hyperion_dsl.constraints import (
    Constraint,
    Positive,
    Bounded,
    Simplex,
    CorrCholesky,
    UnitInterval,
    Real,
    PositiveDefinite,
    LowerTriangular,
    positive,
    bounded,
    simplex,
    corr_cholesky,
    unit_interval,
    real,
    positive_definite,
    lower_triangular,
)
from hyperion_dsl.distributions import (
    Distribution,
    Normal,
    HalfNormal,
    HalfCauchy,
    LogNormal,
    Gamma,
    Beta,
    Uniform,
    Exponential,
    Cauchy,
    StudentT,
    Dirichlet,
    MultivariateNormal,
    LKJCholesky,
    Bernoulli,
    Categorical,
    Poisson,
    InverseGamma,
    Horseshoe,
    GaussianProcess,
    Binomial,
    Multinomial,
    Delta,
)

__all__ = [
    # primitives
    "sample",
    "plate",
    "param",
    "deterministic",
    "factor",
    "model",
    # constraints
    "Constraint",
    "Positive",
    "Bounded",
    "Simplex",
    "CorrCholesky",
    "UnitInterval",
    "Real",
    "PositiveDefinite",
    "LowerTriangular",
    "positive",
    "bounded",
    "simplex",
    "corr_cholesky",
    "unit_interval",
    "real",
    "positive_definite",
    "lower_triangular",
    # distributions
    "Distribution",
    "Normal",
    "HalfNormal",
    "HalfCauchy",
    "LogNormal",
    "Gamma",
    "Beta",
    "Uniform",
    "Exponential",
    "Cauchy",
    "StudentT",
    "Dirichlet",
    "MultivariateNormal",
    "LKJCholesky",
    "Bernoulli",
    "Categorical",
    "Poisson",
    "InverseGamma",
    "Horseshoe",
    "GaussianProcess",
    "Binomial",
    "Multinomial",
    "Delta",
]
