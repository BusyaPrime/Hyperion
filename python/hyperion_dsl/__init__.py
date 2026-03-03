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

__all__ = [
    "sample",
    "plate",
    "param",
    "deterministic",
    "factor",
    "model",
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
]
