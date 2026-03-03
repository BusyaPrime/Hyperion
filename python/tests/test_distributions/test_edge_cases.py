"""Тесты краевых случаев для распределений.

Проверяем поведение на границах, NaN-устойчивость, support checks.
Если log_prob не возвращает -inf за пределами support — это баг.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from hyperion_dsl.distributions import (
    Normal,
    HalfNormal,
    HalfCauchy,
    LogNormal,
    Gamma,
    Beta,
    Uniform,
    Exponential,
    Bernoulli,
    Categorical,
    Poisson,
    Dirichlet,
    InverseGamma,
    StudentT,
    Cauchy,
)


class TestSupportBoundaries:
    """Все распределения должны возвращать -inf за пределами support."""

    def test_half_normal_negative(self):
        d = HalfNormal(1.0)
        assert d.log_prob(jnp.array(-1.0)) == -jnp.inf

    def test_half_cauchy_negative(self):
        d = HalfCauchy(1.0)
        assert d.log_prob(jnp.array(-1.0)) == -jnp.inf

    def test_lognormal_negative(self):
        d = LogNormal(0.0, 1.0)
        assert d.log_prob(jnp.array(-1.0)) == -jnp.inf

    def test_gamma_negative(self):
        d = Gamma(2.0, 1.0)
        assert d.log_prob(jnp.array(-1.0)) == -jnp.inf

    def test_beta_out_of_range(self):
        d = Beta(2.0, 2.0)
        assert d.log_prob(jnp.array(-0.1)) == -jnp.inf
        assert d.log_prob(jnp.array(1.1)) == -jnp.inf

    def test_exponential_negative(self):
        d = Exponential(1.0)
        assert d.log_prob(jnp.array(-1.0)) == -jnp.inf

    def test_inverse_gamma_negative(self):
        d = InverseGamma(2.0, 1.0)
        assert d.log_prob(jnp.array(-1.0)) == -jnp.inf

    def test_uniform_out_of_range(self):
        d = Uniform(0.0, 1.0)
        assert d.log_prob(jnp.array(-0.1)) == -jnp.inf
        assert d.log_prob(jnp.array(1.1)) == -jnp.inf

    def test_bernoulli_non_binary(self):
        d = Bernoulli(probs=0.5)
        assert d.log_prob(jnp.array(0.5)) == -jnp.inf
        assert d.log_prob(jnp.array(2.0)) == -jnp.inf

    def test_poisson_negative(self):
        d = Poisson(5.0)
        assert d.log_prob(jnp.array(-1.0)) == -jnp.inf

    def test_poisson_non_integer(self):
        d = Poisson(5.0)
        assert d.log_prob(jnp.array(2.5)) == -jnp.inf

    def test_dirichlet_off_simplex(self):
        d = Dirichlet(jnp.array([1.0, 1.0, 1.0]))
        off_simplex = jnp.array([0.5, 0.5, 0.5])
        assert d.log_prob(off_simplex) == -jnp.inf

    def test_dirichlet_negative_component(self):
        d = Dirichlet(jnp.array([1.0, 1.0, 1.0]))
        bad = jnp.array([-0.1, 0.6, 0.5])
        assert d.log_prob(bad) == -jnp.inf


class TestCategoricalNormalization:
    """Categorical log_prob должен быть нормализован (exp(log_prob).sum = 1)."""

    def test_normalized_from_logits(self):
        logits = jnp.array([1.0, 2.0, 3.0])
        d = Categorical(logits=logits)
        log_probs = jnp.array([d.log_prob(jnp.array(i)) for i in range(3)])
        total = jnp.sum(jnp.exp(log_probs))
        assert jnp.allclose(total, 1.0, atol=1e-5), (
            f"Сумма exp(log_probs) = {total}, ожидалось 1.0"
        )

    def test_normalized_from_probs(self):
        probs = jnp.array([0.2, 0.3, 0.5])
        d = Categorical(probs=probs)
        log_probs = jnp.array([d.log_prob(jnp.array(i)) for i in range(3)])
        total = jnp.sum(jnp.exp(log_probs))
        assert jnp.allclose(total, 1.0, atol=1e-5)

    def test_unnormalized_logits(self):
        logits = jnp.array([100.0, 200.0, 300.0])
        d = Categorical(logits=logits)
        log_probs = jnp.array([d.log_prob(jnp.array(i)) for i in range(3)])
        total = jnp.sum(jnp.exp(log_probs))
        assert jnp.allclose(total, 1.0, atol=1e-5)


class TestNumericalStability:
    """Проверяем, что нет NaN/Inf в нормальных условиях."""

    def test_normal_large_value(self):
        d = Normal(0.0, 1.0)
        lp = d.log_prob(jnp.array(100.0))
        assert jnp.isfinite(lp)

    def test_gamma_very_small_value(self):
        d = Gamma(0.5, 1.0)
        lp = d.log_prob(jnp.array(1e-30))
        assert jnp.isfinite(lp)

    def test_beta_near_boundary(self):
        d = Beta(2.0, 2.0)
        lp_low = d.log_prob(jnp.array(1e-6))
        lp_high = d.log_prob(jnp.array(1.0 - 1e-6))
        assert jnp.isfinite(lp_low)
        assert jnp.isfinite(lp_high)

    def test_student_t_extreme(self):
        d = StudentT(df=1.0, loc=0.0, scale=1.0)
        lp = d.log_prob(jnp.array(1e6))
        assert jnp.isfinite(lp)

    def test_cauchy_extreme(self):
        d = Cauchy(0.0, 1.0)
        lp = d.log_prob(jnp.array(1e8))
        assert jnp.isfinite(lp)

    def test_normal_tiny_scale(self):
        d = Normal(0.0, 1e-7)
        lp = d.log_prob(jnp.array(0.0))
        assert jnp.isfinite(lp)


class TestSamplingShapes:
    """Проверяем что sample возвращает правильные формы."""

    def test_normal_batch(self):
        d = Normal(jnp.zeros(3), jnp.ones(3))
        key = jrandom.PRNGKey(0)
        s = d.sample(key, (5,))
        assert s.shape == (5, 3)

    def test_categorical_shape(self):
        d = Categorical(probs=jnp.array([0.2, 0.3, 0.5]))
        key = jrandom.PRNGKey(0)
        s = d.sample(key, (10,))
        assert s.shape == (10,)
        assert jnp.all((s >= 0) & (s < 3))

    def test_dirichlet_shape(self):
        d = Dirichlet(jnp.array([1.0, 2.0, 3.0]))
        key = jrandom.PRNGKey(0)
        s = d.sample(key, (5,))
        assert s.shape == (5, 3)
        assert jnp.allclose(jnp.sum(s, axis=-1), 1.0, atol=1e-5)


class TestCategoricalEdgeCases:
    """Categorical должен возвращать -inf для нецелых и out-of-range значений."""

    def test_non_integer_returns_neg_inf(self):
        d = Categorical(probs=jnp.array([0.3, 0.3, 0.4]))
        assert d.log_prob(jnp.array(1.5)) == -jnp.inf

    def test_negative_index_returns_neg_inf(self):
        d = Categorical(probs=jnp.array([0.5, 0.5]))
        assert d.log_prob(jnp.array(-1)) == -jnp.inf

    def test_out_of_range_returns_neg_inf(self):
        d = Categorical(probs=jnp.array([0.5, 0.5]))
        assert d.log_prob(jnp.array(5)) == -jnp.inf
