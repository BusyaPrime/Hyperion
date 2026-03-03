"""Тесты для базовых вероятностных распределений.

Проверяем что Normal семплит без крашей — звучит тривиально, но мы видели вещи.
Если log_prob вернул NaN — распределение сломано, чини.
KS-тест: проверяем что семплы реально из правильного распределения, а не чушь собачья.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from hyperion_dsl.distributions import (
    Normal,
    HalfNormal,
    LogNormal,
    Gamma,
    Beta,
    Uniform,
    Exponential,
    Cauchy,
    StudentT,
    Bernoulli,
    Poisson,
    InverseGamma,
    Dirichlet,
    MultivariateNormal,
)


class TestNormal:
    """Нормалка — королева распределений. Тестируем её как следует."""

    def test_log_prob(self):
        # Классика: log_prob в центре = -0.5*log(2*pi)
        d = Normal(0.0, 1.0)
        lp = d.log_prob(jnp.array(0.0))
        expected = -0.5 * jnp.log(2.0 * jnp.pi)
        assert jnp.allclose(lp, expected, atol=1e-5)

    def test_sample_shape(self):
        # Проверяем что Normal семплит без крашей — звучит тривиально, но мы видели вещи
        d = Normal(0.0, 1.0)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert samples.shape == (100,)

    def test_sample_statistics(self):
        # Среднее и std должны совпадать с параметрами — иначе семплер врёт
        d = Normal(3.0, 2.0)
        key = jax.random.PRNGKey(42)
        samples = d.sample(key, sample_shape=(10000,))
        assert abs(float(jnp.mean(samples)) - 3.0) < 0.1
        assert abs(float(jnp.std(samples)) - 2.0) < 0.1

    def test_batch_shape(self):
        d = Normal(jnp.zeros(3), jnp.ones(3))
        assert d.batch_shape == (3,)

    def test_support(self):
        d = Normal(0.0, 1.0)
        assert d.support.check(jnp.array(0.0))

    def test_ks_normal(self):
        # KS-тест: проверяем что семплы реально из правильного распределения, а не чушь собачья
        d = Normal(2.0, 1.5)
        key = jax.random.PRNGKey(123)
        samples = np.array(d.sample(key, sample_shape=(5000,)))
        # Сравниваем эмпирическую CDF с теоретической
        stat, pval = stats.kstest(samples, "norm", args=(2.0, 1.5))
        assert pval > 0.01, f"KS отверг: stat={stat}, pval={pval} — семплы не из N(2,1.5)"

    def test_edge_case_tiny_scale(self):
        # Edge case: почти нулевая дисперсия — не должно крашнуться
        d = Normal(5.0, 1e-6)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(10,))
        lp = d.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))


class TestGamma:
    """Гамма — для положительных величин. Rate vs scale — классический источник багов."""

    def test_log_prob_positive(self):
        d = Gamma(2.0, 1.0)
        lp = d.log_prob(jnp.array(1.0))
        assert jnp.isfinite(lp)

    def test_sample_positive(self):
        d = Gamma(2.0, 1.0)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert jnp.all(samples > 0)

    def test_mean(self):
        d = Gamma(5.0, 2.0)
        key = jax.random.PRNGKey(42)
        samples = d.sample(key, sample_shape=(10000,))
        assert abs(float(jnp.mean(samples)) - 2.5) < 0.1

    def test_ks_gamma(self):
        # KS-тест: проверяем что семплы из Gamma(shape, rate)
        shape, rate = 3.0, 2.0
        d = Gamma(shape, rate)
        key = jax.random.PRNGKey(456)
        samples = np.array(d.sample(key, sample_shape=(5000,)))
        stat, pval = stats.kstest(samples, "gamma", args=(shape, 0, 1 / rate))
        assert pval > 0.01, f"KS отверг Gamma: stat={stat}, pval={pval}"

    def test_edge_case_large_concentration(self):
        # Edge case: большая concentration — не должно взорваться
        d = Gamma(100.0, 10.0)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert jnp.all(samples > 0)
        assert jnp.all(jnp.isfinite(d.log_prob(samples)))


class TestBeta:
    """Бета — классика для вероятностей. a=b=1 — Uniform(0,1)."""

    def test_log_prob_in_range(self):
        d = Beta(2.0, 5.0)
        lp = d.log_prob(jnp.array(0.3))
        assert jnp.isfinite(lp)

    def test_sample_in_unit_interval(self):
        d = Beta(2.0, 5.0)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert jnp.all((samples >= 0) & (samples <= 1))

    def test_ks_beta(self):
        # KS-тест: семплы должны быть из Beta(a, b)
        a, b = 2.0, 5.0
        d = Beta(a, b)
        key = jax.random.PRNGKey(789)
        samples = np.array(d.sample(key, sample_shape=(5000,)))
        stat, pval = stats.kstest(samples, "beta", args=(a, b))
        assert pval > 0.01, f"KS отверг Beta: stat={stat}, pval={pval}"

    def test_edge_case_small_concentrations(self):
        # Edge case: a,b < 1 — U-образная бета, границы 0 и 1
        d = Beta(0.5, 0.5)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert jnp.all((samples > 0) & (samples < 1))


class TestDirichlet:
    """Дирихле — Beta на симплексе. Сумма = 1, иначе не жизнь."""

    def test_sample_on_simplex(self):
        d = Dirichlet(jnp.array([1.0, 2.0, 3.0]))
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert samples.shape == (100, 3)
        sums = jnp.sum(samples, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_log_prob(self):
        d = Dirichlet(jnp.array([2.0, 3.0, 4.0]))
        x = jnp.array([0.2, 0.3, 0.5])
        lp = d.log_prob(x)
        assert jnp.isfinite(lp)


class TestMultivariateNormal:
    """MVN — нормалка в многомерном мире. Cholesky внутри — это правильно."""

    def test_sample_shape(self):
        d = MultivariateNormal(
            loc=jnp.zeros(3),
            covariance_matrix=jnp.eye(3),
        )
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(50,))
        assert samples.shape == (50, 3)

    def test_log_prob(self):
        d = MultivariateNormal(
            loc=jnp.zeros(2),
            covariance_matrix=jnp.eye(2),
        )
        lp = d.log_prob(jnp.zeros(2))
        expected = -jnp.log(2 * jnp.pi)
        assert jnp.allclose(lp, expected, atol=1e-4)


class TestUniform:
    """Равномерное — звучит просто, но как prior обычно плохая идея."""

    def test_sample_in_range(self):
        d = Uniform(-2.0, 5.0)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert jnp.all((samples >= -2.0) & (samples <= 5.0))


class TestExponential:
    """Экспоненциальное — время до следующего события."""

    def test_sample_positive(self):
        d = Exponential(2.0)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert jnp.all(samples > 0)

    def test_mean(self):
        d = Exponential(0.5)
        key = jax.random.PRNGKey(42)
        samples = d.sample(key, sample_shape=(10000,))
        assert abs(float(jnp.mean(samples)) - 2.0) < 0.15

    def test_edge_case_high_rate(self):
        d = Exponential(1000.0)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert jnp.all(samples > 0)
        assert float(jnp.mean(samples)) < 0.01


class TestBinomial:
    """Биномиальное — число успехов в n испытаниях."""

    def test_log_prob_valid(self):
        from hyperion_dsl.distributions import Binomial
        d = Binomial(total_count=10, probs=0.5)
        lp = d.log_prob(jnp.array(5.0))
        assert jnp.isfinite(lp)
        assert float(lp) > -10

    def test_log_prob_out_of_range(self):
        from hyperion_dsl.distributions import Binomial
        d = Binomial(total_count=10, probs=0.5)
        lp = d.log_prob(jnp.array(11.0))
        assert float(lp) == float("-inf")

    def test_sample_in_range(self):
        from hyperion_dsl.distributions import Binomial
        d = Binomial(total_count=10, probs=0.3)
        key = jax.random.PRNGKey(42)
        samples = d.sample(key, sample_shape=(50,))
        assert jnp.all(samples >= 0)
        assert jnp.all(samples <= 10)

    def test_mean(self):
        from hyperion_dsl.distributions import Binomial
        d = Binomial(total_count=100, probs=0.4)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(500,))
        assert abs(float(jnp.mean(samples)) - 40.0) < 5.0

    def test_batched_probs(self):
        from hyperion_dsl.distributions import Binomial
        d = Binomial(total_count=20, probs=jnp.array([0.2, 0.8]))
        key = jax.random.PRNGKey(42)
        samples = d.sample(key, sample_shape=(200,))
        assert samples.shape == (200, 2)
        mean_low = float(jnp.mean(samples[:, 0]))
        mean_high = float(jnp.mean(samples[:, 1]))
        assert abs(mean_low - 4.0) < 2.0
        assert abs(mean_high - 16.0) < 2.0

    def test_batched_total_count(self):
        from hyperion_dsl.distributions import Binomial
        d = Binomial(total_count=jnp.array([5, 50]), probs=0.5)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(100,))
        assert samples.shape == (100, 2)
        assert jnp.all(samples[:, 0] <= 5)
        assert jnp.all(samples[:, 1] <= 50)


class TestMultinomial:
    """Мультиномиальное — обобщение биномиального на K категорий."""

    def test_log_prob(self):
        from hyperion_dsl.distributions import Multinomial
        d = Multinomial(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]))
        lp = d.log_prob(jnp.array([2.0, 3.0, 5.0]))
        assert jnp.isfinite(lp)

    def test_sample_shape(self):
        from hyperion_dsl.distributions import Multinomial
        d = Multinomial(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]))
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(50,))
        assert samples.shape == (50, 3)

    def test_sample_sums_to_n(self):
        from hyperion_dsl.distributions import Multinomial
        d = Multinomial(total_count=20, probs=jnp.array([0.3, 0.7]))
        key = jax.random.PRNGKey(42)
        samples = d.sample(key, sample_shape=(30,))
        sums = jnp.sum(samples, axis=-1)
        assert jnp.allclose(sums, 20.0)

    def test_log_prob_valid_input(self):
        from hyperion_dsl.distributions import Multinomial
        d = Multinomial(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]))
        lp = d.log_prob(jnp.array([2.0, 3.0, 5.0]))
        assert jnp.isfinite(lp), "Valid input must produce finite log_prob"

    def test_log_prob_wrong_sum_returns_neginf(self):
        from hyperion_dsl.distributions import Multinomial
        d = Multinomial(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]))
        lp = d.log_prob(jnp.array([3.0, 3.0, 5.0]))  # sum=11 != 10
        assert lp == float("-inf"), f"sum(k)!=n must give -inf, got {lp}"

    def test_log_prob_negative_counts_returns_neginf(self):
        from hyperion_dsl.distributions import Multinomial
        d = Multinomial(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]))
        lp = d.log_prob(jnp.array([-1.0, 6.0, 5.0]))  # sum=10 but negative
        assert lp == float("-inf"), f"Negative counts must give -inf, got {lp}"

    def test_log_prob_non_integer_returns_neginf(self):
        from hyperion_dsl.distributions import Multinomial
        d = Multinomial(total_count=10, probs=jnp.array([0.2, 0.3, 0.5]))
        lp = d.log_prob(jnp.array([2.5, 3.0, 4.5]))
        assert lp == float("-inf"), f"Non-integer counts must give -inf, got {lp}"


class TestDelta:
    """Delta — точечная масса."""

    def test_log_prob_at_value(self):
        from hyperion_dsl.distributions import Delta
        d = Delta(3.0)
        assert float(d.log_prob(jnp.array(3.0))) == 0.0

    def test_log_prob_away(self):
        from hyperion_dsl.distributions import Delta
        d = Delta(3.0)
        assert float(d.log_prob(jnp.array(5.0))) == float("-inf")

    def test_sample_always_value(self):
        from hyperion_dsl.distributions import Delta
        d = Delta(jnp.array([1.0, 2.0]))
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(10,))
        assert samples.shape == (10, 2)
        assert jnp.allclose(samples, jnp.array([1.0, 2.0]))

    def test_log_density(self):
        from hyperion_dsl.distributions import Delta
        d = Delta(0.0, log_density=-2.0)
        assert float(d.log_prob(jnp.array(0.0))) == -2.0
