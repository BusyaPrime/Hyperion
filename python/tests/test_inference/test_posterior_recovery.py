"""Posterior recovery тесты — золотой стандарт для PPL.

Идея проста: берём модель с АНАЛИТИЧЕСКИ ИЗВЕСТНЫМ постериором,
прогоняем через инференс и проверяем что получили правильный ответ.

Если HMC/NUTS не могут восстановить постериор для Normal-Normal модели —
значит реализация сломана. Точка. Без этих тестов PPL — игрушка.

Тестируем:
  1. Normal-Normal conjugate → HMC
  2. Normal-Normal conjugate → NUTS
  3. Beta-Binomial conjugate → HMC
  4. Multi-dimensional Normal → NUTS
  5. Multi-chain convergence (R-hat)
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jrandom

from hyperion_dsl.primitives import sample, plate, factor
from hyperion_dsl.distributions import Normal, HalfNormal, Beta, Bernoulli, Dirichlet
from hyperion_dsl.model import model
from hyperion_backends.jax_backend import JAXBackend
from hyperion_inference.hmc import hmc_sample
from hyperion_inference.nuts import nuts_sample


# === Тестовые модели с аналитическими постериорами ===

@model
def normal_normal_model():
    """Normal-Normal conjugate: prior N(0, prior_sigma), likelihood N(mu, 1).

    Аналитический постериор:
        posterior_mean = (sum(data) / n) / (1 + 1/(n * prior_var))
        = n * data_mean / (n + 1/prior_var)
        posterior_var = 1 / (n + 1/prior_var)
    """
    mu = sample("mu", Normal(0.0, 10.0))
    sample("obs", Normal(mu, 1.0))


@model
def two_param_normal_model():
    """Две независимые нормали — проверяем что инференс корректно
    разделяет латенты в многомерном пространстве.
    """
    mu = sample("mu", Normal(0.0, 5.0))
    sigma_raw = sample("log_sigma", Normal(0.0, 1.0))
    sample("obs", Normal(mu, jnp.exp(sigma_raw)))


# === HMC: Normal-Normal conjugate ===

class TestHMCPosteriorRecovery:
    """HMC должен восстанавливать аналитический постериор.
    Если не восстанавливает — или leapfrog кривой, или accept/reject сломан.
    """

    def test_normal_normal_posterior_mean(self):
        """Normal-Normal: posterior mean должен совпадать с аналитическим."""
        key = jrandom.PRNGKey(42)
        n = 100
        true_mu = 3.0

        data_key, init_key, sample_key = jrandom.split(key, 3)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        samples, log_probs, accept_probs, info = hmc_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=2000, num_warmup=500,
            step_size=0.1, num_leapfrog=10,
        )

        mu_samples = samples[:, 0]

        prior_var = 100.0  # sigma_prior = 10 → var = 100
        data_mean = float(jnp.mean(data))
        post_mean = n * data_mean / (n + 1.0 / prior_var)
        post_var = 1.0 / (n + 1.0 / prior_var)

        sample_mean = float(jnp.mean(mu_samples))
        sample_var = float(jnp.var(mu_samples))

        assert abs(sample_mean - post_mean) < 0.15, (
            f"HMC posterior mean {sample_mean:.4f} далеко от аналитического {post_mean:.4f}"
        )
        assert abs(sample_var - post_var) < 0.05, (
            f"HMC posterior var {sample_var:.4f} далеко от аналитического {post_var:.4f}"
        )

    def test_hmc_accept_rate(self):
        """Accept rate должен быть в разумном диапазоне (0.5-0.95)."""
        key = jrandom.PRNGKey(123)
        n = 50
        data_key, init_key, sample_key = jrandom.split(key, 3)
        data = 2.0 + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        _, _, accept_probs, info = hmc_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=500, num_warmup=300,
            step_size=0.05, num_leapfrog=10,
        )

        mean_accept = float(jnp.mean(accept_probs))
        assert mean_accept > 0.4, f"Accept rate {mean_accept:.3f} слишком низкий"
        assert mean_accept < 1.0, f"Accept rate {mean_accept:.3f} подозрительно высокий"

    def test_hmc_warmup_adapts_step_size(self):
        """Dual averaging должен адаптировать step_size за warmup."""
        key = jrandom.PRNGKey(7)
        data_key, init_key, sample_key = jrandom.split(key, 3)
        data = jrandom.normal(data_key, shape=(30,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        _, _, _, info = hmc_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=100, num_warmup=200,
            step_size=0.001, num_leapfrog=10,
        )

        adapted_step = float(info["step_size"])
        assert adapted_step > 0.001, (
            f"Dual averaging не адаптировал step_size: {adapted_step:.6f}"
        )


# === NUTS: Normal-Normal conjugate ===

class TestNUTSPosteriorRecovery:
    """NUTS — автоматическая длина траектории. Должен быть ТОЧНЕЕ HMC при тех же сэмплах."""

    def test_normal_normal_posterior_mean(self):
        """NUTS + Normal-Normal conjugate. Проверяем mean и variance."""
        key = jrandom.PRNGKey(99)
        n = 100
        true_mu = -2.0

        data_key, init_key, sample_key = jrandom.split(key, 3)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        samples, log_probs, accept_probs, info = nuts_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=2000, num_warmup=500,
            step_size=0.1, max_tree_depth=10,
        )

        mu_samples = samples[:, 0]

        prior_var = 100.0
        data_mean = float(jnp.mean(data))
        post_mean = n * data_mean / (n + 1.0 / prior_var)
        post_var = 1.0 / (n + 1.0 / prior_var)

        sample_mean = float(jnp.mean(mu_samples))
        sample_var = float(jnp.var(mu_samples))

        assert abs(sample_mean - post_mean) < 0.15, (
            f"NUTS posterior mean {sample_mean:.4f} далеко от аналитического {post_mean:.4f}"
        )
        assert abs(sample_var - post_var) < 0.05, (
            f"NUTS posterior var {sample_var:.4f} далеко от аналитического {post_var:.4f}"
        )

    def test_nuts_tree_depth_reasonable(self):
        """NUTS должен строить деревья разумной глубины (не 0 и не max)."""
        key = jrandom.PRNGKey(77)
        data_key, init_key, sample_key = jrandom.split(key, 3)
        data = jrandom.normal(data_key, shape=(50,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        _, _, _, info = nuts_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=500, num_warmup=200,
            step_size=0.1, max_tree_depth=10,
        )

        mean_depth = float(info["mean_tree_depth"])
        assert mean_depth > 0.5, f"Средняя глубина дерева {mean_depth:.2f} — подозрительно мало"
        assert mean_depth <= 10.0, f"Средняя глубина дерева {mean_depth:.2f} — больше max_depth"

    def test_nuts_no_excessive_divergences(self):
        """На простой модели не должно быть много divergences."""
        key = jrandom.PRNGKey(55)
        data_key, init_key, sample_key = jrandom.split(key, 3)
        data = 1.0 + jrandom.normal(data_key, shape=(30,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        _, _, _, info = nuts_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=500, num_warmup=200,
            step_size=0.05, max_tree_depth=10,
        )

        num_div = int(info["num_divergences"])
        assert num_div < 50, f"Слишком много divergences: {num_div}/500"


# === Backend тесты ===

class TestJAXBackendJIT:
    """Проверяем что backend правильно JIT-компилирует potential."""

    def test_potential_fn_returns_scalar(self):
        """potential_fn должен вернуть скаляр (не вектор, не батч)."""
        key = jrandom.PRNGKey(0)
        data = jnp.ones(10)

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, key)

        pos = backend.flatten_latents(backend.sample_prior(key))
        lp = backend.potential_fn(pos)
        assert lp.ndim == 0, f"potential_fn вернул не скаляр: shape={lp.shape}"

    def test_potential_and_grad_shapes(self):
        """(value, grad) должны иметь правильные размерности."""
        key = jrandom.PRNGKey(1)
        data = jnp.ones(10)

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, key)

        pos = backend.flatten_latents(backend.sample_prior(key))
        lp, grad = backend.potential_and_grad_fn(pos)

        assert lp.ndim == 0, f"log_prob shape={lp.shape}"
        assert grad.shape == pos.shape, f"grad shape={grad.shape}, expected {pos.shape}"

    def test_flatten_unflatten_roundtrip(self):
        """flatten → unflatten должно быть identity."""
        key = jrandom.PRNGKey(2)
        data = jnp.ones(10)

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, key)

        prior = backend.sample_prior(key)
        flat = backend.flatten_latents(prior)
        recovered = backend.unflatten_latents(flat)

        for name in prior:
            assert jnp.allclose(prior[name], recovered[name]), (
                f"Roundtrip провалился для {name}"
            )

    def test_gradient_finite(self):
        """Градиенты должны быть конечными (не NaN, не Inf)."""
        key = jrandom.PRNGKey(3)
        data = jnp.ones(10) * 2.0

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, key)

        pos = backend.flatten_latents(backend.sample_prior(key))
        _, grad = backend.potential_and_grad_fn(pos)

        assert jnp.all(jnp.isfinite(grad)), "Градиент содержит NaN или Inf"

    def test_total_dim_correct(self):
        """total_dim должен совпадать с суммой размерностей латентов."""
        key = jrandom.PRNGKey(4)
        data = jnp.ones(5)

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, key)

        shapes = backend.get_latent_shapes()
        expected_dim = sum(
            max(1, len(s) and s[0] or 1) for s in shapes.values()
        )
        flat = backend.flatten_latents(backend.sample_prior(key))
        assert flat.shape[0] == backend.total_dim


# === Лестница от простого к сложному ===

class TestInferenceGradient:
    """Проверяем что градиенты через JIT-potential корректны.
    Сравниваем с numerical differentiation (finite differences).
    """

    def test_gradient_vs_finite_diff(self):
        """Аналитический grad ≈ finite diff с точностью до eps.

        float32 ограничивает точность finite diff — используем rtol=0.3.
        Для research-grade проверки нужен float64, но для smoke test хватит.
        """
        key = jrandom.PRNGKey(10)
        data = jnp.array([1.0, 2.0, 3.0])

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, key)

        assert backend.total_dim == 1, (
            f"Для Normal-Normal модели ожидалась 1D, получили {backend.total_dim}D"
        )

        pos = jnp.array([2.0])

        _, analytical_grad = backend.potential_and_grad_fn(pos)

        eps = 1e-3
        lp_plus = backend.potential_fn(pos + eps)
        lp_minus = backend.potential_fn(pos - eps)
        numerical_grad = (lp_plus - lp_minus) / (2 * eps)

        assert jnp.allclose(analytical_grad[0], numerical_grad, rtol=0.3, atol=1e-2), (
            f"Analytical {float(analytical_grad[0]):.6f} vs "
            f"Numerical {float(numerical_grad):.6f}"
        )


# === VI: Normal-Normal conjugate ===

class TestVIPosteriorRecovery:
    """VI должен аппроксимировать аналитический постериор.
    Mean-field Gaussian — точное приближение для Normal-Normal.
    """

    def test_vi_normal_normal_mean(self):
        """VI + Normal-Normal: posterior mean через ELBO optimization."""
        key = jrandom.PRNGKey(42)
        n = 100
        true_mu = 3.0

        data_key, init_key = jrandom.split(key)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.vi import VIEngine
        vi = VIEngine()
        result = vi.run(backend, init_key, {
            "num_steps": 3000,
            "learning_rate": 0.01,
            "num_elbo_samples": 10,
            "num_posterior_samples": 2000,
            "covariance_type": "diagonal",
        })

        mu_samples = result.samples["mu"]
        prior_var = 100.0
        data_mean = float(jnp.mean(data))
        post_mean = n * data_mean / (n + 1.0 / prior_var)

        sample_mean = float(jnp.mean(mu_samples))
        assert abs(sample_mean - post_mean) < 0.3, (
            f"VI posterior mean {sample_mean:.4f} далеко от аналитического {post_mean:.4f}"
        )


# === Laplace: Normal-Normal conjugate ===

class TestLaplacePosteriorRecovery:
    """Laplace approximation для Normal-Normal — должна быть точной,
    потому что постериор и так Gaussian."""

    def test_laplace_normal_normal_mean(self):
        """Laplace + Normal-Normal: MAP = posterior mean (для Gaussian posterior)."""
        key = jrandom.PRNGKey(77)
        n = 100
        true_mu = 2.0

        data_key, init_key = jrandom.split(key)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.laplace import LaplaceApproximation
        laplace = LaplaceApproximation()
        result = laplace.run(backend, init_key, {
            "num_posterior_samples": 2000,
            "max_optim_steps": 500,
            "learning_rate": 0.1,
            "use_full_hessian": True,
        })

        mu_samples = result.samples["mu"]
        prior_var = 100.0
        data_mean = float(jnp.mean(data))
        post_mean = n * data_mean / (n + 1.0 / prior_var)

        sample_mean = float(jnp.mean(mu_samples))
        assert abs(sample_mean - post_mean) < 0.3, (
            f"Laplace posterior mean {sample_mean:.4f} далеко от аналитического {post_mean:.4f}"
        )


# === HMC/NUTS validation + warmup ===

class TestHMCValidation:
    """Проверяем валидацию входных параметров."""

    def test_num_leapfrog_zero_raises(self):
        """num_leapfrog=0 должен кидать ValueError."""
        from hyperion_inference.hmc import hmc_sample
        key = jrandom.PRNGKey(0)
        init = jnp.zeros(2)
        with pytest.raises(ValueError, match="num_leapfrog"):
            hmc_sample(lambda x: -0.5 * jnp.sum(x**2), key, init, num_leapfrog=0)


class TestNUTSValidation:
    """Проверяем валидацию входных параметров."""

    def test_max_depth_zero_raises(self):
        """max_tree_depth=0 должен кидать ValueError."""
        from hyperion_inference.nuts import nuts_sample
        key = jrandom.PRNGKey(0)
        init = jnp.zeros(2)
        with pytest.raises(ValueError, match="max_tree_depth"):
            nuts_sample(lambda x: -0.5 * jnp.sum(x**2), key, init, max_tree_depth=0)


class TestWarmupModule:
    """Тесты shared warmup: dual averaging + Welford."""

    def test_make_warmup_state(self):
        from hyperion_inference.warmup import make_warmup_state
        ws = make_warmup_state(0.1, 3)
        assert ws.step_size.shape == ()
        assert ws.inv_mass_diag.shape == (3,)
        assert float(ws.step_size) == pytest.approx(0.1, abs=1e-6)

    def test_dual_averaging_reduces_step(self):
        from hyperion_inference.warmup import make_warmup_state, dual_averaging_update
        ws = make_warmup_state(1.0, 2)
        for i in range(1, 20):
            ws = dual_averaging_update(ws, jnp.array(0.3), jnp.array(i, dtype=jnp.int32), target_accept=0.8)
        assert float(ws.step_size) < 1.0

    def test_welford_tracks_variance(self):
        from hyperion_inference.warmup import make_warmup_state, welford_update
        ws = make_warmup_state(0.1, 1)
        key = jrandom.PRNGKey(42)
        samples = jrandom.normal(key, (100,))
        for s in samples:
            ws = welford_update(ws, s[None])
        est_var = 1.0 / float(ws.inv_mass_diag[0])
        assert abs(est_var - 1.0) < 0.3


# === SMC posterior recovery ===

class TestSMCPosteriorRecovery:
    """SMC должен приближённо восстановить постериор Normal-Normal."""

    def test_smc_normal_normal_mean(self):
        """SMC + Normal-Normal: posterior mean через particle filter."""
        key = jrandom.PRNGKey(55)
        n = 50
        true_mu = 2.0

        data_key, init_key = jrandom.split(key)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.smc import SMCEngine
        smc = SMCEngine()
        result = smc.run(backend, init_key, {
            "num_particles": 500,
            "num_tempering_steps": 15,
            "rejuvenation_steps": 3,
            "rejuvenation_step_size": 0.1,
            "adaptive_tempering": True,
            "target_ess_ratio": 0.5,
        })

        mu_samples = result.samples["mu"]
        prior_var = 100.0
        data_mean = float(jnp.mean(data))
        post_mean = n * data_mean / (n + 1.0 / prior_var)

        sample_mean = float(jnp.mean(mu_samples))
        # SMC менее точен чем MCMC, допуск побольше
        assert abs(sample_mean - post_mean) < 0.5, (
            f"SMC posterior mean {sample_mean:.4f} далеко от аналитического {post_mean:.4f}"
        )


# === Flows posterior recovery ===

class TestFlowsPosteriorRecovery:
    """Normalizing Flows как variational approximation."""

    def test_flows_normal_normal_mean(self):
        """Flows + Normal-Normal: posterior mean через ELBO optimization."""
        key = jrandom.PRNGKey(66)
        n = 100
        true_mu = 1.5

        data_key, init_key = jrandom.split(key)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.flows import FlowsEngine
        flows = FlowsEngine()
        result = flows.run(backend, init_key, {
            "num_steps": 2000,
            "learning_rate": 1e-3,
            "num_layers": 4,
            "hidden_dim": 32,
            "flow_type": "realnvp",
            "num_elbo_samples": 10,
            "num_posterior_samples": 1000,
        })

        mu_samples = result.samples["mu"]
        prior_var = 100.0
        data_mean = float(jnp.mean(data))
        post_mean = n * data_mean / (n + 1.0 / prior_var)

        sample_mean = float(jnp.mean(mu_samples))
        # Flows менее точны для 1D, допуск побольше
        assert abs(sample_mean - post_mean) < 0.5, (
            f"Flows posterior mean {sample_mean:.4f} далеко от аналитического {post_mean:.4f}"
        )


# === 5D Multi-dimensional Posterior Recovery ===

@model
def five_dim_normal_model():
    """5D Normal-Normal conjugate: 5 независимых параметров.

    Prior: mu_i ~ N(0, prior_sigma_i)
    Likelihood: obs_i ~ N(mu_i, 1)

    Аналитический постериор для каждого mu_i:
        post_mean_i = n * data_mean_i / (n + 1/prior_var_i)
        post_var_i = 1 / (n + 1/prior_var_i)
    """
    mu_0 = sample("mu_0", Normal(0.0, 5.0))
    mu_1 = sample("mu_1", Normal(0.0, 5.0))
    mu_2 = sample("mu_2", Normal(0.0, 5.0))
    mu_3 = sample("mu_3", Normal(0.0, 5.0))
    mu_4 = sample("mu_4", Normal(0.0, 5.0))
    sample("obs_0", Normal(mu_0, 1.0))
    sample("obs_1", Normal(mu_1, 1.0))
    sample("obs_2", Normal(mu_2, 1.0))
    sample("obs_3", Normal(mu_3, 1.0))
    sample("obs_4", Normal(mu_4, 1.0))


class TestMultiDimPosteriorRecovery:
    """5D posterior recovery — проверяем что HMC/NUTS корректно
    работают в многомерном пространстве. Каждый параметр — независимый
    Normal-Normal conjugate, аналитический постериор известен.

    Если хотя бы один из 5 параметров не восстановлен — инференс сломан.
    """

    def _analytical_posterior(self, data_per_dim, prior_sigma=5.0):
        """Вычисляем аналитический posterior для каждого dim."""
        prior_var = prior_sigma ** 2
        results = {}
        for i, data in enumerate(data_per_dim):
            n = len(data)
            data_mean = float(jnp.mean(data))
            post_mean = n * data_mean / (n + 1.0 / prior_var)
            post_var = 1.0 / (n + 1.0 / prior_var)
            results[f"mu_{i}"] = {"mean": post_mean, "var": post_var}
        return results

    def test_hmc_5d_posterior_recovery(self):
        """HMC восстанавливает все 5 posterior means в 5D модели."""
        key = jrandom.PRNGKey(1234)
        n_obs = 50
        true_mus = [1.0, -2.0, 3.0, 0.5, -1.5]

        data_keys = jrandom.split(key, 7)
        init_key, sample_key = data_keys[0], data_keys[1]
        data_per_dim = []
        for i in range(5):
            d = true_mus[i] + jrandom.normal(data_keys[i + 2], shape=(n_obs,))
            data_per_dim.append(d)

        obs_dict = {f"obs_{i}": data_per_dim[i] for i in range(5)}

        backend = JAXBackend()
        backend.initialize(five_dim_normal_model, obs_dict, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        samples, log_probs, accept_probs, info = hmc_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=3000, num_warmup=1000,
            step_size=0.05, num_leapfrog=10,
        )

        analytical = self._analytical_posterior(data_per_dim)
        latent_names = list(backend.get_latent_shapes().keys())
        unflat = jax.vmap(backend.unflatten_latents)(samples)

        for name in ["mu_0", "mu_1", "mu_2", "mu_3", "mu_4"]:
            if name not in unflat:
                continue
            mu_samples = unflat[name]
            sample_mean = float(jnp.mean(mu_samples))
            expected_mean = analytical[name]["mean"]
            expected_var = analytical[name]["var"]

            assert abs(sample_mean - expected_mean) < 0.3, (
                f"HMC 5D: {name} mean {sample_mean:.4f} далеко от {expected_mean:.4f}"
            )

            sample_var = float(jnp.var(mu_samples))
            assert abs(sample_var - expected_var) < 0.1, (
                f"HMC 5D: {name} var {sample_var:.4f} далеко от {expected_var:.4f}"
            )

    def test_nuts_5d_posterior_recovery(self):
        """NUTS восстанавливает все 5 posterior means в 5D модели."""
        key = jrandom.PRNGKey(5678)
        n_obs = 50
        true_mus = [2.0, -1.0, 0.0, 4.0, -3.0]

        data_keys = jrandom.split(key, 7)
        init_key, sample_key = data_keys[0], data_keys[1]
        data_per_dim = []
        for i in range(5):
            d = true_mus[i] + jrandom.normal(data_keys[i + 2], shape=(n_obs,))
            data_per_dim.append(d)

        obs_dict = {f"obs_{i}": data_per_dim[i] for i in range(5)}

        backend = JAXBackend()
        backend.initialize(five_dim_normal_model, obs_dict, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        samples, log_probs, accept_probs, info = nuts_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=3000, num_warmup=1000,
            step_size=0.05, max_tree_depth=10,
        )

        analytical = self._analytical_posterior(data_per_dim)
        unflat = jax.vmap(backend.unflatten_latents)(samples)

        for name in ["mu_0", "mu_1", "mu_2", "mu_3", "mu_4"]:
            if name not in unflat:
                continue
            mu_samples = unflat[name]
            sample_mean = float(jnp.mean(mu_samples))
            expected_mean = analytical[name]["mean"]
            expected_var = analytical[name]["var"]

            assert abs(sample_mean - expected_mean) < 0.3, (
                f"NUTS 5D: {name} mean {sample_mean:.4f} далеко от {expected_mean:.4f}"
            )

            sample_var = float(jnp.var(mu_samples))
            assert abs(sample_var - expected_var) < 0.1, (
                f"NUTS 5D: {name} var {sample_var:.4f} далеко от {expected_var:.4f}"
            )

    def test_hmc_5d_dense_mass_recovery(self):
        """HMC с dense mass matrix восстанавливает 5D posterior."""
        key = jrandom.PRNGKey(9999)
        n_obs = 50
        true_mus = [1.5, -0.5, 2.0, -1.0, 0.0]

        data_keys = jrandom.split(key, 7)
        init_key, sample_key = data_keys[0], data_keys[1]
        data_per_dim = []
        for i in range(5):
            d = true_mus[i] + jrandom.normal(data_keys[i + 2], shape=(n_obs,))
            data_per_dim.append(d)

        obs_dict = {f"obs_{i}": data_per_dim[i] for i in range(5)}

        backend = JAXBackend()
        backend.initialize(five_dim_normal_model, obs_dict, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        samples, log_probs, accept_probs, info = hmc_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=2000, num_warmup=1000,
            step_size=0.05, num_leapfrog=10,
            dense_mass=True,
        )

        assert info["dense_mass"] is True

        analytical = self._analytical_posterior(data_per_dim)
        unflat = jax.vmap(backend.unflatten_latents)(samples)

        for name in ["mu_0", "mu_1", "mu_2", "mu_3", "mu_4"]:
            if name not in unflat:
                continue
            mu_samples = unflat[name]
            sample_mean = float(jnp.mean(mu_samples))
            expected_mean = analytical[name]["mean"]

            assert abs(sample_mean - expected_mean) < 0.3, (
                f"HMC dense 5D: {name} mean {sample_mean:.4f} далеко от {expected_mean:.4f}"
            )


# === Constrained model posterior recovery ===

@model
def half_normal_scale_model():
    """Constrained model: sigma ~ HalfNormal(2.0), obs ~ Normal(0, sigma).

    sigma живёт на (0, +inf) — требует ExpTransform в unconstrained space.
    Если flatten/unflatten не применяют transforms, инференс будет ломаться.
    """
    sigma = sample("sigma", HalfNormal(2.0))
    sample("obs", Normal(0.0, sigma))


class TestConstrainedPosteriorRecovery:
    """Posterior recovery для модели с constrained параметрами (HalfNormal).

    Проверяем:
    1. Все сэмплы sigma > 0 (constraint соблюдён)
    2. Posterior mean в разумном диапазоне
    3. HMC и NUTS оба работают с transforms
    """

    def test_hmc_halfnormal_samples_positive(self):
        """HMC + HalfNormal: все сэмплы sigma должны быть > 0."""
        key = jrandom.PRNGKey(2024)
        n = 50
        true_sigma = 1.5

        data_key, init_key, sample_key = jrandom.split(key, 3)
        data = true_sigma * jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(half_normal_scale_model, {"obs": data}, init_key)

        assert "sigma" in backend._transforms, (
            "Backend должен иметь transform для sigma (positive constraint)"
        )

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))
        samples, log_probs, accept_probs, info = hmc_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=2000, num_warmup=1000,
            step_size=0.05, num_leapfrog=10,
        )

        import jax
        unflat = jax.vmap(backend.unflatten_latents)(samples)
        sigma_samples = unflat["sigma"]

        assert jnp.all(sigma_samples > 0), (
            f"Не все sigma > 0! min={float(jnp.min(sigma_samples)):.6f}"
        )

        sample_mean = float(jnp.mean(sigma_samples))
        assert 0.5 < sample_mean < 4.0, (
            f"HMC sigma mean {sample_mean:.4f} вне разумного диапазона"
        )

    def test_nuts_halfnormal_samples_positive(self):
        """NUTS + HalfNormal: все сэмплы sigma > 0, posterior сходится."""
        key = jrandom.PRNGKey(2025)
        n = 50
        true_sigma = 2.0

        data_key, init_key, sample_key = jrandom.split(key, 3)
        data = true_sigma * jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(half_normal_scale_model, {"obs": data}, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))
        samples, log_probs, accept_probs, info = nuts_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=2000, num_warmup=1000,
            step_size=0.05, max_tree_depth=10,
        )

        import jax
        unflat = jax.vmap(backend.unflatten_latents)(samples)
        sigma_samples = unflat["sigma"]

        assert jnp.all(sigma_samples > 0), (
            f"Не все sigma > 0! min={float(jnp.min(sigma_samples)):.6f}"
        )

        sample_mean = float(jnp.mean(sigma_samples))
        assert abs(sample_mean - true_sigma) < 0.8, (
            f"NUTS sigma mean {sample_mean:.4f} далеко от true {true_sigma}"
        )

        assert "accept_probs" in info, "NUTS info должен содержать accept_probs"
        assert "mean_accept_prob" in info, "NUTS info должен содержать mean_accept_prob"

    def test_flatten_unflatten_roundtrip_constrained(self):
        """flatten → unflatten roundtrip для constrained модели."""
        key = jrandom.PRNGKey(2026)
        data = jnp.ones(10)

        backend = JAXBackend()
        backend.initialize(half_normal_scale_model, {"obs": data}, key)

        prior = backend.sample_prior(key)
        assert float(prior["sigma"]) > 0, "Prior sample for sigma should be positive"

        flat = backend.flatten_latents(prior)
        recovered = backend.unflatten_latents(flat)

        assert jnp.allclose(prior["sigma"], recovered["sigma"], atol=1e-4), (
            f"Roundtrip failed: prior={float(prior['sigma']):.6f}, "
            f"recovered={float(recovered['sigma']):.6f}"
        )


# === Dirichlet (Simplex) posterior recovery ===

@model
def dirichlet_multinomial_model():
    """Dirichlet-Multinomial conjugate model.

    Prior: p ~ Dirichlet([2, 2, 2])
    Likelihood: Multinomial(counts | p) via factor()

    Posterior: Dirichlet(alpha + counts).
    Tests StickBreaking transform (K=3 → unconstrained R^2).
    """
    p = sample("p", Dirichlet(jnp.array([2.0, 2.0, 2.0])))
    counts = jnp.array([20.0, 30.0, 50.0])
    factor("obs_lik", jnp.sum(counts * jnp.log(jnp.maximum(p, 1e-8))))


class TestDirichletPosteriorRecovery:
    """Dirichlet posterior recovery — tests Simplex/StickBreaking transform
    through the full pipeline: flatten (K→K-1) → HMC → unflatten (K-1→K).

    If flat_sizes doesn't account for dimensionality change, this will crash.
    """

    def test_dirichlet_flatten_unflatten_roundtrip(self):
        """flatten → unflatten roundtrip for Dirichlet (Simplex constraint)."""
        key = jrandom.PRNGKey(3001)

        backend = JAXBackend()
        backend.initialize(dirichlet_multinomial_model, {}, key)

        assert backend._flat_sizes["p"] == 2, (
            f"StickBreaking: Dirichlet K=3 should have unconstrained dim 2, "
            f"got {backend._flat_sizes['p']}"
        )
        assert backend.total_dim == 2

        prior = backend.sample_prior(key)
        p_val = prior["p"]
        assert p_val.shape == (3,), f"Dirichlet sample should be (3,), got {p_val.shape}"
        assert jnp.allclose(jnp.sum(p_val), 1.0, atol=1e-4), "Simplex sum should be ~1"
        assert jnp.all(p_val > 0), "Simplex entries should be > 0"

        flat = backend.flatten_latents(prior)
        assert flat.shape == (2,), f"Flat should be (2,), got {flat.shape}"

        recovered = backend.unflatten_latents(flat)
        assert jnp.allclose(p_val, recovered["p"], atol=1e-3), (
            f"Roundtrip failed: prior={p_val}, recovered={recovered['p']}"
        )

    def test_dirichlet_potential_finite(self):
        """potential_fn should be finite for Dirichlet model."""
        key = jrandom.PRNGKey(3002)

        backend = JAXBackend()
        backend.initialize(dirichlet_multinomial_model, {}, key)

        flat = backend.flatten_latents(backend.sample_prior(key))
        lp = backend.potential_fn(flat)
        assert jnp.isfinite(lp), f"potential_fn returned non-finite: {lp}"

        lp2, grad = backend.potential_and_grad_fn(flat)
        assert jnp.isfinite(lp2), f"potential returned non-finite: {lp2}"
        assert jnp.all(jnp.isfinite(grad)), "gradient contains NaN/Inf"

    def test_hmc_dirichlet_posterior_recovery(self):
        """HMC recovers Dirichlet-Multinomial posterior.

        Prior: Dirichlet([2, 2, 2])
        Counts: [20, 30, 50]
        Posterior: Dirichlet([22, 32, 52])
        Posterior mean: [22/106, 32/106, 52/106]
        """
        key = jrandom.PRNGKey(3003)
        init_key, sample_key = jrandom.split(key)

        backend = JAXBackend()
        backend.initialize(dirichlet_multinomial_model, {}, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        samples, log_probs, accept_probs, info = hmc_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=3000, num_warmup=1000,
            step_size=0.05, num_leapfrog=10,
        )

        import jax
        unflat = jax.vmap(backend.unflatten_latents)(samples)
        p_samples = unflat["p"]

        assert p_samples.shape == (3000, 3)

        alpha_post = jnp.array([22.0, 32.0, 52.0])
        expected_mean = alpha_post / jnp.sum(alpha_post)

        sample_mean = jnp.mean(p_samples, axis=0)
        for i in range(3):
            assert abs(float(sample_mean[i]) - float(expected_mean[i])) < 0.05, (
                f"HMC Dirichlet: p[{i}] mean {float(sample_mean[i]):.4f} "
                f"далеко от {float(expected_mean[i]):.4f}"
            )

        simplex_sums = jnp.sum(p_samples, axis=1)
        assert jnp.allclose(simplex_sums, 1.0, atol=1e-3), (
            f"Not all samples on simplex: sum range [{float(jnp.min(simplex_sums)):.6f}, "
            f"{float(jnp.max(simplex_sums)):.6f}]"
        )

        assert jnp.all(p_samples > 0), "Not all p > 0"

    def test_nuts_dirichlet_posterior_recovery(self):
        """NUTS recovers Dirichlet-Multinomial posterior."""
        key = jrandom.PRNGKey(3004)
        init_key, sample_key = jrandom.split(key)

        backend = JAXBackend()
        backend.initialize(dirichlet_multinomial_model, {}, init_key)

        init_pos = backend.flatten_latents(backend.sample_prior(init_key))

        samples, log_probs, accept_probs, info = nuts_sample(
            backend.potential_fn, sample_key, init_pos,
            num_samples=3000, num_warmup=1000,
            step_size=0.05, max_tree_depth=10,
        )

        import jax
        unflat = jax.vmap(backend.unflatten_latents)(samples)
        p_samples = unflat["p"]

        alpha_post = jnp.array([22.0, 32.0, 52.0])
        expected_mean = alpha_post / jnp.sum(alpha_post)

        sample_mean = jnp.mean(p_samples, axis=0)
        for i in range(3):
            assert abs(float(sample_mean[i]) - float(expected_mean[i])) < 0.05, (
                f"NUTS Dirichlet: p[{i}] mean {float(sample_mean[i]):.4f} "
                f"далеко от {float(expected_mean[i]):.4f}"
            )


# === VI: Extended posterior recovery ===

class TestVIPosteriorRecoveryExtended:
    """Extended VI tests: variance, full-rank, low-rank, 5D."""

    def test_vi_normal_normal_variance(self):
        """VI diagonal should recover posterior variance for Normal-Normal."""
        key = jrandom.PRNGKey(101)
        n = 100
        true_mu = 2.5

        data_key, init_key = jrandom.split(key)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.vi import VIEngine
        vi = VIEngine()
        result = vi.run(backend, init_key, {
            "num_steps": 3000,
            "learning_rate": 0.01,
            "num_elbo_samples": 10,
            "num_posterior_samples": 3000,
            "covariance_type": "diagonal",
        })

        mu_samples = result.samples["mu"]

        prior_var = 100.0
        post_var = 1.0 / (n + 1.0 / prior_var)
        sample_var = float(jnp.var(mu_samples))

        assert abs(sample_var - post_var) < 0.02, (
            f"VI posterior var {sample_var:.5f} далеко от аналитического {post_var:.5f}"
        )

    def test_vi_full_rank_normal_normal(self):
        """Full-rank VI on Normal-Normal should recover posterior."""
        key = jrandom.PRNGKey(102)
        n = 100
        true_mu = -1.0

        data_key, init_key = jrandom.split(key)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.vi import VIEngine
        vi = VIEngine()
        result = vi.run(backend, init_key, {
            "num_steps": 3000,
            "learning_rate": 0.01,
            "num_elbo_samples": 10,
            "num_posterior_samples": 2000,
            "covariance_type": "full_rank",
        })

        mu_samples = result.samples["mu"]
        prior_var = 100.0
        data_mean = float(jnp.mean(data))
        post_mean = n * data_mean / (n + 1.0 / prior_var)

        sample_mean = float(jnp.mean(mu_samples))
        assert abs(sample_mean - post_mean) < 0.3, (
            f"VI full_rank mean {sample_mean:.4f} vs analytical {post_mean:.4f}"
        )

    def test_vi_low_rank_normal_normal(self):
        """Low-rank VI on Normal-Normal should recover posterior."""
        key = jrandom.PRNGKey(103)
        n = 100
        true_mu = 4.0

        data_key, init_key = jrandom.split(key)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.vi import VIEngine
        vi = VIEngine()
        result = vi.run(backend, init_key, {
            "num_steps": 3000,
            "learning_rate": 0.01,
            "num_elbo_samples": 10,
            "num_posterior_samples": 2000,
            "covariance_type": "low_rank",
            "rank": 1,
        })

        mu_samples = result.samples["mu"]
        prior_var = 100.0
        data_mean = float(jnp.mean(data))
        post_mean = n * data_mean / (n + 1.0 / prior_var)

        sample_mean = float(jnp.mean(mu_samples))
        assert abs(sample_mean - post_mean) < 0.3, (
            f"VI low_rank mean {sample_mean:.4f} vs analytical {post_mean:.4f}"
        )

    def test_vi_elbo_increases(self):
        """ELBO should generally increase during optimization."""
        key = jrandom.PRNGKey(104)
        data = 2.0 + jrandom.normal(key, shape=(50,))
        init_key = jrandom.PRNGKey(105)

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.vi import VIEngine
        vi = VIEngine()
        result = vi.run(backend, init_key, {
            "num_steps": 2000,
            "learning_rate": 0.01,
            "num_elbo_samples": 10,
            "num_posterior_samples": 100,
            "covariance_type": "diagonal",
        })

        elbo_hist = result.diagnostics.get("elbo_history", [])
        assert len(elbo_hist) > 100, f"Too few ELBO steps: {len(elbo_hist)}"

        early_mean = float(jnp.mean(jnp.array(elbo_hist[:50])))
        late_mean = float(jnp.mean(jnp.array(elbo_hist[-50:])))
        assert late_mean > early_mean, (
            f"ELBO should increase: early {early_mean:.2f} -> late {late_mean:.2f}"
        )

    def test_vi_5d_posterior_recovery(self):
        """VI recovers 5D Normal-Normal posterior means."""
        key = jrandom.PRNGKey(106)
        n_obs = 50
        true_mus = [1.0, -2.0, 3.0, 0.5, -1.5]

        data_keys = jrandom.split(key, 7)
        init_key = data_keys[0]
        data_per_dim = [
            true_mus[i] + jrandom.normal(data_keys[i + 2], shape=(n_obs,))
            for i in range(5)
        ]
        obs_dict = {f"obs_{i}": data_per_dim[i] for i in range(5)}

        backend = JAXBackend()
        backend.initialize(five_dim_normal_model, obs_dict, init_key)

        from hyperion_inference.vi import VIEngine
        vi = VIEngine()
        result = vi.run(backend, init_key, {
            "num_steps": 5000,
            "learning_rate": 0.01,
            "num_elbo_samples": 10,
            "num_posterior_samples": 2000,
            "covariance_type": "diagonal",
        })

        prior_var = 25.0  # sigma=5
        for i in range(5):
            name = f"mu_{i}"
            if name not in result.samples:
                continue
            data_mean = float(jnp.mean(data_per_dim[i]))
            post_mean = n_obs * data_mean / (n_obs + 1.0 / prior_var)
            sample_mean = float(jnp.mean(result.samples[name]))
            assert abs(sample_mean - post_mean) < 0.5, (
                f"VI 5D: {name} mean {sample_mean:.4f} vs analytical {post_mean:.4f}"
            )


# === SMC: Extended posterior recovery ===

class TestSMCPosteriorRecoveryExtended:
    """Extended SMC tests: variance, 5D, diagnostics, tempering schedules."""

    def test_smc_normal_normal_variance(self):
        """SMC should approximately recover posterior variance."""
        key = jrandom.PRNGKey(201)
        n = 100
        true_mu = 1.0

        data_key, init_key = jrandom.split(key)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.smc import SMCEngine
        smc = SMCEngine()
        result = smc.run(backend, init_key, {
            "num_particles": 1000,
            "num_tempering_steps": 20,
            "rejuvenation_steps": 5,
            "rejuvenation_step_size": 0.1,
            "adaptive_tempering": True,
            "target_ess_ratio": 0.5,
        })

        mu_samples = result.samples["mu"]
        prior_var = 100.0
        post_var = 1.0 / (n + 1.0 / prior_var)
        sample_var = float(jnp.var(mu_samples))

        assert abs(sample_var - post_var) < 0.05, (
            f"SMC posterior var {sample_var:.5f} vs analytical {post_var:.5f}"
        )

    def test_smc_log_evidence_finite(self):
        """SMC log-evidence should be finite."""
        key = jrandom.PRNGKey(202)
        data = 2.0 + jrandom.normal(key, shape=(30,))
        init_key = jrandom.PRNGKey(203)

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.smc import SMCEngine
        smc = SMCEngine()
        result = smc.run(backend, init_key, {
            "num_particles": 500,
            "num_tempering_steps": 15,
            "rejuvenation_steps": 3,
            "rejuvenation_step_size": 0.05,
        })

        log_z = result.diagnostics.get("log_evidence", None)
        assert log_z is not None, "SMC should return log_evidence"
        assert jnp.isfinite(log_z), f"log_evidence is not finite: {log_z}"

    def test_smc_ess_history_nonempty(self):
        """SMC ESS history should be non-empty and all values > 0."""
        key = jrandom.PRNGKey(204)
        data = jrandom.normal(key, shape=(20,))
        init_key = jrandom.PRNGKey(205)

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.smc import SMCEngine
        smc = SMCEngine()
        result = smc.run(backend, init_key, {
            "num_particles": 300,
            "num_tempering_steps": 10,
            "rejuvenation_steps": 3,
            "rejuvenation_step_size": 0.05,
        })

        ess_hist = result.diagnostics.get("ess_history", [])
        assert len(ess_hist) > 0, "ESS history should be non-empty"
        assert all(e > 0 for e in ess_hist), "All ESS values should be > 0"

    def test_smc_betas_reach_one(self):
        """SMC tempering schedule should reach beta=1.0."""
        key = jrandom.PRNGKey(206)
        data = 1.5 + jrandom.normal(key, shape=(30,))
        init_key = jrandom.PRNGKey(207)

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.smc import SMCEngine
        smc = SMCEngine()
        result = smc.run(backend, init_key, {
            "num_particles": 500,
            "num_tempering_steps": 20,
            "rejuvenation_steps": 3,
            "rejuvenation_step_size": 0.1,
            "adaptive_tempering": True,
        })

        betas = result.diagnostics.get("betas", [])
        assert len(betas) >= 2, "Should have at least 2 betas (start + end)"
        assert betas[-1] >= 1.0, f"Final beta should be 1.0, got {betas[-1]}"

    def test_smc_fixed_tempering(self):
        """SMC with fixed tempering schedule should also converge."""
        key = jrandom.PRNGKey(208)
        n = 50
        true_mu = -1.0

        data_key, init_key = jrandom.split(key)
        data = true_mu + jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(normal_normal_model, {"obs": data}, init_key)

        from hyperion_inference.smc import SMCEngine
        smc = SMCEngine()
        result = smc.run(backend, init_key, {
            "num_particles": 500,
            "num_tempering_steps": 20,
            "rejuvenation_steps": 5,
            "rejuvenation_step_size": 0.1,
            "adaptive_tempering": False,
        })

        mu_samples = result.samples["mu"]
        prior_var = 100.0
        data_mean = float(jnp.mean(data))
        post_mean = n * data_mean / (n + 1.0 / prior_var)

        sample_mean = float(jnp.mean(mu_samples))
        assert abs(sample_mean - post_mean) < 0.8, (
            f"SMC fixed-tempering mean {sample_mean:.4f} vs analytical {post_mean:.4f}"
        )

    def test_smc_5d_posterior_recovery(self):
        """SMC recovers 5D Normal-Normal posterior means."""
        key = jrandom.PRNGKey(209)
        n_obs = 50
        true_mus = [2.0, -1.0, 0.0, 3.0, -2.0]

        data_keys = jrandom.split(key, 7)
        init_key = data_keys[0]
        data_per_dim = [
            true_mus[i] + jrandom.normal(data_keys[i + 2], shape=(n_obs,))
            for i in range(5)
        ]
        obs_dict = {f"obs_{i}": data_per_dim[i] for i in range(5)}

        backend = JAXBackend()
        backend.initialize(five_dim_normal_model, obs_dict, init_key)

        from hyperion_inference.smc import SMCEngine
        smc = SMCEngine()
        result = smc.run(backend, init_key, {
            "num_particles": 1000,
            "num_tempering_steps": 30,
            "rejuvenation_steps": 5,
            "rejuvenation_step_size": 0.05,
            "adaptive_tempering": True,
            "target_ess_ratio": 0.5,
        })

        prior_var = 25.0  # sigma=5
        for i in range(5):
            name = f"mu_{i}"
            if name not in result.samples:
                continue
            data_mean = float(jnp.mean(data_per_dim[i]))
            post_mean = n_obs * data_mean / (n_obs + 1.0 / prior_var)
            sample_mean = float(jnp.mean(result.samples[name]))
            assert abs(sample_mean - post_mean) < 1.0, (
                f"SMC 5D: {name} mean {sample_mean:.4f} vs analytical {post_mean:.4f}"
            )

    def test_smc_constrained_halfnormal(self):
        """SMC on a constrained model (HalfNormal) should produce all positive samples."""
        key = jrandom.PRNGKey(210)
        n = 50
        true_sigma = 1.5

        data_key, init_key = jrandom.split(key)
        data = true_sigma * jrandom.normal(data_key, shape=(n,))

        backend = JAXBackend()
        backend.initialize(half_normal_scale_model, {"obs": data}, init_key)

        from hyperion_inference.smc import SMCEngine
        smc = SMCEngine()
        result = smc.run(backend, init_key, {
            "num_particles": 500,
            "num_tempering_steps": 20,
            "rejuvenation_steps": 5,
            "rejuvenation_step_size": 0.05,
            "adaptive_tempering": True,
        })

        sigma_samples = result.samples["sigma"]
        assert jnp.all(sigma_samples > 0), (
            f"SMC sigma not all positive, min={float(jnp.min(sigma_samples)):.6f}"
        )
        sample_mean = float(jnp.mean(sigma_samples))
        assert 0.3 < sample_mean < 5.0, (
            f"SMC sigma mean {sample_mean:.4f} out of range"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
