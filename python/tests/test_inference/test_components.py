"""Тесты компонентов: handlers, warmup windows, dense mass, IR.

Покрываем каждый новый компонент unit-тестами:
  - Composable handlers (TraceMessenger, SubstituteMessenger, ReplayMessenger, BlockMessenger)
  - Windowed warmup (build_adaptation_schedule, finalize_mass, windowed loop)
  - Dense mass matrix (apply_inv_mass, kinetic_energy, sample_momentum, leapfrog)
  - IR pipeline (ModelCompiler, IRGraph, compute_log_joint)

Без халтур — проверяем поведение, а не факт отсутствия ошибок.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jrandom

from hyperion_dsl.primitives import sample, plate, factor
from hyperion_dsl.distributions import Normal, HalfNormal, Bernoulli
from hyperion_dsl.model import model


# ═══════════════════════════════════════════════════════════
# Composable Effect Handlers
# ═══════════════════════════════════════════════════════════

class TestTraceMessenger:
    """TraceMessenger: запись sample/param/factor в trace."""

    def test_captures_sample_sites(self):
        """Все sample-сайты записываются в trace."""
        from hyperion_trace.handlers import TraceMessenger

        @model
        def simple():
            x = sample("x", Normal(0, 1))
            sample("obs", Normal(x, 1))

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as tm:
            simple()

        assert "x" in tm.trace
        assert "obs" in tm.trace
        assert len(tm.trace) == 2

    def test_observed_site_uses_obs_value(self):
        """Если передано obs — value == obs, observed=True."""
        from hyperion_trace.handlers import TraceMessenger

        @model
        def obs_model():
            x = sample("x", Normal(0, 1))
            sample("y", Normal(x, 1), obs=jnp.array(5.0))

        with TraceMessenger(rng_key=jrandom.PRNGKey(1)) as tm:
            obs_model()

        assert tm.trace["y"].observed is True
        assert float(tm.trace["y"].value) == pytest.approx(5.0)

    def test_log_prob_computed(self):
        """Каждый sample-сайт имеет log_prob != None."""
        from hyperion_trace.handlers import TraceMessenger

        @model
        def m():
            sample("z", Normal(0.0, 1.0))

        with TraceMessenger(rng_key=jrandom.PRNGKey(2)) as tm:
            m()

        assert tm.trace["z"].log_prob is not None
        assert jnp.isfinite(tm.trace["z"].log_prob)


class TestSubstituteMessenger:
    """SubstituteMessenger: подставляем значения вместо сэмплинга."""

    def test_substitutes_latent_value(self):
        """Подставленное значение записывается в trace."""
        from hyperion_trace.handlers import TraceMessenger, SubstituteMessenger

        @model
        def m():
            x = sample("x", Normal(0, 1))
            return x

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as tm:
            with SubstituteMessenger(data={"x": jnp.array(42.0)}):
                m()

        assert float(tm.trace["x"].value) == pytest.approx(42.0)

    def test_does_not_substitute_observed(self):
        """Если сайт уже observed — substitute НЕ перезаписывает."""
        from hyperion_trace.handlers import TraceMessenger, SubstituteMessenger

        @model
        def m():
            sample("y", Normal(0, 1), obs=jnp.array(10.0))

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as tm:
            with SubstituteMessenger(data={"y": jnp.array(99.0)}):
                m()

        assert float(tm.trace["y"].value) == pytest.approx(10.0)

    def test_substitute_only_named_sites(self):
        """Подставляются только совпадающие по имени сайты."""
        from hyperion_trace.handlers import TraceMessenger, SubstituteMessenger

        @model
        def m():
            a = sample("a", Normal(0, 1))
            b = sample("b", Normal(0, 1))
            return a, b

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as tm:
            with SubstituteMessenger(data={"a": jnp.array(7.0)}):
                m()

        assert float(tm.trace["a"].value) == pytest.approx(7.0)
        assert float(tm.trace["b"].value) != pytest.approx(7.0)


class TestReplayMessenger:
    """ReplayMessenger: воспроизводим значения из другого trace."""

    def test_replays_latent_values(self):
        """Значения из guide-trace воспроизводятся в model-trace."""
        from hyperion_trace.handlers import TraceMessenger, ReplayMessenger

        @model
        def m():
            sample("z", Normal(0, 1))

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as guide_tm:
            m()

        guide_value = float(guide_tm.trace["z"].value)

        with TraceMessenger(rng_key=jrandom.PRNGKey(99)) as model_tm:
            with ReplayMessenger(trace=guide_tm.trace):
                m()

        assert float(model_tm.trace["z"].value) == pytest.approx(guide_value)

    def test_does_not_replay_observed(self):
        """Observed сайты не реплеируются — у них своё значение."""
        from hyperion_trace.handlers import TraceMessenger, ReplayMessenger

        @model
        def m():
            sample("obs", Normal(0, 1), obs=jnp.array(5.0))

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as guide_tm:
            m()

        with TraceMessenger(rng_key=jrandom.PRNGKey(1)) as model_tm:
            with ReplayMessenger(trace=guide_tm.trace):
                m()

        assert float(model_tm.trace["obs"].value) == pytest.approx(5.0)


class TestBlockMessenger:
    """BlockMessenger: блокируем видимость определённых sample-сайтов."""

    def test_blocks_named_sites(self):
        """Заблокированные сайты не попадают в trace."""
        from hyperion_trace.handlers import TraceMessenger, BlockMessenger

        @model
        def m():
            sample("visible", Normal(0, 1))
            sample("hidden", Normal(0, 1))

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as tm:
            with BlockMessenger(hide=["hidden"]):
                m()

        assert "visible" in tm.trace
        assert "hidden" not in tm.trace

    def test_block_with_hide_fn(self):
        """Блокировка через функцию."""
        from hyperion_trace.handlers import TraceMessenger, BlockMessenger

        @model
        def m():
            sample("param_a", Normal(0, 1))
            sample("param_b", Normal(0, 1))
            sample("obs_c", Normal(0, 1))

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as tm:
            with BlockMessenger(hide_fn=lambda name: name.startswith("param")):
                m()

        assert "param_a" not in tm.trace
        assert "param_b" not in tm.trace
        assert "obs_c" in tm.trace

    def test_blocked_site_still_returns_value(self):
        """Заблокированный сайт семплит значение (для downstream) но не записывает."""
        from hyperion_trace.handlers import TraceMessenger, BlockMessenger

        result = {}

        @model
        def m():
            x = sample("x", Normal(0, 1))
            result["x"] = x

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as tm:
            with BlockMessenger(hide=["x"]):
                m()

        assert "x" not in tm.trace
        assert "x" in result
        assert jnp.isfinite(result["x"])

    def test_block_deterministic_rng(self):
        """BlockMessenger использует deterministic RNG — воспроизводимый результат."""
        from hyperion_trace.handlers import TraceMessenger, BlockMessenger

        results = []
        for _ in range(2):
            vals = {}

            @model
            def m():
                v = sample("x", Normal(0, 1))
                vals["v"] = v

            with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as tm:
                with BlockMessenger(hide=["x"]):
                    m()
            results.append(float(vals["v"]))

        assert results[0] == pytest.approx(results[1])


class TestHandlerComposition:
    """Композиция handler-ов: вложенные контексты."""

    def test_substitute_plus_trace(self):
        """SubstituteMessenger + TraceMessenger вместе."""
        from hyperion_trace.handlers import TraceMessenger, SubstituteMessenger

        @model
        def m():
            x = sample("x", Normal(0, 1))
            sample("y", Normal(x, 1))

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as tm:
            with SubstituteMessenger(data={"x": jnp.array(2.0)}):
                m()

        assert float(tm.trace["x"].value) == pytest.approx(2.0)
        assert "y" in tm.trace

    def test_block_plus_substitute(self):
        """Block одни сайты, substitute другие."""
        from hyperion_trace.handlers import TraceMessenger, SubstituteMessenger, BlockMessenger

        @model
        def m():
            a = sample("a", Normal(0, 1))
            b = sample("b", Normal(0, 1))
            sample("c", Normal(a + b, 1))

        with TraceMessenger(rng_key=jrandom.PRNGKey(0)) as tm:
            with BlockMessenger(hide=["a"]):
                with SubstituteMessenger(data={"b": jnp.array(5.0)}):
                    m()

        assert "a" not in tm.trace
        assert float(tm.trace["b"].value) == pytest.approx(5.0)
        assert "c" in tm.trace


# ═══════════════════════════════════════════════════════════
# Windowed Warmup
# ═══════════════════════════════════════════════════════════

class TestBuildAdaptationSchedule:
    """build_adaptation_schedule: Stan-style окна."""

    def test_short_warmup_single_window(self):
        """Для маленького warmup — одно окно на весь диапазон."""
        from hyperion_inference.warmup import build_adaptation_schedule
        schedule = build_adaptation_schedule(100)
        assert len(schedule) == 1
        assert schedule[0] == (0, 100)

    def test_standard_warmup_has_multiple_windows(self):
        """Для warmup >= 150 — несколько окон с удвоением."""
        from hyperion_inference.warmup import build_adaptation_schedule
        schedule = build_adaptation_schedule(500)
        assert len(schedule) > 1

        for start, end in schedule:
            assert start < end
            assert start >= 75

        assert schedule[-1][1] <= 450

    def test_windows_cover_adaptation_range(self):
        """Окна покрывают весь adaptation range без пропусков."""
        from hyperion_inference.warmup import build_adaptation_schedule
        schedule = build_adaptation_schedule(1000)

        for i in range(1, len(schedule)):
            assert schedule[i][0] == schedule[i - 1][1], (
                f"Gap: window {i-1} ends at {schedule[i-1][1]}, window {i} starts at {schedule[i][0]}"
            )

    def test_windows_respect_buffers(self):
        """Все окна — после initial_buffer (75) и до final_buffer (50)."""
        from hyperion_inference.warmup import build_adaptation_schedule
        schedule = build_adaptation_schedule(500)
        assert schedule[0][0] >= 75
        assert schedule[-1][1] <= 450

    def test_window_sizes_grow(self):
        """Размеры окон растут (doubling)."""
        from hyperion_inference.warmup import build_adaptation_schedule
        schedule = build_adaptation_schedule(1000)
        sizes = [end - start for start, end in schedule]
        for i in range(1, len(sizes) - 1):
            assert sizes[i] >= sizes[i - 1]


class TestFinalizeMass:
    """finalize_mass: регуляризация + сброс Welford."""

    def test_finalize_resets_welford(self):
        """После finalize — welford_count == 0, welford_mean == 0."""
        from hyperion_inference.warmup import make_warmup_state, welford_update, finalize_mass
        ws = make_warmup_state(0.1, 3)
        key = jrandom.PRNGKey(0)
        for i in range(20):
            ws = welford_update(ws, jrandom.normal(jrandom.PRNGKey(i), (3,)))

        assert int(ws.welford_count) == 20
        ws = finalize_mass(ws)
        assert int(ws.welford_count) == 0
        assert jnp.allclose(ws.welford_mean, 0.0)
        assert jnp.allclose(ws.welford_m2, 0.0)

    def test_finalize_updates_inv_mass(self):
        """finalize ставит inv_mass из Welford stats."""
        from hyperion_inference.warmup import make_warmup_state, welford_update, finalize_mass
        ws = make_warmup_state(0.1, 2)
        key = jrandom.PRNGKey(42)

        for i in range(100):
            k1, k2 = jrandom.split(jrandom.PRNGKey(i))
            s = jnp.array([jrandom.normal(k1) * 2.0, jrandom.normal(k2) * 0.5])
            ws = welford_update(ws, s)

        ws = finalize_mass(ws)
        assert ws.inv_mass_diag.shape == (2,)
        assert not jnp.allclose(ws.inv_mass_diag, jnp.ones(2))

    def test_finalize_updates_mass_chol(self):
        """finalize вычисляет mass_chol (Cholesky of mass)."""
        from hyperion_inference.warmup import make_warmup_state, welford_update, finalize_mass
        ws = make_warmup_state(0.1, 3, dense=True)

        for i in range(50):
            ws = welford_update(ws, jrandom.normal(jrandom.PRNGKey(i), (3,)))

        ws = finalize_mass(ws)
        L = ws.mass_chol
        assert L.shape == (3, 3)
        reconstructed = L @ L.T
        assert jnp.all(jnp.linalg.eigvalsh(reconstructed) > 0), "mass_chol @ mass_chol^T not PD"

    def test_finalize_inv_mass_dense_correct(self):
        """inv_mass_dense ≈ inv(mass_chol @ mass_chol^T)."""
        from hyperion_inference.warmup import make_warmup_state, welford_update, finalize_mass
        ws = make_warmup_state(0.1, 3, dense=True)

        for i in range(50):
            ws = welford_update(ws, jrandom.normal(jrandom.PRNGKey(i), (3,)))

        ws = finalize_mass(ws)
        mass = ws.mass_chol @ ws.mass_chol.T
        inv_mass_expected = jnp.linalg.inv(mass)
        assert jnp.allclose(ws.inv_mass_dense, inv_mass_expected, atol=1e-4), (
            f"inv_mass_dense vs inv(L@L^T) mismatch"
        )


class TestWindowedWarmupIntegration:
    """Windowed warmup интеграция в HMC."""

    def test_hmc_with_windowed_warmup_runs(self):
        """HMC с достаточным warmup использует windowed schedule."""
        from hyperion_inference.hmc import hmc_sample

        def potential(z):
            return -0.5 * jnp.sum(z ** 2)

        key = jrandom.PRNGKey(0)
        init = jnp.zeros(2)
        samples, lp, ap, info = hmc_sample(
            potential, key, init,
            num_samples=100, num_warmup=500,
            step_size=0.1, num_leapfrog=5,
        )

        assert samples.shape == (100, 2)
        assert "warmup_schedule" in info
        assert len(info["warmup_schedule"]) > 1

    def test_nuts_with_windowed_warmup_runs(self):
        """NUTS с достаточным warmup использует windowed schedule."""
        from hyperion_inference.nuts import nuts_sample

        def potential(z):
            return -0.5 * jnp.sum(z ** 2)

        key = jrandom.PRNGKey(0)
        init = jnp.zeros(2)
        samples, lp, ap, info = nuts_sample(
            potential, key, init,
            num_samples=100, num_warmup=500,
            step_size=0.1, max_tree_depth=5,
        )

        assert samples.shape == (100, 2)
        assert "warmup_schedule" in info
        assert len(info["warmup_schedule"]) > 1


# ═══════════════════════════════════════════════════════════
# Dense Mass Matrix
# ═══════════════════════════════════════════════════════════

class TestDenseMassHelpers:
    """Тесты для apply_inv_mass, kinetic_energy, sample_momentum."""

    def test_apply_inv_mass_diagonal(self):
        """Diagonal: element-wise multiply."""
        from hyperion_inference.warmup import apply_inv_mass
        inv_mass = jnp.array([2.0, 0.5, 1.0])
        r = jnp.array([1.0, 2.0, 3.0])
        result = apply_inv_mass(inv_mass, r)
        expected = jnp.array([2.0, 1.0, 3.0])
        assert jnp.allclose(result, expected)

    def test_apply_inv_mass_dense(self):
        """Dense: matrix-vector multiply."""
        from hyperion_inference.warmup import apply_inv_mass
        inv_mass = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        r = jnp.array([1.0, 2.0])
        result = apply_inv_mass(inv_mass, r)
        expected = inv_mass @ r
        assert jnp.allclose(result, expected)

    def test_kinetic_energy_diagonal(self):
        """Diagonal: 0.5 * sum(inv_mass * r^2)."""
        from hyperion_inference.warmup import kinetic_energy
        inv_mass = jnp.array([2.0, 0.5])
        r = jnp.array([1.0, 2.0])
        ke = kinetic_energy(inv_mass, r)
        expected = 0.5 * (2.0 * 1.0 + 0.5 * 4.0)
        assert float(ke) == pytest.approx(expected)

    def test_kinetic_energy_dense(self):
        """Dense: 0.5 * r^T @ inv_mass @ r."""
        from hyperion_inference.warmup import kinetic_energy
        inv_mass = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        r = jnp.array([1.0, 2.0])
        ke = kinetic_energy(inv_mass, r)
        expected = 0.5 * (r @ inv_mass @ r)
        assert float(ke) == pytest.approx(float(expected))

    def test_kinetic_energy_diagonal_dense_agree_when_diagonal(self):
        """Diagonal и dense дают одинаковый результат для диагональной матрицы."""
        from hyperion_inference.warmup import kinetic_energy
        diag_values = jnp.array([2.0, 0.5, 3.0])
        r = jnp.array([1.0, -1.0, 0.5])

        ke_diag = kinetic_energy(diag_values, r)
        ke_dense = kinetic_energy(jnp.diag(diag_values), r)
        assert float(ke_diag) == pytest.approx(float(ke_dense), abs=1e-6)

    def test_sample_momentum_diagonal_shape(self):
        """Diagonal momentum: правильная форма."""
        from hyperion_inference.warmup import sample_momentum
        key = jrandom.PRNGKey(0)
        inv_mass = jnp.ones(5)
        mass_chol = jnp.eye(5)
        r = sample_momentum(key, inv_mass, mass_chol)
        assert r.shape == (5,)
        assert jnp.all(jnp.isfinite(r))

    def test_sample_momentum_dense_shape(self):
        """Dense momentum: правильная форма."""
        from hyperion_inference.warmup import sample_momentum
        key = jrandom.PRNGKey(0)
        inv_mass = jnp.eye(3) * 2.0
        L = jnp.linalg.cholesky(jnp.linalg.inv(inv_mass))
        r = sample_momentum(key, inv_mass, L)
        assert r.shape == (3,)
        assert jnp.all(jnp.isfinite(r))

    def test_sample_momentum_dense_covariance(self):
        """Dense momentum семплит с правильной ковариацией (проверяем статистически)."""
        from hyperion_inference.warmup import sample_momentum
        inv_mass = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        mass = jnp.linalg.inv(inv_mass)
        L = jnp.linalg.cholesky(mass)

        keys = jrandom.split(jrandom.PRNGKey(0), 5000)
        samples = jax.vmap(lambda k: sample_momentum(k, inv_mass, L))(keys)

        emp_cov = jnp.cov(samples.T)
        assert jnp.allclose(emp_cov, mass, atol=0.15), (
            f"Empirical cov:\n{emp_cov}\nvs expected:\n{mass}"
        )


class TestDenseMassLeapfrog:
    """Dense mass в leapfrog — HMC и NUTS."""

    def test_hmc_dense_mass_runs(self):
        """HMC с dense_mass=True запускается без ошибок."""
        from hyperion_inference.hmc import hmc_sample

        def potential(z):
            return -0.5 * jnp.sum(z ** 2)

        key = jrandom.PRNGKey(0)
        init = jnp.zeros(3)
        samples, lp, ap, info = hmc_sample(
            potential, key, init,
            num_samples=50, num_warmup=100,
            step_size=0.1, num_leapfrog=5,
            dense_mass=True,
        )
        assert samples.shape == (50, 3)
        assert info["dense_mass"] is True

    def test_nuts_dense_mass_runs(self):
        """NUTS с dense_mass=True запускается без ошибок."""
        from hyperion_inference.nuts import nuts_sample

        def potential(z):
            return -0.5 * jnp.sum(z ** 2)

        key = jrandom.PRNGKey(0)
        init = jnp.zeros(3)
        samples, lp, ap, info = nuts_sample(
            potential, key, init,
            num_samples=50, num_warmup=100,
            step_size=0.1, max_tree_depth=5,
            dense_mass=True,
        )
        assert samples.shape == (50, 3)
        assert info["dense_mass"] is True

    def test_hmc_dense_mass_samples_standard_normal(self):
        """HMC с dense_mass семплит из стандартной нормали."""
        from hyperion_inference.hmc import hmc_sample

        def potential(z):
            return -0.5 * jnp.sum(z ** 2)

        key = jrandom.PRNGKey(42)
        init = jnp.zeros(2)
        samples, _, _, _ = hmc_sample(
            potential, key, init,
            num_samples=3000, num_warmup=1000,
            step_size=0.1, num_leapfrog=10,
            dense_mass=True,
        )
        assert abs(float(jnp.mean(samples[:, 0]))) < 0.2
        assert abs(float(jnp.var(samples[:, 0])) - 1.0) < 0.35


# ═══════════════════════════════════════════════════════════
# IR Pipeline
# ═══════════════════════════════════════════════════════════

class TestIRPipeline:
    """Тесты для IR: компиляция, структура графа, compute_log_joint."""

    def test_ir_from_trace_creates_nodes(self):
        """IRGraph.from_trace корректно создаёт ноды из trace."""
        from hyperion_trace.trace import trace_model
        from hyperion_ir.ir import IRGraph, IRNodeType

        @model
        def m():
            x = sample("x", Normal(0, 1))
            sample("y", Normal(x, 1), obs=jnp.array(2.0))

        trace = trace_model(m, rng_key=jrandom.PRNGKey(0))
        ir = IRGraph.from_trace(trace)

        assert "x" in ir.nodes
        assert "y" in ir.nodes
        assert ir.nodes["x"].node_type == IRNodeType.SAMPLE
        assert ir.nodes["y"].node_type == IRNodeType.OBSERVED

    def test_ir_latent_and_observed_lists(self):
        """latent_nodes и observed_nodes правильно фильтруют."""
        from hyperion_trace.trace import trace_model
        from hyperion_ir.ir import IRGraph

        @model
        def m():
            mu = sample("mu", Normal(0, 5))
            sigma = sample("sigma", HalfNormal(1.0))
            sample("obs", Normal(mu, sigma), obs=jnp.array(3.0))

        trace = trace_model(m, rng_key=jrandom.PRNGKey(0))
        ir = IRGraph.from_trace(trace)

        assert set(ir.latent_names) == {"mu", "sigma"}
        assert set(ir.observed_names) == {"obs"}

    def test_ir_compute_log_joint(self):
        """compute_log_joint возвращает конечный скаляр."""
        from hyperion_trace.trace import trace_model
        from hyperion_ir.ir import IRGraph

        @model
        def m():
            x = sample("x", Normal(0, 1))
            sample("y", Normal(x, 1), obs=jnp.array(2.0))

        trace = trace_model(m, rng_key=jrandom.PRNGKey(0))
        ir = IRGraph.from_trace(trace)

        lj = ir.compute_log_joint(
            latent_values={"x": jnp.array(1.5)},
            observed_values={"y": jnp.array(2.0)},
        )
        assert jnp.isfinite(lj)
        assert lj.ndim == 0

    def test_ir_compute_log_joint_correctness(self):
        """compute_log_joint == sum(log_prob) для модели без inter-latent dependencies.

        IR captures trace-time distribution params, поэтому тестируем
        модель с фиксированными параметрами (без зависимостей между латентами).
        """
        from hyperion_trace.trace import trace_model
        from hyperion_ir.ir import IRGraph

        @model
        def m():
            x = sample("x", Normal(0.0, 1.0))
            sample("y", Normal(3.0, 1.0), obs=jnp.array(2.0))

        trace = trace_model(m, rng_key=jrandom.PRNGKey(0))
        ir = IRGraph.from_trace(trace)

        x_val = jnp.array(1.0)
        y_val = jnp.array(2.0)

        ir_lj = ir.compute_log_joint(
            latent_values={"x": x_val},
            observed_values={"y": y_val},
        )

        prior_lp = Normal(0.0, 1.0).log_prob(x_val)
        lik_lp = Normal(3.0, 1.0).log_prob(y_val)
        expected = prior_lp + lik_lp

        assert float(ir_lj) == pytest.approx(float(expected), abs=1e-5)

    def test_ir_grad_log_joint_finite(self):
        """grad_log_joint возвращает конечные градиенты."""
        from hyperion_trace.trace import trace_model
        from hyperion_ir.ir import IRGraph

        @model
        def m():
            x = sample("x", Normal(0, 1))
            sample("y", Normal(x, 1), obs=jnp.array(2.0))

        trace = trace_model(m, rng_key=jrandom.PRNGKey(0))
        ir = IRGraph.from_trace(trace)

        grads = ir.grad_log_joint(
            latent_values={"x": jnp.array(1.0)},
            observed_values={"y": jnp.array(2.0)},
        )
        assert "x" in grads
        assert jnp.isfinite(grads["x"])

    def test_compiler_pipeline(self):
        """ModelCompiler.compile() производит валидный IRGraph."""
        from hyperion_ir.compiler import ModelCompiler

        @model
        def m():
            x = sample("x", Normal(0, 1))
            sample("y", Normal(x, 1), obs=jnp.array(2.0))

        compiler = ModelCompiler(optimize=True)
        ir = compiler.compile(m, rng_key=jrandom.PRNGKey(0))

        assert len(ir.nodes) >= 2
        assert "model_name" in ir.metadata
        assert len(ir.execution_order) == len(ir.nodes)

    def test_ir_topological_order_consistent(self):
        """Топологический порядок согласован с parent-child рёбрами."""
        from hyperion_ir.compiler import ModelCompiler

        @model
        def m():
            mu = sample("mu", Normal(0, 5))
            sample("obs", Normal(mu, 1), obs=jnp.array(1.0))

        compiler = ModelCompiler(optimize=True)
        ir = compiler.compile(m, rng_key=jrandom.PRNGKey(0))

        topo = ir.topological_order()
        positions = {name: i for i, name in enumerate(topo)}
        for name, node in ir.nodes.items():
            for parent in node.parents:
                if parent in positions and name in positions:
                    assert positions[parent] < positions[name], (
                        f"Parent {parent} after child {name} in topological order"
                    )

    def test_ir_dynamic_model_log_joint_uses_trace_time_params(self):
        """IR compute_log_joint captures trace-time distribution params.

        For a model y ~ Normal(x, 1) where x is latent, the observed node's
        log_prob is frozen at the trace-time value of x. Changing x in
        latent_values does NOT affect the observed node's log_prob in IR.
        This documents the known limitation.
        """
        from hyperion_trace.trace import trace_model
        from hyperion_ir.ir import IRGraph

        @model
        def dynamic_model():
            x = sample("x", Normal(0.0, 1.0))
            sample("y", Normal(x, 1.0), obs=jnp.array(2.0))

        trace = trace_model(dynamic_model, rng_key=jrandom.PRNGKey(42))
        ir = IRGraph.from_trace(trace)

        x_trace = float(trace["x"].value)

        lj_at_trace = ir.compute_log_joint(
            latent_values={"x": jnp.array(x_trace)},
            observed_values={"y": jnp.array(2.0)},
        )
        lj_at_different = ir.compute_log_joint(
            latent_values={"x": jnp.array(x_trace + 5.0)},
            observed_values={"y": jnp.array(2.0)},
        )

        # Observed node's log_prob stays the same (trace-time x is baked in)
        obs_lp_trace = Normal(x_trace, 1.0).log_prob(jnp.array(2.0))
        obs_lp_different = Normal(x_trace + 5.0, 1.0).log_prob(jnp.array(2.0))

        # The x latent's log_prob DOES change correctly
        x_lp_trace = Normal(0.0, 1.0).log_prob(jnp.array(x_trace))
        x_lp_different = Normal(0.0, 1.0).log_prob(jnp.array(x_trace + 5.0))

        expected_trace = float(x_lp_trace + obs_lp_trace)
        expected_different_correct = float(x_lp_different + obs_lp_different)
        expected_different_ir = float(x_lp_different + obs_lp_trace)

        assert float(lj_at_trace) == pytest.approx(expected_trace, abs=1e-5)
        # IR uses frozen obs log_prob, not the updated one
        assert float(lj_at_different) == pytest.approx(expected_different_ir, abs=1e-5)
        assert float(lj_at_different) != pytest.approx(expected_different_correct, abs=1e-3)

    def test_ir_independent_model_log_joint_correct(self):
        """IR compute_log_joint is fully correct for models without latent-dependent obs."""
        from hyperion_trace.trace import trace_model
        from hyperion_ir.ir import IRGraph

        @model
        def independent_model():
            mu = sample("mu", Normal(0.0, 10.0))
            sigma = sample("sigma", HalfNormal(2.0))
            sample("y1", Normal(5.0, 1.0), obs=jnp.array(4.5))
            sample("y2", Normal(5.0, 1.0), obs=jnp.array(5.5))

        trace = trace_model(independent_model, rng_key=jrandom.PRNGKey(0))
        ir = IRGraph.from_trace(trace)

        mu_val = jnp.array(1.0)
        sigma_val = jnp.array(1.5)

        ir_lj = ir.compute_log_joint(
            latent_values={"mu": mu_val, "sigma": sigma_val},
            observed_values={"y1": jnp.array(4.5), "y2": jnp.array(5.5)},
        )

        expected = (
            Normal(0.0, 10.0).log_prob(mu_val)
            + HalfNormal(2.0).log_prob(sigma_val)
            + Normal(5.0, 1.0).log_prob(jnp.array(4.5))
            + Normal(5.0, 1.0).log_prob(jnp.array(5.5))
        )

        assert jnp.isfinite(ir_lj)
        assert float(ir_lj) == pytest.approx(float(expected), abs=1e-5)

    def test_ir_grad_log_joint_dynamic_model_latent_grad_correct(self):
        """IR grad_log_joint is correct for the latent node even in dynamic models."""
        from hyperion_trace.trace import trace_model
        from hyperion_ir.ir import IRGraph

        @model
        def m():
            x = sample("x", Normal(0.0, 1.0))
            sample("y", Normal(x, 1.0), obs=jnp.array(2.0))

        trace = trace_model(m, rng_key=jrandom.PRNGKey(0))
        ir = IRGraph.from_trace(trace)

        x_val = jnp.array(1.0)
        grads = ir.grad_log_joint(
            latent_values={"x": x_val},
            observed_values={"y": jnp.array(2.0)},
        )

        # Gradient of Normal(0,1).log_prob(x) w.r.t. x = -x
        expected_grad = -x_val
        assert jnp.isfinite(grads["x"])
        assert float(grads["x"]) == pytest.approx(float(expected_grad), abs=1e-4)

    def test_ir_factor_node_contribution(self):
        """IR correctly includes factor nodes in log_joint."""
        from hyperion_trace.trace import trace_model
        from hyperion_ir.ir import IRGraph

        @model
        def factor_model():
            x = sample("x", Normal(0.0, 1.0))
            factor("penalty", -0.5 * x ** 2)

        trace = trace_model(factor_model, rng_key=jrandom.PRNGKey(0))
        ir = IRGraph.from_trace(trace)

        x_val = jnp.array(1.0)
        lj = ir.compute_log_joint(
            latent_values={"x": x_val},
            observed_values={},
        )

        expected_prior = Normal(0.0, 1.0).log_prob(x_val)
        factor_val = -0.5 * x_val ** 2
        x_trace = float(trace["x"].value)
        expected_factor = -0.5 * x_trace ** 2
        expected = expected_prior + expected_factor

        assert jnp.isfinite(lj)
        assert float(lj) == pytest.approx(float(expected), abs=1e-5)

    def test_ir_dynamic_compute_log_joint_dynamic(self):
        """compute_log_joint_dynamic re-traces model — correct for dynamic observed."""
        from hyperion_ir.ir import IRGraph
        from hyperion_ir.compiler import ModelCompiler

        @model
        def dynamic():
            x = sample("x", Normal(0.0, 1.0))
            sample("y", Normal(x, 1.0), obs=jnp.array(2.0))

        compiler = ModelCompiler(optimize=False)
        ir = compiler.compile(dynamic, rng_key=jrandom.PRNGKey(0),
                              substitutions={"y": jnp.array(2.0)})

        assert ir._has_dynamic_observed is True

        x_val = jnp.array(3.0)
        lj = ir.compute_log_joint_dynamic(
            latent_values={"x": x_val},
            observed_values={"y": jnp.array(2.0)},
        )
        expected = (
            Normal(0.0, 1.0).log_prob(x_val)
            + Normal(x_val, 1.0).log_prob(jnp.array(2.0))
        )
        assert float(lj) == pytest.approx(float(expected), abs=1e-4)

    def test_ir_static_model_not_marked_dynamic(self):
        """Static models (no latent-dependent obs) are not marked dynamic."""
        from hyperion_ir.ir import IRGraph
        from hyperion_ir.compiler import ModelCompiler

        @model
        def static():
            x = sample("x", Normal(0.0, 1.0))

        compiler = ModelCompiler(optimize=False)
        ir = compiler.compile(static, rng_key=jrandom.PRNGKey(0))
        assert ir._has_dynamic_observed is False


# ═══════════════════════════════════════════════════════════
# InferenceEngine Interface
# ═══════════════════════════════════════════════════════════

class TestInferenceEngineInterface:
    """Проверяем что InferenceEngine interface работает корректно."""

    def test_run_is_abstract(self):
        """run() — abstract, нельзя инстанцировать InferenceEngine."""
        from hyperion_inference.base import InferenceEngine
        with pytest.raises(TypeError, match="abstract"):
            InferenceEngine()

    def test_optional_methods_raise(self):
        """initialize/step/get_samples/get_metrics бросают NotImplementedError."""
        from hyperion_inference.hmc import HMCKernel
        kernel = HMCKernel()
        with pytest.raises(NotImplementedError):
            kernel.initialize(None, None, {})
        with pytest.raises(NotImplementedError):
            kernel.step(None)
        with pytest.raises(NotImplementedError):
            kernel.get_samples(None)
        with pytest.raises(NotImplementedError):
            kernel.get_metrics(None)

    def test_all_engines_have_run(self):
        """Все движки реализуют run()."""
        from hyperion_inference import (
            HMCKernel, NUTSKernel, SMCEngine, VIEngine, FlowsEngine, LaplaceApproximation,
        )
        for cls in [HMCKernel, NUTSKernel, SMCEngine, VIEngine, FlowsEngine, LaplaceApproximation]:
            instance = cls()
            assert hasattr(instance, "run")
            assert callable(instance.run)


# ═══════════════════════════════════════════════════════════
# Predictive class
# ═══════════════════════════════════════════════════════════

class TestPredictive:
    """Predictive: prior/posterior predictive generation."""

    def test_prior_predictive(self):
        from hyperion_inference.predictive import Predictive
        import numpy as np

        @model
        def simple():
            mu = sample("mu", Normal(0.0, 1.0))
            sample("x", Normal(mu, 0.5), obs=jnp.array(1.0))

        pred = Predictive(simple, num_samples=20)
        result = pred(jax.random.PRNGKey(0))
        assert "mu" in result
        assert result["mu"].shape[0] == 20

    def test_posterior_predictive(self):
        from hyperion_inference.predictive import Predictive
        import numpy as np

        @model
        def simple():
            mu = sample("mu", Normal(0.0, 1.0))
            sample("x", Normal(mu, 0.5), obs=jnp.array(1.0))

        posterior = {"mu": np.random.normal(1.0, 0.1, size=30)}
        pred = Predictive(simple, posterior_samples=posterior)
        result = pred(jax.random.PRNGKey(1))
        assert "mu" in result
        assert "x" in result
        assert result["mu"].shape[0] == 30

    def test_return_sites(self):
        from hyperion_inference.predictive import Predictive
        import numpy as np

        @model
        def multi():
            a = sample("a", Normal(0.0, 1.0))
            b = sample("b", Normal(a, 1.0))
            sample("y", Normal(b, 0.5), obs=jnp.array(2.0))

        pred = Predictive(multi, num_samples=10, return_sites=["a"])
        result = pred(jax.random.PRNGKey(0))
        assert "a" in result
        assert "b" not in result

    def test_raises_without_samples_or_num(self):
        from hyperion_inference.predictive import Predictive

        @model
        def m():
            sample("x", Normal(0.0, 1.0))

        with pytest.raises(ValueError):
            Predictive(m)


# ═══════════════════════════════════════════════════════════
# MCMC high-level API
# ═══════════════════════════════════════════════════════════

class TestMCMCHighLevel:
    """MCMC class: high-level runner with print_summary."""

    def test_mcmc_runs_hmc(self):
        from hyperion_inference.mcmc import MCMC
        from hyperion_inference.hmc import HMCKernel

        @model
        def normal_model():
            sample("mu", Normal(0.0, 1.0))

        kernel = HMCKernel()
        mcmc = MCMC(kernel, num_warmup=100, num_samples=200, num_chains=1)
        mcmc.run(jax.random.PRNGKey(0), normal_model)
        samples = mcmc.get_samples()
        assert "mu" in samples
        assert samples["mu"].shape[0] == 200

    def test_mcmc_runs_nuts(self):
        from hyperion_inference.mcmc import MCMC
        from hyperion_inference.nuts import NUTSKernel

        @model
        def normal_model():
            sample("mu", Normal(0.0, 1.0))

        kernel = NUTSKernel()
        mcmc = MCMC(kernel, num_warmup=100, num_samples=200, num_chains=1)
        mcmc.run(jax.random.PRNGKey(42), normal_model)
        samples = mcmc.get_samples()
        assert "mu" in samples
        assert samples["mu"].shape[0] == 200

    def test_mcmc_print_summary(self, capsys):
        from hyperion_inference.mcmc import MCMC
        from hyperion_inference.hmc import HMCKernel

        @model
        def normal_model():
            sample("mu", Normal(0.0, 1.0))

        kernel = HMCKernel()
        mcmc = MCMC(kernel, num_warmup=100, num_samples=200)
        mcmc.run(jax.random.PRNGKey(0), normal_model)
        mcmc.print_summary()
        captured = capsys.readouterr()
        assert "mu" in captured.out
        assert "mean" in captured.out

    def test_mcmc_raises_without_run(self):
        from hyperion_inference.mcmc import MCMC
        from hyperion_inference.hmc import HMCKernel

        mcmc = MCMC(HMCKernel(), num_warmup=10, num_samples=10)
        with pytest.raises(RuntimeError, match="run"):
            mcmc.get_samples()


# ═══════════════════════════════════════════════════════════
# Multi-chain structure in InferenceResult
# ═══════════════════════════════════════════════════════════

class TestMultiChainResult:
    """InferenceResult stores chain-level samples for R-hat."""

    def test_result_fields(self):
        from hyperion_inference.base import InferenceResult
        result = InferenceResult(
            samples={"x": jnp.ones(100)},
            num_chains=2,
            samples_by_chain={"x": jnp.ones((2, 50))},
        )
        assert result.num_chains == 2
        assert result.samples_by_chain is not None
        assert result.samples_by_chain["x"].shape == (2, 50)

    def test_single_chain_default(self):
        from hyperion_inference.base import InferenceResult
        result = InferenceResult(samples={"x": jnp.ones(100)})
        assert result.num_chains == 1
        assert result.samples_by_chain is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
