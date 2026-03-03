"""Integration test: собираем всю пирамиду от DSL до IR и смотрим что не рухнуло.

Тест на дым: модель не взрывается при trace и compile.
Регрессионный тест: детерминированная модель с фиксированным ключом даёт тот же output.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyperion_dsl.primitives import sample
from hyperion_dsl.distributions import Normal, HalfNormal
from hyperion_dsl.model import model
from hyperion_trace.trace import trace_model
from hyperion_ir.compiler import ModelCompiler


@model
def bayesian_linear_regression(x_obs, y_obs):
    alpha = sample("alpha", Normal(0.0, 10.0))
    beta = sample("beta", Normal(0.0, 10.0))
    sigma = sample("sigma", HalfNormal(1.0))
    mu = alpha + beta * x_obs
    y = sample("y", Normal(mu, sigma), obs=y_obs)
    return y


class TestEndToEnd:
    """End-to-end: trace -> compile -> diagnostics. Вся цепочка должна работать."""

    @pytest.fixture
    def regression_data(self):
        np.random.seed(42)
        n = 50
        x = np.linspace(-2, 2, n).astype(np.float32)
        y = 1.5 + 0.7 * x + np.random.normal(0, 0.3, n).astype(np.float32)
        return {"x_obs": jnp.array(x), "y_obs": jnp.array(y)}

    def test_trace_linear_regression(self, regression_data):
        # trace_model должен собрать latent и observed, log_joint конечен
        key = jax.random.PRNGKey(0)
        trace = trace_model(
            bayesian_linear_regression,
            regression_data["x_obs"],
            regression_data["y_obs"],
            rng_key=key,
        )
        assert "alpha" in trace.latent_names
        assert "beta" in trace.latent_names
        assert "sigma" in trace.latent_names
        assert "y" in trace.observed_names
        assert jnp.isfinite(trace.log_joint())

    def test_compile_linear_regression(self, regression_data):
        # Compiler собирает IR. nodes и latent_names должны быть
        compiler = ModelCompiler()
        ir = compiler.compile(
            bayesian_linear_regression,
            regression_data["x_obs"],
            regression_data["y_obs"],
            rng_key=jax.random.PRNGKey(0),
        )
        assert len(ir.nodes) >= 4
        assert len(ir.latent_names) >= 3

    def test_model_info(self):
        # model.info — метаданные модели. name и arg_names
        info = bayesian_linear_regression.info
        assert info.name == "bayesian_linear_regression"
        assert "x_obs" in info.arg_names
        assert "y_obs" in info.arg_names

    def test_deterministic_trace_same_key(self, regression_data):
        # Регрессионный тест: тот же ключ = тот же trace (детерминизм)
        key = jax.random.PRNGKey(42)
        trace1 = trace_model(
            bayesian_linear_regression,
            regression_data["x_obs"],
            regression_data["y_obs"],
            rng_key=key,
        )
        trace2 = trace_model(
            bayesian_linear_regression,
            regression_data["x_obs"],
            regression_data["y_obs"],
            rng_key=key,
        )
        assert jnp.allclose(trace1["alpha"].value, trace2["alpha"].value)
        assert jnp.allclose(trace1["beta"].value, trace2["beta"].value)
        assert jnp.allclose(trace1["sigma"].value, trace2["sigma"].value)
