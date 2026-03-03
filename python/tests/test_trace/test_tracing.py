"""Тесты для трейсинга моделей.

trace_model должен собрать все sample-ноды в кучу и не забыть observed.
log_joint — сумма log_prob по всем нодам. Если NaN — модель сломана.
substitutions — подставляем значения вместо семплирования. Регрессионный тест на детерминизм.
"""

import jax
import jax.numpy as jnp
import pytest

from hyperion_dsl.primitives import sample, plate, deterministic, factor
from hyperion_dsl.distributions import Normal, HalfNormal
from hyperion_dsl.model import model
from hyperion_trace.trace import trace_model, NodeType


@model
def simple_model():
    mu = sample("mu", Normal(0.0, 10.0))
    sigma = sample("sigma", HalfNormal(1.0))
    x = sample("x", Normal(mu, sigma), obs=jnp.array(1.5))
    return x


@model
def hierarchical_model():
    mu_pop = sample("mu_pop", Normal(0.0, 10.0))
    sigma_pop = sample("sigma_pop", HalfNormal(5.0))
    with plate("groups", 3):
        mu_group = sample("mu_group", Normal(mu_pop, sigma_pop))
    return mu_group


class TestTracing:
    """Тест на дым: trace не взрывается при запуске. Это уже полдела."""

    def test_simple_trace(self):
        # trace_model должен собрать все sample-ноды
        key = jax.random.PRNGKey(0)
        trace = trace_model(simple_model, rng_key=key)

        assert len(trace) == 3
        assert "mu" in trace
        assert "sigma" in trace
        assert "x" in trace

    def test_latent_names(self):
        # latent = не observed. x observed — значит не latent
        key = jax.random.PRNGKey(0)
        trace = trace_model(simple_model, rng_key=key)

        assert "mu" in trace.latent_names
        assert "sigma" in trace.latent_names
        assert "x" not in trace.latent_names

    def test_observed_names(self):
        key = jax.random.PRNGKey(0)
        trace = trace_model(simple_model, rng_key=key)

        assert "x" in trace.observed_names

    def test_log_joint(self):
        # log_joint — сумма log_prob. Если NaN — распределение сломано, чини
        key = jax.random.PRNGKey(0)
        trace = trace_model(simple_model, rng_key=key)

        lj = trace.log_joint()
        assert jnp.isfinite(lj)

    def test_substitutions(self):
        # substitutions — подставляем значения. Регрессионный тест: те же subs = тот же trace
        key = jax.random.PRNGKey(0)
        subs = {"mu": jnp.array(2.0), "sigma": jnp.array(1.0)}
        trace = trace_model(simple_model, rng_key=key, substitutions=subs)

        assert jnp.allclose(trace["mu"].value, 2.0)
        assert jnp.allclose(trace["sigma"].value, 1.0)

    def test_node_types(self):
        # NodeType.SAMPLE и observed — должны быть выставлены правильно
        key = jax.random.PRNGKey(0)
        trace = trace_model(simple_model, rng_key=key)

        assert trace["mu"].node_type == NodeType.SAMPLE
        assert trace["x"].observed is True
