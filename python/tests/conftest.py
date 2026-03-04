"""Shared fixtures для всего test suite.

Всё что нужно в каждом втором тесте — ключи, модели, данные —
лежит тут. Не дублируем, не пишем boilerplate в каждом файле.
"""

import pytest
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from hyperion_dsl.model import model
from hyperion_dsl.primitives import sample
from hyperion_dsl.distributions import Normal, HalfNormal


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: slow integration tests")
    config.addinivalue_line("markers", "benchmark: performance benchmarks")


@pytest.fixture
def rng_key():
    """Детерминированный JAX PRNG ключ для воспроизводимости."""
    return jrandom.PRNGKey(42)


@pytest.fixture
def rng_keys():
    """Четыре ключа для multi-chain тестов."""
    return jrandom.split(jrandom.PRNGKey(0), 4)


@pytest.fixture
def simple_model():
    """Простая Normal-Normal модель — рабочая лошадка для smoke тестов."""
    @model
    def _model():
        mu = sample("mu", Normal(0.0, 10.0))
        sample("obs", Normal(mu, 1.0))
    return _model


@pytest.fixture
def hierarchical_model():
    """Иерархическая модель — mu/sigma → obs."""
    @model
    def _model():
        mu = sample("mu", Normal(0.0, 5.0))
        sigma = sample("sigma", HalfNormal(2.0))
        sample("obs", Normal(mu, sigma))
    return _model


@pytest.fixture
def observed_data():
    """Синтетические наблюдения из N(3.0, 1.0), n=100."""
    key = jrandom.PRNGKey(123)
    return {"obs": jrandom.normal(key, shape=(100,)) + 3.0}


@pytest.fixture
def small_observed_data():
    """Маленький датасет для быстрых тестов, n=20."""
    key = jrandom.PRNGKey(456)
    return {"obs": jrandom.normal(key, shape=(20,)) + 2.0}


@pytest.fixture
def mcmc_chain():
    """Синтетическая MCMC цепь для тестов диагностики."""
    np.random.seed(42)
    n = 1000
    chain = np.cumsum(np.random.randn(n) * 0.1) + 5.0
    return chain.astype(np.float32)


@pytest.fixture
def mcmc_chains():
    """Четыре синтетических цепи (4, 500) для multi-chain диагностики."""
    np.random.seed(42)
    chains = []
    for _ in range(4):
        chain = np.cumsum(np.random.randn(500) * 0.1) + 5.0
        chains.append(chain)
    return np.array(chains, dtype=np.float32)
