"""HYPERION Demo Models — обязательный набор из спецификации.

1. Bayesian Linear Regression — closed-form sanity check
2. Hierarchical Normal Model — частый источник боли
3. Logistic Regression с Horseshoe prior — спарсити
4. State-Space Model — калмановщина
5. Gaussian Mixture Model — мультимодальность
6. Stochastic Volatility — таймсерии для финансистов
"""

import jax
import jax.numpy as jnp

from hyperion_dsl.primitives import sample, plate, deterministic, factor
from hyperion_dsl.distributions import (
    Normal, HalfNormal, HalfCauchy, Cauchy, Beta,
    Gamma, MultivariateNormal, Bernoulli, InverseGamma,
    Dirichlet,
)
from hyperion_dsl.model import model


# --- Model 1: Bayesian Linear Regression ---
# Байесовская линейная регрессия — Hello World мира PPL. Если это не работает, дальше можно не идти

@model
def bayesian_linear_regression(x, y_obs):
    """Байесовская линейная регрессия с нормальными приорами. Классика жанра."""
    alpha = sample("alpha", Normal(0.0, 10.0))
    beta = sample("beta", Normal(0.0, 10.0))
    sigma = sample("sigma", HalfNormal(1.0))
    mu = alpha + beta * x
    y = sample("y", Normal(mu, sigma), obs=y_obs)
    return y


# --- Model 2: Hierarchical Normal Model ---
# Иерархическая нормаль — когда группы связаны, а данные разъехались

@model
def hierarchical_normal(y_obs, group_ids, num_groups):
    """Иерархическая нормальная модель с групповыми эффектами. Частый pain point в реальных данных."""
    mu_pop = sample("mu_pop", Normal(0.0, 10.0))
    sigma_pop = sample("sigma_pop", HalfNormal(5.0))
    sigma_obs = sample("sigma_obs", HalfNormal(1.0))

    with plate("groups", num_groups):
        mu_group = sample("mu_group", Normal(mu_pop, sigma_pop))

    # Модель наблюдений — маппим группу на каждую точку
    mu_y = mu_group[group_ids]
    y = sample("y", Normal(mu_y, sigma_obs), obs=y_obs)
    return y


# --- Model 3: Logistic Regression with Horseshoe Prior ---
# Horseshoe prior — когда тебе надо автоматически обнулять ненужные фичи

@model
def logistic_regression_horseshoe(x, y_obs, num_features):
    """Логистическая регрессия с horseshoe prior для спарсити. Автоматический feature selection."""
    tau = sample("tau", HalfCauchy(1.0))

    with plate("features", num_features):
        lam = sample("lambda", HalfCauchy(1.0))
        beta = sample("beta", Normal(0.0, tau * lam))

    alpha = sample("alpha", Normal(0.0, 5.0))
    logits = alpha + x @ beta
    y = sample("y", Bernoulli(logits=logits), obs=y_obs)
    return y


# --- Model 4: State-Space Model (Local Level) ---
# State-space — random walk + шум, калмановская классика

@model
def state_space_model(y_obs, T):
    """Local level (random walk + noise) state-space модель. Калман одобряет."""
    sigma_state = sample("sigma_state", HalfNormal(0.5))
    sigma_obs = sample("sigma_obs", HalfNormal(1.0))

    x_init = sample("x_0", Normal(0.0, 10.0))
    x_prev = x_init

    xs = [x_init]
    for t in range(1, T):
        x_t = sample(f"x_{t}", Normal(x_prev, sigma_state))
        xs.append(x_t)
        x_prev = x_t

    x_stack = jnp.stack(xs)
    y = sample("y", Normal(x_stack, sigma_obs), obs=y_obs)
    return y


# --- Model 5: Gaussian Mixture Model ---
# Гауссовская смесь — каноническая задачка на мультимодальный постериор

@model
def gaussian_mixture(y_obs, K):
    """Гауссовская смесь с K компонентами.

    Note: Используем continuous relaxation для весов компонент.
    Настоящий дискретный инференс потребовал бы enumeration — это отдельная история.
    """
    # Веса смеси (симметричный Dirichlet)
    weights = sample("weights", Dirichlet(jnp.ones(K)))

    with plate("components", K):
        mu_k = sample("mu_k", Normal(0.0, 10.0))
        sigma_k = sample("sigma_k", HalfNormal(2.0))

    # Считаем log-likelihood как смесь
    N = y_obs.shape[0]
    log_likes = jnp.stack([
        Normal(mu_k[k], sigma_k[k]).log_prob(y_obs)
        for k in range(K)
    ], axis=0)  # (K, N)

    log_weights = jnp.log(weights)
    log_lik = jnp.sum(
        jax.scipy.special.logsumexp(log_likes + log_weights[:, None], axis=0)
    )
    factor("mixture_log_lik", log_lik)


# --- Model 6: Stochastic Volatility ---
# Стохастическая волатильность — для тех, кто любит AR(1) в логарифмах

@model
def stochastic_volatility(y_obs, T):
    """Стохастическая волатильность для финансовых таймсерий.

    log(sigma_t) следует AR(1) процессу. Финансисты плачут от счастья.
    """
    mu = sample("mu", Normal(0.0, 10.0))
    phi = sample("phi", Beta(20.0, 1.5))  # персистентность, близко к 1
    sigma_vol = sample("sigma_vol", HalfNormal(0.5))

    h_init = sample("h_0", Normal(mu, sigma_vol / jnp.sqrt(1.0 - phi ** 2)))
    h_prev = h_init

    hs = [h_init]
    for t in range(1, T):
        h_t = sample(f"h_{t}", Normal(mu + phi * (h_prev - mu), sigma_vol))
        hs.append(h_t)
        h_prev = h_t

    h_stack = jnp.stack(hs)
    y = sample("y", Normal(0.0, jnp.exp(h_stack / 2.0)), obs=y_obs)
    return y
