#!/usr/bin/env python
"""Запускаем демо-модель HYPERION — проверяем, что всё установилось и работает."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import jax
import jax.numpy as jnp
import numpy as np

from hyperion_dsl.primitives import sample
from hyperion_dsl.distributions import Normal, HalfNormal
from hyperion_dsl.model import model
from hyperion_trace.trace import trace_model
from hyperion_ir.compiler import ModelCompiler


@model
def demo_linear_regression(x, y_obs):
    alpha = sample("alpha", Normal(0.0, 10.0))
    beta = sample("beta", Normal(0.0, 10.0))
    sigma = sample("sigma", HalfNormal(1.0))
    mu = alpha + beta * x
    y = sample("y", Normal(mu, sigma), obs=y_obs)
    return y


def main():
    print("=" * 60)
    print("HYPERION PPL — Demo Run")
    print("=" * 60)

    # Генерим синтетические данные
    np.random.seed(42)
    n = 50
    true_alpha, true_beta, true_sigma = 1.5, 0.7, 0.3
    x = np.linspace(-2, 2, n).astype(np.float32)
    y = true_alpha + true_beta * x + np.random.normal(0, true_sigma, n).astype(np.float32)

    x_jax = jnp.array(x)
    y_jax = jnp.array(y)

    print(f"\nTrue parameters: alpha={true_alpha}, beta={true_beta}, sigma={true_sigma}")
    print(f"Data: {n} observations")

    # Шаг 1: Трейсим модель
    print("\n--- Step 1: Model Tracing ---")
    key = jax.random.PRNGKey(0)
    trace = trace_model(demo_linear_regression, x_jax, y_jax, rng_key=key)

    print(f"Trace entries: {len(trace)}")
    print(f"Latent variables: {trace.latent_names}")
    print(f"Observed variables: {trace.observed_names}")
    print(f"Log-joint: {trace.log_joint():.4f}")

    # Шаг 2: Компилируем в IR
    print("\n--- Step 2: IR Compilation ---")
    compiler = ModelCompiler()
    ir = compiler.compile(demo_linear_regression, x_jax, y_jax, rng_key=key)

    print(f"IR nodes: {len(ir.nodes)}")
    print(f"Latent nodes: {ir.latent_names}")
    print(f"Execution order: {ir.execution_order}")

    # Шаг 3: Инфо о модели
    print("\n--- Step 3: Model Info ---")
    info = demo_linear_regression.info
    print(f"Model: {info.name}")
    print(f"Source hash: {info.source_hash}")
    print(f"Arguments: {info.arg_names}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
