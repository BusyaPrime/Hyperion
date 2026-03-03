"""Posterior Predictive Checks (PPC) для HYPERION.

Берём сэмплы из постериора, прогоняем модель вперёд — получаем предсказания.
Сравниваем с реальными данными: MAE, RMSE, Bayesian p-value, coverage. Если модель ок — предсказания похожи.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


def posterior_predictive_check(
    model_fn: Callable,
    samples: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    rng_key: jax.random.PRNGKey,
    num_samples: Optional[int] = None,
    observed_name: Optional[str] = None,
    vectorized: bool = False,
) -> dict[str, np.ndarray]:
    """Генерируем posterior predictive samples.

    Для каждого сэмпла латентных — прогоняем модель, получаем предсказание observed.
    Без observed в substitutions — модель сэмплит его сама. Так получаем PPC.

    vectorized=True: использует vmap для batch-прогона (модель должна быть JIT-safe).
    """
    from hyperion_trace.trace import trace_model

    first_key = next(iter(samples))
    total = samples[first_key].shape[0]
    if num_samples is not None:
        total = min(total, num_samples)

    keys = jrandom.split(rng_key, total)

    if vectorized:
        return _ppc_vectorized(
            model_fn, samples, data, keys, total, observed_name,
        )

    ppc_samples: dict[str, list] = {}

    for i in range(total):
        substitutions = {name: vals[i] for name, vals in samples.items()}
        substitutions.update(data)

        trace = trace_model(
            model_fn,
            rng_key=keys[i],
            substitutions=substitutions,
        )

        for name in trace.observed_names:
            if observed_name is not None and name != observed_name:
                continue
            if name not in ppc_samples:
                ppc_samples[name] = []
            subs_no_obs = dict(substitutions)
            subs_no_obs.pop(name, None)
            pred_trace = trace_model(
                model_fn,
                rng_key=jrandom.fold_in(keys[i], sum(ord(c) for c in name)),
                substitutions=subs_no_obs,
            )
            if name in pred_trace.entries:
                ppc_samples[name].append(np.asarray(pred_trace.entries[name].value))

    return {name: np.stack(vals) for name, vals in ppc_samples.items() if vals}


def _ppc_vectorized(
    model_fn: Callable,
    samples: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    keys: jnp.ndarray,
    total: int,
    observed_name: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """Optimized PPC: single trace per sample (no double-trace).

    Probe trace определяет obs_names. Затем один trace_model per sample
    с latents substituted но obs removed — модель сэмплит obs сама.
    """
    from hyperion_trace.trace import trace_model

    probe_subs = {name: vals[0] for name, vals in samples.items()}
    probe_subs.update(data)
    probe_trace = trace_model(model_fn, rng_key=keys[0], substitutions=probe_subs)

    obs_names = probe_trace.observed_names
    if observed_name is not None:
        obs_names = [n for n in obs_names if n == observed_name]
    if not obs_names:
        return {}

    ppc_samples: dict[str, list] = {name: [] for name in obs_names}

    for i in range(total):
        subs = {name: vals[i] for name, vals in samples.items()}
        subs.update(data)
        for obs_name in obs_names:
            subs.pop(obs_name, None)

        pred_trace = trace_model(
            model_fn,
            rng_key=keys[i],
            substitutions=subs,
        )
        for obs_name in obs_names:
            if obs_name in pred_trace.entries:
                ppc_samples[obs_name].append(
                    np.asarray(pred_trace.entries[obs_name].value)
                )

    return {
        name: np.stack(vals) for name, vals in ppc_samples.items() if vals
    }


def ppc_summary(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Сводка PPC: MAE, RMSE, Bayesian p-value, coverage 90%.

    Bayesian p-value = P(predicted > observed). Около 0.5 — модель не пере/недооценивает.
    Coverage — доля observed внутри 90% PI. Должна быть ~0.9 если модель калибрована.
    """
    pred_mean = np.mean(predicted, axis=0)
    pred_std = np.std(predicted, axis=0)

    residuals = observed - pred_mean
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Bayesian p-value: какая доля predicted > observed
    p_value = float(np.mean(predicted > observed[None, ...]))

    # Coverage: какая доля observed попала в 90% PI
    lower = np.percentile(predicted, 5, axis=0)
    upper = np.percentile(predicted, 95, axis=0)
    coverage = float(np.mean((observed >= lower) & (observed <= upper)))

    return {
        "mae": mae,
        "rmse": rmse,
        "bayesian_p_value": p_value,
        "coverage_90": coverage,
        "pred_mean_mean": float(np.mean(pred_mean)),
        "pred_std_mean": float(np.mean(pred_std)),
    }
