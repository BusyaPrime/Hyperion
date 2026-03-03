"""Predictive — prior/posterior predictive utilities для HYPERION.

Predictive(model, posterior_samples) — удобная обёртка для генерации
предсказаний по posterior samples. Аналог numpyro.infer.Predictive.

Два режима:
  - posterior=None: prior predictive (для sanity check модели)
  - posterior=dict[str, array]: posterior predictive (для PPC)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from hyperion_trace.trace import trace_model


class Predictive:
    """Генерация predictive samples из модели.

    prior predictive:
        pred = Predictive(model, num_samples=100)
        samples = pred(rng_key)

    posterior predictive:
        pred = Predictive(model, posterior_samples=result.samples)
        samples = pred(rng_key)
    """

    def __init__(
        self,
        model_fn: Callable,
        posterior_samples: Optional[dict[str, np.ndarray]] = None,
        num_samples: Optional[int] = None,
        return_sites: Optional[list[str]] = None,
    ):
        """Инициализация predictive-обёртки.

        Либо posterior_samples (posterior predictive), либо num_samples (prior predictive).
        Оба сразу — можно, тогда num_samples ограничит число сэмплов в posterior режиме.

        Args:
            model_fn: модель с @model
            posterior_samples: dict сэмплов из MCMC — для posterior predictive
            num_samples: для prior — сколько раз сэмплить; для posterior — лимит
            return_sites: только эти сайты вернуть (None = все)
        """
        self.model_fn = model_fn
        self.posterior_samples = posterior_samples
        self.num_samples = num_samples
        self.return_sites = return_sites

        if posterior_samples is None and num_samples is None:
            raise ValueError(
                "Укажи posterior_samples или num_samples для prior predictive"
            )

    def __call__(
        self,
        rng_key: jax.random.PRNGKey,
        *args,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Генерирует predictive samples.

        Prior: прогоняет модель num_samples раз с новыми сэмплами из prior.
        Posterior: подставляет posterior_samples и генерирует предсказания.

        Args:
            rng_key: JAX PRNG ключ
            *args, **kwargs: аргументы для model_fn (обычно data)

        Returns:
            dict[str, np.ndarray] — имя сайта -> массив сэмплов
        """
        if self.posterior_samples is not None:
            return self._posterior_predictive(rng_key, *args, **kwargs)
        return self._prior_predictive(rng_key, *args, **kwargs)

    def _prior_predictive(
        self,
        rng_key: jax.random.PRNGKey,
        *args,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        n = self.num_samples
        keys = jrandom.split(rng_key, n)
        results: dict[str, list] = {}

        for i in range(n):
            trace = trace_model(self.model_fn, *args, rng_key=keys[i], **kwargs)
            for name, entry in trace.entries.items():
                if self.return_sites and name not in self.return_sites:
                    continue
                if entry.value is not None:
                    results.setdefault(name, []).append(np.asarray(entry.value))

        return {name: np.stack(vals) for name, vals in results.items() if vals}

    def _posterior_predictive(
        self,
        rng_key: jax.random.PRNGKey,
        *args,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        first_key = next(iter(self.posterior_samples))
        total = self.posterior_samples[first_key].shape[0]
        if self.num_samples is not None:
            total = min(total, self.num_samples)

        keys = jrandom.split(rng_key, total)
        results: dict[str, list] = {}

        for i in range(total):
            subs = {name: vals[i] for name, vals in self.posterior_samples.items()}
            trace = trace_model(
                self.model_fn, *args,
                rng_key=keys[i], substitutions=subs,
                **kwargs,
            )
            for name, entry in trace.entries.items():
                if self.return_sites and name not in self.return_sites:
                    continue
                if entry.value is not None:
                    results.setdefault(name, []).append(np.asarray(entry.value))

        return {name: np.stack(vals) for name, vals in results.items() if vals}
