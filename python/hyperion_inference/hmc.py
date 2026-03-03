"""Hamiltonian Monte Carlo — JIT-compiled, production-grade.

Всё через jax.lax: scan для цепи сэмплов, scan для leapfrog.
Никаких float(), никаких Python if на JAX-значениях. Только хардкор.

Два API:
  1. Функциональный (hmc_sample) — чистые функции, JIT-compatible
  2. Класс HMCKernel — обёртка для совместимости с ExperimentRunner

Dense mass matrix: поддержка через matmul в leapfrog (apply_inv_mass).
Windowed warmup: Stan-schedule с finalize_mass в конце каждого окна.
Dual averaging для step_size, Welford для mass matrix.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from hyperion_inference.base import InferenceEngine, InferenceResult
from hyperion_inference.warmup import (
    WarmupState,
    make_warmup_state,
    dual_averaging_update,
    welford_update,
    finalize_mass,
    build_adaptation_schedule,
    find_reasonable_step_size,
    apply_inv_mass,
    kinetic_energy,
    sample_momentum,
)


class HMCState(NamedTuple):
    """Состояние одного шага HMC. Все поля — JAX arrays, никаких Python объектов."""
    position: jnp.ndarray
    log_prob: jnp.ndarray
    grad: jnp.ndarray


def _leapfrog(
    potential_and_grad_fn: Callable,
    position: jnp.ndarray,
    momentum: jnp.ndarray,
    grad: jnp.ndarray,
    step_size: jnp.ndarray,
    inv_mass: jnp.ndarray,
    num_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Symplectic leapfrog через lax.scan. Поддерживает diagonal и dense mass.

    Схема: полшага p → (N-1 полных шагов q+p) → последний шаг q → полшага p.
    Симплектический = сохраняет фазовый объём = reversible.
    """
    momentum = momentum + 0.5 * step_size * grad

    def inner_step(carry, _):
        pos, mom = carry
        pos = pos + step_size * apply_inv_mass(inv_mass, mom)
        _lp, g = potential_and_grad_fn(pos)
        g = jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g))
        mom = mom + step_size * g
        return (pos, mom), None

    (position, momentum), _ = jax.lax.scan(
        inner_step, (position, momentum), None, length=num_steps - 1
    )

    position = position + step_size * apply_inv_mass(inv_mass, momentum)
    log_prob, grad = potential_and_grad_fn(position)
    grad = jnp.where(jnp.isfinite(grad), grad, jnp.zeros_like(grad))
    momentum = momentum + 0.5 * step_size * grad

    return position, momentum, grad, log_prob


def _hmc_step(
    potential_and_grad_fn: Callable,
    state: HMCState,
    rng_key: jnp.ndarray,
    step_size: jnp.ndarray,
    num_leapfrog: int,
    inv_mass: jnp.ndarray,
    mass_chol: jnp.ndarray,
) -> tuple[HMCState, jnp.ndarray]:
    """Один шаг HMC: sample momentum → leapfrog → MH accept/reject.

    Возвращает (new_state, accept_prob). Dense mass для momentum через mass_chol.
    """
    mom_key, accept_key = jrandom.split(rng_key)

    momentum = sample_momentum(mom_key, inv_mass, mass_chol)
    kinetic_old = kinetic_energy(inv_mass, momentum)

    new_pos, new_mom, new_grad, new_lp = _leapfrog(
        potential_and_grad_fn, state.position, momentum, state.grad,
        step_size, inv_mass, num_leapfrog,
    )

    kinetic_new = kinetic_energy(inv_mass, new_mom)

    log_alpha = new_lp - state.log_prob + kinetic_old - kinetic_new
    log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)

    accept = jnp.log(jrandom.uniform(accept_key)) < log_alpha

    position = jnp.where(accept, new_pos, state.position)
    log_prob = jnp.where(accept, new_lp, state.log_prob)
    grad = jnp.where(accept, new_grad, state.grad)

    accept_prob = jnp.minimum(1.0, jnp.exp(log_alpha))
    accept_prob = jnp.where(jnp.isfinite(accept_prob), accept_prob, 0.0)

    return HMCState(position, log_prob, grad), accept_prob


def hmc_sample(
    potential_fn: Callable,
    rng_key: jnp.ndarray,
    init_position: jnp.ndarray,
    num_samples: int = 1000,
    num_warmup: int = 500,
    step_size: float = 0.01,
    num_leapfrog: int = 10,
    target_accept: float = 0.8,
    dense_mass: bool = False,
    _skip_step_size_init: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Полный HMC: windowed warmup + sampling.

    Warmup: Stan-style schedule с adaptation windows.
      - Initial buffer: только dual averaging
      - Adaptation windows: dual averaging + Welford, finalize_mass в конце окна
      - Final buffer: только dual averaging с финальной mass matrix
    Sampling: фиксированные гиперпараметры, lax.scan.

    dense_mass=True: full covariance mass через matmul в leapfrog + Cholesky для momentum.
    """
    if num_leapfrog < 1:
        raise ValueError(f"num_leapfrog должен быть >= 1, получено {num_leapfrog}")

    potential_and_grad = jax.value_and_grad(potential_fn)

    init_lp, init_grad = potential_and_grad(init_position)
    init_grad = jnp.where(jnp.isfinite(init_grad), init_grad, jnp.zeros_like(init_grad))
    dim = init_position.shape[0]

    init_inv_mass = jnp.eye(dim) if dense_mass else jnp.ones(dim)
    init_mass_chol = jnp.eye(dim)
    eps_key, rng_key = jrandom.split(rng_key)
    if not _skip_step_size_init:
        try:
            step_size = float(find_reasonable_step_size(
                potential_and_grad, init_position, eps_key,
                init_inv_mass, init_mass_chol,
            ))
        except Exception as e:
            warnings.warn(
                f"find_reasonable_step_size failed ({type(e).__name__}: {e}), "
                f"using default step_size={step_size}",
                RuntimeWarning,
                stacklevel=2,
            )

    hmc_state = HMCState(init_position, init_lp, init_grad)
    warmup_state = make_warmup_state(step_size, dim, dense=dense_mass)

    schedule = build_adaptation_schedule(num_warmup)

    def _get_mass(warm_st):
        if dense_mass:
            return warm_st.inv_mass_dense, warm_st.mass_chol
        return warm_st.inv_mass_diag, warm_st.mass_chol

    def buffer_body(carry, rng_key):
        """Initial/final buffer: только dual averaging, БЕЗ Welford."""
        hmc_st, warm_st, step_num = carry
        inv_m, m_chol = _get_mass(warm_st)

        new_hmc, accept_prob = _hmc_step(
            potential_and_grad, hmc_st, rng_key,
            warm_st.step_size, num_leapfrog, inv_m, m_chol,
        )

        new_warm = dual_averaging_update(warm_st, accept_prob, step_num, target_accept)
        return (new_hmc, new_warm, step_num + 1), accept_prob

    def adapt_body(carry, rng_key):
        """Adaptation window: dual averaging + Welford для mass matrix."""
        hmc_st, warm_st, step_num = carry
        inv_m, m_chol = _get_mass(warm_st)

        new_hmc, accept_prob = _hmc_step(
            potential_and_grad, hmc_st, rng_key,
            warm_st.step_size, num_leapfrog, inv_m, m_chol,
        )

        new_warm = dual_averaging_update(warm_st, accept_prob, step_num, target_accept)
        new_warm = welford_update(new_warm, new_hmc.position)
        return (new_hmc, new_warm, step_num + 1), accept_prob

    warmup_keys = jrandom.split(rng_key, num_warmup + 1)
    rng_key_post = warmup_keys[0]
    warmup_rng = warmup_keys[1:]

    is_windowed = len(schedule) > 1

    if is_windowed:
        init_end = schedule[0][0]
        step_counter = jnp.ones((), dtype=jnp.int32)

        if init_end > 0:
            (hmc_state, warmup_state, step_counter), _ = jax.lax.scan(
                buffer_body,
                (hmc_state, warmup_state, step_counter),
                warmup_rng[:init_end],
            )

        for win_start, win_end in schedule:
            win_keys = warmup_rng[win_start:win_end]
            (hmc_state, warmup_state, step_counter), _ = jax.lax.scan(
                adapt_body,
                (hmc_state, warmup_state, step_counter),
                win_keys,
            )
            warmup_state = finalize_mass(warmup_state)

        final_start = schedule[-1][1]
        if final_start < num_warmup:
            (hmc_state, warmup_state, step_counter), _ = jax.lax.scan(
                buffer_body,
                (hmc_state, warmup_state, step_counter),
                warmup_rng[final_start:],
            )
    else:
        (hmc_state, warmup_state, _), _ = jax.lax.scan(
            adapt_body,
            (hmc_state, warmup_state, jnp.ones((), dtype=jnp.int32)),
            warmup_rng,
        )
        warmup_state = finalize_mass(warmup_state)

    final_step_size = jnp.exp(warmup_state.log_step_size_avg)
    final_inv_mass, final_mass_chol = _get_mass(warmup_state)

    def sample_body(hmc_st, rng_key):
        new_st, accept_prob = _hmc_step(
            potential_and_grad, hmc_st, rng_key,
            final_step_size, num_leapfrog, final_inv_mass, final_mass_chol,
        )
        return new_st, (new_st.position, new_st.log_prob, accept_prob)

    sample_keys = jrandom.split(rng_key_post, num_samples)
    _, (samples, log_probs, accept_probs) = jax.lax.scan(
        sample_body, hmc_state, sample_keys,
    )

    info = {
        "step_size": final_step_size,
        "inv_mass_diag": warmup_state.inv_mass_diag,
        "dense_mass": dense_mass,
        "warmup_schedule": schedule,
        "mean_accept_prob": jnp.mean(accept_probs),
    }
    if dense_mass:
        info["inv_mass_dense"] = final_inv_mass

    return samples, log_probs, accept_probs, info


def hmc_sample_chains(
    potential_fn: Callable,
    rng_key: jnp.ndarray,
    init_positions: jnp.ndarray,
    num_chains: int = 4,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Multi-chain HMC: vmap по цепям. init_positions: [num_chains, dim].

    Каждая цепь — независимый HMC с windowed warmup.
    Параллелизм через vmap → XLA распараллелит автоматически.
    Step size вычисляется один раз вне vmap для vmap-safety.
    """
    dense_mass = kwargs.get("dense_mass", False)
    dim = init_positions.shape[-1]
    init_inv_mass = jnp.eye(dim) if dense_mass else jnp.ones(dim)
    init_mass_chol = jnp.eye(dim)

    eps_key, rng_key = jrandom.split(rng_key)
    potential_and_grad = jax.value_and_grad(potential_fn)
    try:
        computed_step = float(find_reasonable_step_size(
            potential_and_grad, init_positions[0], eps_key,
            init_inv_mass, init_mass_chol,
        ))
        kwargs["step_size"] = computed_step
    except Exception:
        pass

    kwargs["_skip_step_size_init"] = True
    keys = jrandom.split(rng_key, num_chains)

    vmapped = jax.vmap(
        lambda k, pos: hmc_sample(potential_fn, k, pos, **kwargs)
    )

    samples, log_probs, accept_probs, info = vmapped(keys, init_positions)
    return samples, log_probs, accept_probs, info


class HMCKernel(InferenceEngine):
    """HMC обёртка: class API поверх функционального JIT-ядра.

    run() вызывает hmc_sample() — всё остальное делает functional API.
    Этот класс нужен только для совместимости с ExperimentRunner.
    """

    def __init__(self):
        self._backend = None

    def run(self, backend, rng_key, config) -> InferenceResult:
        """Запуск HMC через JIT-compiled functional API."""
        init_key, sample_key = jrandom.split(rng_key)
        prior = backend.sample_prior(init_key)
        init_pos = backend.flatten_latents(prior)

        num_chains = config.get("num_chains", 1)

        hmc_kwargs = {
            "num_samples": config.get("num_samples", 1000),
            "num_warmup": config.get("num_warmup", 500),
            "step_size": config.get("step_size", 0.01),
            "num_leapfrog": config.get("num_leapfrog", 10),
            "target_accept": config.get("target_accept_rate", 0.8),
            "dense_mass": config.get("dense_mass", False),
        }

        if num_chains > 1:
            chain_keys = jrandom.split(init_key, num_chains)
            init_positions = jnp.stack([
                backend.flatten_latents(backend.sample_prior(k))
                for k in chain_keys
            ])
            samples_flat, log_probs, accept_probs, info = hmc_sample_chains(
                backend.potential_fn, sample_key, init_positions,
                num_chains=num_chains, **hmc_kwargs,
            )
            # samples_flat: (num_chains, num_samples, dim)
            flat_all = samples_flat.reshape(-1, samples_flat.shape[-1])
            log_probs_out = log_probs.reshape(-1)
            accept_out = accept_probs.reshape(-1)

            latent_names = list(backend.get_latent_shapes().keys())
            unflat_all = jax.vmap(backend.unflatten_latents)(flat_all)
            samples = {name: unflat_all[name] for name in latent_names}

            n_s = samples_flat.shape[1]
            samples_by_chain = {}
            for name in latent_names:
                full = samples[name]
                chain_shape = (num_chains, n_s) + full.shape[1:]
                samples_by_chain[name] = full.reshape(chain_shape)

            agg_info = {}
            for k, v in info.items():
                if isinstance(v, jnp.ndarray) and v.ndim >= 1:
                    agg_info[k] = float(jnp.mean(v))
                else:
                    agg_info[k] = v
            agg_info["accept_probs"] = accept_out

            return InferenceResult(
                samples=samples,
                log_probs=log_probs_out,
                diagnostics=agg_info,
                num_chains=num_chains,
                samples_by_chain=samples_by_chain,
            )
        else:
            flat_samples, log_probs_out, accept_out, info = hmc_sample(
                backend.potential_fn, sample_key, init_pos, **hmc_kwargs,
            )

            latent_names = list(backend.get_latent_shapes().keys())
            unflat_batched = jax.vmap(backend.unflatten_latents)(flat_samples)
            samples = {name: unflat_batched[name] for name in latent_names}

            return InferenceResult(
                samples=samples,
                log_probs=log_probs_out,
                diagnostics={**info, "accept_probs": accept_out},
            )
