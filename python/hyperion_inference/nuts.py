"""No-U-Turn Sampler — bidirectional tree doubling, JIT-compiled.

Полная реализация NUTS из Hoffman & Gelman 2014 с multinomial weighting
из Betancourt 2017. Bidirectional tree doubling с корректным U-turn
criterion на обоих концах траектории.

Dense mass matrix: поддержка через apply_inv_mass в leapfrog.
Windowed warmup: Stan-schedule с finalize_mass в конце каждого окна.

JIT-совместимость:
  - while_loop для tree doubling (внешний цикл)
  - fori_loop для leapfrog интеграции внутри поддерева
  - Все состояния — NamedTuple с фиксированными dtype
"""

from __future__ import annotations

import logging
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from hyperion_inference.base import InferenceEngine, InferenceResult

logger = logging.getLogger(__name__)
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


class NUTSState(NamedTuple):
    """Состояние NUTS сэмплера. Все поля — JAX arrays."""
    position: jnp.ndarray
    log_prob: jnp.ndarray
    grad: jnp.ndarray


class _Leaf(NamedTuple):
    """Одна точка на фазовой траектории (position + momentum)."""
    z: jnp.ndarray
    r: jnp.ndarray
    log_prob: jnp.ndarray
    grad: jnp.ndarray


class _SubtreeInfo(NamedTuple):
    """Результат интеграции поддерева через fori_loop."""
    endpoint: _Leaf
    first_z: jnp.ndarray
    first_r: jnp.ndarray
    proposal_z: jnp.ndarray
    proposal_lp: jnp.ndarray
    proposal_grad: jnp.ndarray
    log_sum_weight: jnp.ndarray
    turning: jnp.ndarray
    diverging: jnp.ndarray
    sum_accept_prob: jnp.ndarray
    num_proposals: jnp.ndarray
    rng_key: jnp.ndarray


class _TreeState(NamedTuple):
    """Состояние полного дерева NUTS для while_loop."""
    left_z: jnp.ndarray
    left_r: jnp.ndarray
    left_lp: jnp.ndarray
    left_grad: jnp.ndarray
    right_z: jnp.ndarray
    right_r: jnp.ndarray
    right_lp: jnp.ndarray
    right_grad: jnp.ndarray
    proposal_z: jnp.ndarray
    proposal_lp: jnp.ndarray
    proposal_grad: jnp.ndarray
    log_sum_weight: jnp.ndarray
    depth: jnp.ndarray
    turning: jnp.ndarray
    diverging: jnp.ndarray
    sum_accept_prob: jnp.ndarray
    num_proposals: jnp.ndarray
    energy_init: jnp.ndarray
    rng_key: jnp.ndarray


def _leapfrog_one(potential_and_grad_fn, leaf, step_size, inv_mass):
    """Один leapfrog шаг: полшага r → шаг z → полшага r.
    Поддерживает diagonal (1D) и dense (2D) mass через apply_inv_mass.
    """
    new_r = leaf.r + 0.5 * step_size * leaf.grad
    new_z = leaf.z + step_size * apply_inv_mass(inv_mass, new_r)
    new_lp, new_grad = potential_and_grad_fn(new_z)
    new_grad = jnp.where(jnp.isfinite(new_grad), new_grad, jnp.zeros_like(new_grad))
    new_r = new_r + 0.5 * step_size * new_grad
    return _Leaf(new_z, new_r, new_lp, new_grad)


def _compute_energy(leaf, inv_mass):
    """Hamiltonian = -log_prob + 0.5 * r^T M^{-1} r. Dense-aware."""
    return -leaf.log_prob + kinetic_energy(inv_mass, leaf.r)


def _integrate_subtree(
    potential_and_grad_fn, start_leaf, direction, depth,
    step_size, inv_mass, energy_init, max_energy_delta, rng_key,
):
    """Построить поддерево глубины depth: 2^depth leapfrog шагов в direction.

    Используем fori_loop: на каждом шаге один leapfrog,
    multinomial proposal update, divergence check.
    U-turn check на полном поддереве делается после цикла.
    """
    num_steps = jnp.power(2, depth).astype(jnp.int32)
    directed_step = direction * step_size

    init_carry = (
        start_leaf,
        start_leaf.z,
        start_leaf.r,
        start_leaf.z,
        start_leaf.log_prob,
        start_leaf.grad,
        -jnp.inf,
        jnp.array(False),
        jnp.zeros(()),
        jnp.zeros((), dtype=jnp.int32),
        rng_key,
        jnp.array(True),
    )

    def step_fn(i, carry):
        (leaf, first_z, first_r, prop_z, prop_lp, prop_grad,
         lsw, diverging, sap, n_prop, rng, is_first) = carry

        rng, step_key = jrandom.split(rng)
        new_leaf = _leapfrog_one(potential_and_grad_fn, leaf, directed_step, inv_mass)

        delta_H = _compute_energy(new_leaf, inv_mass) - energy_init
        is_div = (delta_H > max_energy_delta) | ~jnp.isfinite(new_leaf.log_prob)

        neg_delta = -delta_H
        log_w = jnp.where(
            jnp.isfinite(neg_delta), neg_delta,
            jnp.where(neg_delta > 0, max_energy_delta, -jnp.inf),
        )
        new_lsw = jnp.logaddexp(lsw, log_w)
        accept = jrandom.uniform(step_key) < jnp.exp(log_w - new_lsw)

        new_prop_z = jnp.where(accept, new_leaf.z, prop_z)
        new_prop_lp = jnp.where(accept, new_leaf.log_prob, prop_lp)
        new_prop_grad = jnp.where(accept, new_leaf.grad, prop_grad)

        new_first_z = jnp.where(is_first, new_leaf.z, first_z)
        new_first_r = jnp.where(is_first, new_leaf.r, first_r)

        mh = jnp.minimum(1.0, jnp.exp(-delta_H))
        mh = jnp.where(jnp.isfinite(mh), mh, 0.0)

        return (new_leaf, new_first_z, new_first_r,
                new_prop_z, new_prop_lp, new_prop_grad,
                new_lsw, diverging | is_div, sap + mh, n_prop + 1,
                rng, jnp.array(False))

    result = jax.lax.fori_loop(0, num_steps, step_fn, init_carry)
    (endpoint, first_z, first_r, prop_z, prop_lp, prop_grad,
     lsw, diverging, sap, n_prop, rng, _) = result

    diff = endpoint.z - first_z
    subtree_turn = (jnp.dot(diff, endpoint.r) < 0) | (jnp.dot(diff, first_r) < 0)
    subtree_turn = subtree_turn & (n_prop > 1)

    return _SubtreeInfo(
        endpoint=endpoint,
        first_z=first_z,
        first_r=first_r,
        proposal_z=prop_z,
        proposal_lp=prop_lp,
        proposal_grad=prop_grad,
        log_sum_weight=lsw,
        turning=subtree_turn | diverging,
        diverging=diverging,
        sum_accept_prob=sap,
        num_proposals=n_prop,
        rng_key=rng,
    )


def _nuts_step(
    potential_and_grad_fn, state, rng_key, step_size, inv_mass,
    max_depth, max_energy_delta=1000.0, mass_chol=None,
):
    """Один шаг NUTS: bidirectional tree doubling с multinomial proposal.

    Dense mass: momentum через mass_chol, kinetic energy через inv_mass.
    """
    mom_key, rng_key = jrandom.split(rng_key)
    dim = state.position.shape[0]

    if mass_chol is None:
        mass_chol = jnp.eye(dim)
    momentum = sample_momentum(mom_key, inv_mass, mass_chol)

    init_leaf = _Leaf(state.position, momentum, state.log_prob, state.grad)
    energy_init = _compute_energy(init_leaf, inv_mass)

    tree = _TreeState(
        left_z=state.position, left_r=momentum,
        left_lp=state.log_prob, left_grad=state.grad,
        right_z=state.position, right_r=momentum,
        right_lp=state.log_prob, right_grad=state.grad,
        proposal_z=state.position, proposal_lp=state.log_prob,
        proposal_grad=state.grad,
        log_sum_weight=jnp.zeros(()),
        depth=jnp.zeros((), dtype=jnp.int32),
        turning=jnp.array(False),
        diverging=jnp.array(False),
        sum_accept_prob=jnp.zeros(()),
        num_proposals=jnp.ones((), dtype=jnp.int32),
        energy_init=energy_init,
        rng_key=rng_key,
    )

    def cond_fn(tree):
        return (~tree.turning) & (~tree.diverging) & (tree.depth < max_depth)

    def body_fn(tree):
        rng, dir_key, sub_key, merge_key = jrandom.split(tree.rng_key, 4)

        going_right = jrandom.bernoulli(dir_key)
        direction = jnp.where(going_right, 1.0, -1.0)

        start_z = jnp.where(going_right, tree.right_z, tree.left_z)
        start_r = jnp.where(going_right, tree.right_r, tree.left_r)
        start_lp = jnp.where(going_right, tree.right_lp, tree.left_lp)
        start_grad = jnp.where(going_right, tree.right_grad, tree.left_grad)
        start_leaf = _Leaf(start_z, start_r, start_lp, start_grad)

        sub = _integrate_subtree(
            potential_and_grad_fn, start_leaf, direction, tree.depth,
            step_size, inv_mass, tree.energy_init, max_energy_delta, sub_key,
        )

        new_lsw = jnp.logaddexp(tree.log_sum_weight, sub.log_sum_weight)
        accept_sub = jrandom.uniform(merge_key) < jnp.exp(
            sub.log_sum_weight - new_lsw
        )

        prop_z = jnp.where(accept_sub, sub.proposal_z, tree.proposal_z)
        prop_lp = jnp.where(accept_sub, sub.proposal_lp, tree.proposal_lp)
        prop_grad = jnp.where(accept_sub, sub.proposal_grad, tree.proposal_grad)

        new_left_z = jnp.where(going_right, tree.left_z, sub.endpoint.z)
        new_left_r = jnp.where(going_right, tree.left_r, sub.endpoint.r)
        new_left_lp = jnp.where(going_right, tree.left_lp, sub.endpoint.log_prob)
        new_left_grad = jnp.where(going_right, tree.left_grad, sub.endpoint.grad)

        new_right_z = jnp.where(going_right, sub.endpoint.z, tree.right_z)
        new_right_r = jnp.where(going_right, sub.endpoint.r, tree.right_r)
        new_right_lp = jnp.where(going_right, sub.endpoint.log_prob, tree.right_lp)
        new_right_grad = jnp.where(going_right, sub.endpoint.grad, tree.right_grad)

        diff = new_right_z - new_left_z
        full_turn = (jnp.dot(diff, new_left_r) < 0) | (jnp.dot(diff, new_right_r) < 0)
        turning = full_turn | sub.turning

        return _TreeState(
            left_z=new_left_z, left_r=new_left_r,
            left_lp=new_left_lp, left_grad=new_left_grad,
            right_z=new_right_z, right_r=new_right_r,
            right_lp=new_right_lp, right_grad=new_right_grad,
            proposal_z=prop_z, proposal_lp=prop_lp, proposal_grad=prop_grad,
            log_sum_weight=new_lsw,
            depth=tree.depth + 1,
            turning=turning,
            diverging=tree.diverging | sub.diverging,
            sum_accept_prob=tree.sum_accept_prob + sub.sum_accept_prob,
            num_proposals=tree.num_proposals + sub.num_proposals,
            energy_init=tree.energy_init,
            rng_key=rng,
        )

    final = jax.lax.while_loop(cond_fn, body_fn, tree)

    new_state = NUTSState(final.proposal_z, final.proposal_lp, final.proposal_grad)
    avg_accept = final.sum_accept_prob / jnp.maximum(
        final.num_proposals.astype(jnp.float32), 1.0
    )

    return new_state, avg_accept, final.depth, final.diverging


def nuts_sample(
    potential_fn, rng_key, init_position,
    num_samples=1000, num_warmup=500, step_size=0.01,
    max_tree_depth=10, target_accept=0.8, max_energy_delta=1000.0,
    dense_mass=False, _skip_step_size_init=False,
):
    """Bidirectional NUTS: windowed warmup + sampling через lax.scan.

    Windowed warmup: Stan-schedule с finalize_mass в конце каждого окна.
    Dense mass: full covariance mass через matmul в leapfrog.

    Возвращает (samples, log_probs, accept_probs, info).
    """
    if max_tree_depth < 1:
        raise ValueError(f"max_tree_depth должен быть >= 1, получено {max_tree_depth}")

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
            logger.warning(
                "find_reasonable_step_size failed (%s: %s), using default step_size=%s",
                type(e).__name__, e, step_size,
            )

    nuts_state = NUTSState(init_position, init_lp, init_grad)
    warmup_state = make_warmup_state(step_size, dim, dense=dense_mass)

    schedule = build_adaptation_schedule(num_warmup)

    def _get_mass(warm_st):
        if dense_mass:
            return warm_st.inv_mass_dense, warm_st.mass_chol
        return warm_st.inv_mass_diag, warm_st.mass_chol

    def buffer_body(carry, rng_key):
        """Initial/final buffer: только dual averaging, БЕЗ Welford."""
        nuts_st, warm_st, step_num = carry
        inv_m, m_chol = _get_mass(warm_st)

        new_nuts, accept_prob, _depth, _div = _nuts_step(
            potential_and_grad, nuts_st, rng_key,
            warm_st.step_size, inv_m,
            max_tree_depth, max_energy_delta,
            mass_chol=m_chol,
        )

        new_warm = dual_averaging_update(warm_st, accept_prob, step_num, target_accept)
        return (new_nuts, new_warm, step_num + 1), accept_prob

    def adapt_body(carry, rng_key):
        """Adaptation window: dual averaging + Welford для mass matrix."""
        nuts_st, warm_st, step_num = carry
        inv_m, m_chol = _get_mass(warm_st)

        new_nuts, accept_prob, _depth, _div = _nuts_step(
            potential_and_grad, nuts_st, rng_key,
            warm_st.step_size, inv_m,
            max_tree_depth, max_energy_delta,
            mass_chol=m_chol,
        )

        new_warm = dual_averaging_update(warm_st, accept_prob, step_num, target_accept)
        new_warm = welford_update(new_warm, new_nuts.position)
        return (new_nuts, new_warm, step_num + 1), accept_prob

    warmup_keys = jrandom.split(rng_key, num_warmup + 1)
    rng_key_post = warmup_keys[0]
    warmup_rng = warmup_keys[1:]

    is_windowed = len(schedule) > 1

    if is_windowed:
        init_end = schedule[0][0]
        step_counter = jnp.ones((), dtype=jnp.int32)

        if init_end > 0:
            (nuts_state, warmup_state, step_counter), _ = jax.lax.scan(
                buffer_body,
                (nuts_state, warmup_state, step_counter),
                warmup_rng[:init_end],
            )

        for win_start, win_end in schedule:
            win_keys = warmup_rng[win_start:win_end]
            (nuts_state, warmup_state, step_counter), _ = jax.lax.scan(
                adapt_body,
                (nuts_state, warmup_state, step_counter),
                win_keys,
            )
            warmup_state = finalize_mass(warmup_state)

        final_start = schedule[-1][1]
        if final_start < num_warmup:
            (nuts_state, warmup_state, step_counter), _ = jax.lax.scan(
                buffer_body,
                (nuts_state, warmup_state, step_counter),
                warmup_rng[final_start:],
            )
    else:
        (nuts_state, warmup_state, _), _ = jax.lax.scan(
            adapt_body,
            (nuts_state, warmup_state, jnp.ones((), dtype=jnp.int32)),
            warmup_rng,
        )
        warmup_state = finalize_mass(warmup_state)

    final_step_size = jnp.exp(warmup_state.log_step_size_avg)
    final_inv_mass, final_mass_chol = _get_mass(warmup_state)

    def sample_body(nuts_st, rng_key):
        new_st, accept_prob, depth, diverging = _nuts_step(
            potential_and_grad, nuts_st, rng_key,
            final_step_size, final_inv_mass,
            max_tree_depth, max_energy_delta,
            mass_chol=final_mass_chol,
        )
        return new_st, (new_st.position, new_st.log_prob, accept_prob, depth, diverging)

    sample_keys = jrandom.split(rng_key_post, num_samples)
    _, (samples, log_probs, accept_probs, depths, divergences) = jax.lax.scan(
        sample_body, nuts_state, sample_keys,
    )

    num_max_depth = jnp.sum((depths >= max_tree_depth).astype(jnp.int32))

    info = {
        "step_size": final_step_size,
        "inv_mass_diag": warmup_state.inv_mass_diag,
        "dense_mass": dense_mass,
        "warmup_schedule": schedule,
        "tree_depths": depths,
        "num_divergences": jnp.sum(divergences.astype(jnp.int32)),
        "mean_tree_depth": jnp.mean(depths.astype(jnp.float32)),
        "accept_probs": accept_probs,
        "mean_accept_prob": jnp.mean(accept_probs),
        "max_tree_depth": max_tree_depth,
        "num_max_treedepth": num_max_depth,
    }
    if dense_mass:
        info["inv_mass_dense"] = final_inv_mass

    try:
        n_max = int(num_max_depth)
        frac_max = n_max / max(num_samples, 1)
        if frac_max > 0.05:
            logger.warning(
                "NUTS: %d/%d transitions (%.0f%%) hit max_tree_depth=%d. "
                "Consider increasing max_tree_depth or tuning step_size.",
                n_max, num_samples, frac_max * 100, max_tree_depth,
            )
    except (jax.errors.ConcretizationTypeError, TypeError):
        pass

    return samples, log_probs, accept_probs, info


def nuts_sample_chains(
    potential_fn, rng_key, init_positions, num_chains=4, **kwargs,
):
    """Multi-chain bidirectional NUTS через vmap.

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
    vmapped = jax.vmap(lambda k, pos: nuts_sample(potential_fn, k, pos, **kwargs))
    return vmapped(keys, init_positions)


class NUTSKernel(InferenceEngine):
    """NUTS обёртка для ExperimentRunner. Внутри — bidirectional tree doubling."""

    def __init__(self):
        self._backend = None

    def run(self, backend, rng_key, config) -> InferenceResult:
        init_key, sample_key = jrandom.split(rng_key)
        prior = backend.sample_prior(init_key)
        init_pos = backend.flatten_latents(prior)
        num_chains = config.get("num_chains", 1)

        nuts_kwargs = {
            "num_samples": config.get("num_samples", 1000),
            "num_warmup": config.get("num_warmup", 500),
            "step_size": config.get("step_size", 0.01),
            "max_tree_depth": config.get("max_tree_depth", 10),
            "target_accept": config.get("target_accept_rate", 0.8),
            "max_energy_delta": config.get("max_energy_delta", 1000.0),
            "dense_mass": config.get("dense_mass", False),
        }

        if num_chains > 1:
            chain_keys = jrandom.split(init_key, num_chains)
            init_positions = jnp.stack([
                backend.flatten_latents(backend.sample_prior(k))
                for k in chain_keys
            ])
            samples_flat, log_probs, accept_probs, info = nuts_sample_chains(
                backend.potential_fn, sample_key, init_positions,
                num_chains=num_chains, **nuts_kwargs,
            )
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
                    if k in ("num_divergences", "num_max_treedepth"):
                        agg_info[k] = int(jnp.sum(v))
                    else:
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
            flat_samples, log_probs_out, accept_probs, info = nuts_sample(
                backend.potential_fn, sample_key, init_pos, **nuts_kwargs,
            )

            latent_names = list(backend.get_latent_shapes().keys())
            unflat_batched = jax.vmap(backend.unflatten_latents)(flat_samples)
            samples = {name: unflat_batched[name] for name in latent_names}

            return InferenceResult(
                samples=samples,
                log_probs=log_probs_out,
                diagnostics={**info, "accept_probs": accept_probs},
            )
