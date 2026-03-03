"""Sequential Monte Carlo (SMC) для HYPERION.

Партиклы ресемплим когда ESS падает ниже порога — классический bootstrap PF.
Темперинг + MALA-реjuvenation = получаем сэмплы из постериора без MCMC-болей.
vmap по партиклам — батчим potential_fn, без Python-циклов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from hyperion_inference.base import InferenceEngine, InferenceState, InferenceResult


@dataclass
class SMCConfig:
    """Конфиг SMC: партиклы, темперинг, порог ESS, схема ресемплинга."""
    num_particles: int = 1000
    num_tempering_steps: int = 20
    ess_threshold: float = 0.5
    rejuvenation_steps: int = 5
    rejuvenation_step_size: float = 0.01
    adaptive_tempering: bool = True
    target_ess_ratio: float = 0.5
    resampling_scheme: Literal["systematic", "multinomial", "stratified"] = "systematic"


@dataclass
class SMCState(InferenceState):
    """Состояние SMC: позиции партиклов, лог-веса, история ancestry для отладки."""
    positions: Optional[jnp.ndarray] = None
    log_weights: Optional[jnp.ndarray] = None
    betas: list = field(default_factory=list)
    ess_history: list = field(default_factory=list)
    ancestry: Optional[jnp.ndarray] = None  # Индексы родителей после ресемплинга
    log_evidence: float = 0.0

    num_particles: int = 1000
    num_tempering_steps: int = 20
    ess_threshold: float = 0.5
    rejuvenation_steps: int = 5
    rejuvenation_step_size: float = 0.01
    adaptive_tempering: bool = True
    target_ess_ratio: float = 0.5
    resampling_scheme: str = "systematic"


class SMCEngine(InferenceEngine):
    """SMC с темперингом и MALA-реjuvenation. ESS показывает сколько РЕАЛЬНО независимых сэмплов."""

    def __init__(self):
        self._backend = None

    def initialize(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> SMCState:
        self._backend = backend
        cfg = SMCConfig(**{k: v for k, v in config.items() if hasattr(SMCConfig, k)})

        keys = jrandom.split(rng_key, cfg.num_particles + 1)
        rng_key = keys[0]
        init_keys = keys[1:]

        # vmap: батчим sample_prior + flatten — без Python-цикла по партиклам
        positions = jax.vmap(
            lambda k: backend.flatten_latents(backend.sample_prior(k))
        )(init_keys)

        # Лог-веса в log-space: равномерно = -log(N). Никаких exp до последнего момента!
        log_weights = jnp.full(cfg.num_particles, -jnp.log(cfg.num_particles))

        # Ancestry: изначально каждый партикл — сам себе родитель
        ancestry = jnp.arange(cfg.num_particles)

        return SMCState(
            step=0,
            rng_key=rng_key,
            positions=positions,
            log_weights=log_weights,
            betas=[0.0],
            ancestry=ancestry,
            log_evidence=0.0,
            num_particles=cfg.num_particles,
            num_tempering_steps=cfg.num_tempering_steps,
            ess_threshold=cfg.ess_threshold,
            rejuvenation_steps=cfg.rejuvenation_steps,
            rejuvenation_step_size=cfg.rejuvenation_step_size,
            adaptive_tempering=cfg.adaptive_tempering,
            target_ess_ratio=cfg.target_ess_ratio,
            resampling_scheme=getattr(cfg, "resampling_scheme", "systematic"),
        )

    def _compute_ess(self, log_weights: jnp.ndarray) -> float:
        """ESS = 1 / sum(w^2). В log-space: logsumexp(2*log_w_norm) — численно стабильно."""
        log_w_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
        return float(jnp.exp(-jax.scipy.special.logsumexp(2.0 * log_w_normalized)))

    def _systematic_resample(
        self,
        key: jax.random.PRNGKey,
        log_weights: jnp.ndarray,
        positions: jnp.ndarray,
        ancestry: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Systematic resampling — меньше variance чем multinomial, детерминированнее."""
        N = positions.shape[0]
        weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))
        cumsum = jnp.cumsum(weights)

        u = jrandom.uniform(key) / N
        u_vec = u + jnp.arange(N, dtype=jnp.float32) / N

        indices = jnp.searchsorted(cumsum, u_vec)
        indices = jnp.clip(indices, 0, N - 1)
        return positions[indices], ancestry[indices]

    def _multinomial_resample(
        self,
        key: jax.random.PRNGKey,
        log_weights: jnp.ndarray,
        positions: jnp.ndarray,
        ancestry: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Multinomial resampling — классика, но variance побольше."""
        N = positions.shape[0]
        weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))
        indices = jrandom.categorical(key, jnp.log(weights + 1e-10), shape=(N,))
        return positions[indices], ancestry[indices]

    def _stratified_resample(
        self,
        key: jax.random.PRNGKey,
        log_weights: jnp.ndarray,
        positions: jnp.ndarray,
        ancestry: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Stratified — делим [0,1] на N страт, в каждой страте один uniform. Хороший компромисс."""
        N = positions.shape[0]
        weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))
        cumsum = jnp.cumsum(weights)

        keys = jrandom.split(key, N)
        u_strat = (jnp.arange(N) + jrandom.uniform(keys[0], (N,))) / N

        indices = jnp.searchsorted(cumsum, u_strat)
        indices = jnp.clip(indices, 0, N - 1)
        return positions[indices], ancestry[indices]

    def _resample(
        self,
        key: jax.random.PRNGKey,
        log_weights: jnp.ndarray,
        positions: jnp.ndarray,
        ancestry: jnp.ndarray,
        scheme: str,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Роутер: выбираем схему ресемплинга. Systematic по умолчанию — работает лучше всех."""
        if scheme == "multinomial":
            return self._multinomial_resample(key, log_weights, positions, ancestry)
        if scheme == "stratified":
            return self._stratified_resample(key, log_weights, positions, ancestry)
        return self._systematic_resample(key, log_weights, positions, ancestry)

    def _find_next_beta(
        self,
        current_beta: float,
        positions: jnp.ndarray,
        log_weights: jnp.ndarray,
        target_ess_ratio: float,
    ) -> float:
        """Бисекция: ищем следующий beta чтобы ESS не упал ниже target. Темперинг на автомате."""
        low = current_beta
        high = 1.0

        N = positions.shape[0]
        target_ess = target_ess_ratio * N

        # vmap: батчим potential_fn вместо цикла — JIT-friendly
        log_joints = jax.vmap(self._backend.potential_fn)(positions)

        for _ in range(20):
            mid = (low + high) / 2.0
            delta_beta = mid - current_beta
            log_inc_weights = delta_beta * log_joints
            log_new_weights = log_weights + log_inc_weights
            log_new_weights_norm = log_new_weights - jax.scipy.special.logsumexp(log_new_weights)
            ess = float(jnp.exp(-jax.scipy.special.logsumexp(2.0 * log_new_weights_norm)))

            if ess > target_ess:
                low = mid
            else:
                high = mid

            if high - low < 1e-6:
                break

        return min(high, 1.0)

    def _rejuvenation_step(
        self,
        key: jax.random.PRNGKey,
        position: jnp.ndarray,
        beta: float,
    ) -> jnp.ndarray:
        """MALA-like rejuvenation: полшага градиента + гауссиан. Метрополис принимаем/отклоняем."""
        key1, key2 = jrandom.split(key)

        log_p = self._backend.potential_fn(position)
        grad = self._backend.grad_fn(position)
        grad = jnp.where(jnp.isfinite(grad), grad, 0.0)

        step_size = self._step_size

        proposal = (
            position
            + 0.5 * step_size ** 2 * beta * grad
            + step_size * jrandom.normal(key1, shape=position.shape)
        )

        log_p_new = self._backend.potential_fn(proposal)
        grad_new = self._backend.grad_fn(proposal)
        grad_new = jnp.where(jnp.isfinite(grad_new), grad_new, 0.0)

        # Forward/backward log-proposal для MALA (симметричность учитываем)
        diff_fwd = proposal - position - 0.5 * step_size ** 2 * beta * grad
        log_q_fwd = -0.5 * jnp.sum(diff_fwd ** 2) / step_size ** 2

        diff_bwd = position - proposal - 0.5 * step_size ** 2 * beta * grad_new
        log_q_bwd = -0.5 * jnp.sum(diff_bwd ** 2) / step_size ** 2

        log_alpha = beta * (log_p_new - log_p) + log_q_bwd - log_q_fwd
        log_alpha = jnp.minimum(0.0, log_alpha)

        accept = jrandom.uniform(key2) < jnp.exp(log_alpha)
        return jnp.where(accept, proposal, position)

    def step(self, state: SMCState) -> SMCState:
        key, resample_key, rejuv_key = jrandom.split(state.rng_key, 3)

        current_beta = state.betas[-1]

        if current_beta >= 1.0:
            return state

        if state.adaptive_tempering:
            next_beta = self._find_next_beta(
                current_beta, state.positions, state.log_weights, state.target_ess_ratio
            )
        else:
            next_beta = min(
                current_beta + 1.0 / state.num_tempering_steps, 1.0
            )

        delta_beta = next_beta - current_beta

        N = state.num_particles
        # Log-space арифметика: log_weights_new = log_weights + delta_beta * log_joints
        # vmap: батчим potential_fn по всем партиклам
        log_joints = jax.vmap(self._backend.potential_fn)(state.positions)
        log_inc_weights = delta_beta * log_joints

        log_weights_new = state.log_weights + log_inc_weights

        # Log-evidence increment: log(sum w_new) - log(sum w_old) = logsumexp(log_w_new) - logsumexp(log_w_old)
        log_z_old = jax.scipy.special.logsumexp(state.log_weights)
        log_z_new = jax.scipy.special.logsumexp(log_weights_new)
        log_evidence_increment = float(log_z_new - log_z_old)

        ess = self._compute_ess(log_weights_new)

        positions = state.positions
        ancestry = state.ancestry if state.ancestry is not None else jnp.arange(N)

        if ess < state.ess_threshold * N:
            positions, ancestry = self._resample(
                resample_key, log_weights_new, positions, ancestry,
                state.resampling_scheme,
            )
            log_weights_new = jnp.full(N, -jnp.log(N))

        # Rejuvenation — разгоняем партиклы MALA-шагами. vmap по партиклам + scan по шагам
        self._step_size = state.rejuvenation_step_size
        rejuv_keys = jrandom.split(rejuv_key, N * state.rejuvenation_steps)
        rejuv_keys = rejuv_keys.reshape(N, state.rejuvenation_steps, -1)

        def _scan_rejuv(carry: jnp.ndarray, key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            new_pos = self._rejuvenation_step(key, carry, next_beta)
            return new_pos, new_pos

        def _rejuvenate_one(keys: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
            final_pos, _ = jax.lax.scan(_scan_rejuv, pos, keys)
            return final_pos

        positions = jax.vmap(_rejuvenate_one)(rejuv_keys, positions)

        return SMCState(
            step=state.step + 1,
            rng_key=key,
            positions=positions,
            log_weights=log_weights_new,
            betas=state.betas + [next_beta],
            ess_history=state.ess_history + [ess],
            ancestry=ancestry,
            log_evidence=state.log_evidence + log_evidence_increment,
            num_particles=state.num_particles,
            num_tempering_steps=state.num_tempering_steps,
            ess_threshold=state.ess_threshold,
            rejuvenation_steps=state.rejuvenation_steps,
            rejuvenation_step_size=state.rejuvenation_step_size,
            adaptive_tempering=state.adaptive_tempering,
            target_ess_ratio=state.target_ess_ratio,
            resampling_scheme=state.resampling_scheme,
        )

    def get_samples(self, state: SMCState) -> dict[str, jnp.ndarray]:
        if state.positions is None:
            return {}

        log_w_norm = state.log_weights - jax.scipy.special.logsumexp(state.log_weights)
        ess = float(jnp.exp(-jax.scipy.special.logsumexp(2.0 * log_w_norm)))
        N = state.positions.shape[0]

        positions = state.positions
        if ess < 0.9 * N:
            key, _ = jrandom.split(state.rng_key)
            weights_norm = jnp.exp(log_w_norm)
            cumsum = jnp.cumsum(weights_norm)
            u = jrandom.uniform(key) / N
            u_vec = u + jnp.arange(N, dtype=jnp.float32) / N
            indices = jnp.searchsorted(cumsum, u_vec)
            indices = jnp.clip(indices, 0, N - 1)
            positions = positions[indices]

        latent_names = list(self._backend.get_latent_shapes().keys())
        unflat_batched = jax.vmap(self._backend.unflatten_latents)(positions)
        return {name: unflat_batched[name] for name in latent_names}

    def get_metrics(self, state: SMCState) -> dict[str, Any]:
        return {
            "log_evidence": state.log_evidence,
            "betas": state.betas,
            "ess_history": state.ess_history,
            "num_tempering_steps": len(state.betas) - 1,
            "final_ess": state.ess_history[-1] if state.ess_history else None,
            "ancestry": state.ancestry,
        }

    def run(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> InferenceResult:
        state = self.initialize(backend, rng_key, config)

        max_steps = config.get("num_tempering_steps", 20) * 2
        for _ in range(max_steps):
            state = self.step(state)
            if state.betas[-1] >= 1.0:
                break

        return InferenceResult(
            samples=self.get_samples(state),
            diagnostics=self.get_metrics(state),
        )
