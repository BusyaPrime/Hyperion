"""Variational Inference (VI) для HYPERION.

ELBO — нижняя оценка маргинального правдоподобия. Чем больше, тем лучше (обычно).
Reparameterization trick — семплим eps~N(0,1), потом z = mu + sigma*eps. JAX скажет спасибо.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from hyperion_inference.base import InferenceEngine, InferenceState, InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class VIConfig:
    """Конфиг VI: mean-field, low-rank или full-rank. Natural gradient — опция для продвинутых."""
    num_steps: int = 5000
    learning_rate: float = 0.01
    num_elbo_samples: int = 10
    num_posterior_samples: int = 1000
    covariance_type: Literal["diagonal", "low_rank", "full_rank"] = "diagonal"
    rank: int = 5
    lr_schedule: str = "cosine"
    clip_grad_norm: float = 10.0
    convergence_tol: float = 1e-6
    patience: int = 100
    use_natural_gradient: bool = False
    natural_grad_damping: float = 1e-4


@dataclass
class VIState(InferenceState):
    """Состояние VI: params, opt_state, история ELBO. best_params — для early stopping."""
    params: Optional[dict[str, jnp.ndarray]] = None
    opt_state: Optional[Any] = None
    elbo_history: list = field(default_factory=list)
    best_elbo: float = float("-inf")
    best_params: Optional[dict[str, jnp.ndarray]] = None
    steps_without_improvement: int = 0
    converged: bool = False

    num_steps: int = 5000
    num_elbo_samples: int = 10
    covariance_type: str = "diagonal"
    rank: int = 5
    clip_grad_norm: float = 10.0
    convergence_tol: float = 1e-6
    patience: int = 100
    use_natural_gradient: bool = False
    natural_grad_damping: float = 1e-4


class VIEngine(InferenceEngine):
    """VI с mean-field (diagonal), low-rank или full-rank Gaussian. Natural gradient — бонус."""

    def __init__(self):
        self._backend = None
        self._optimizer = None

    def _init_params(self, dim: int, cov_type: str, rank: int) -> dict[str, jnp.ndarray]:
        """Инициализация: mu=0, log_sigma=-1. Full-rank — L для Cholesky Sigma = L@L.T."""
        params = {
            "mu": jnp.zeros(dim),
            "log_sigma": jnp.zeros(dim) - 1.0,
        }
        if cov_type == "low_rank":
            params["factor"] = jnp.zeros((dim, rank)) * 0.01
        elif cov_type == "full_rank":
            # L — нижнетреугольная, Sigma = L @ L.T. Инициализируем как sqrt(diag)
            params["L"] = jnp.diag(jnp.exp(jnp.zeros(dim) - 1.0))
        return params

    def _sample_q(
        self,
        key: jax.random.PRNGKey,
        params: dict[str, jnp.ndarray],
        num_samples: int,
        cov_type: str,
    ) -> jnp.ndarray:
        """Сэмплим из q(z)=N(mu, Sigma). Reparameterization: z = mu + L@eps, eps~N(0,I)."""
        mu = params["mu"]
        sigma = jnp.exp(params["log_sigma"])
        dim = mu.shape[0]

        eps = jrandom.normal(key, shape=(num_samples, dim))

        if cov_type == "diagonal":
            return mu[None, :] + sigma[None, :] * eps
        elif cov_type == "low_rank":
            factor = params["factor"]
            z = mu[None, :] + sigma[None, :] * eps
            eps2 = jrandom.normal(jrandom.fold_in(key, 1), shape=(num_samples, factor.shape[1]))
            z = z + eps2 @ factor.T
            return z
        else:  # full_rank
            L = jnp.tril(params["L"])
            return mu[None, :] + eps @ L.T

    def _log_q(
        self,
        z: jnp.ndarray,
        params: dict[str, jnp.ndarray],
        cov_type: str,
    ) -> jnp.ndarray:
        """Лог-плотность q(z). Full-rank: -0.5*(z-mu)^T Sigma^{-1} (z-mu) - 0.5*log|Sigma|."""
        mu = params["mu"]
        log_sigma = params["log_sigma"]
        sigma = jnp.exp(log_sigma)
        dim = mu.shape[0]

        if cov_type == "diagonal":
            diff = z - mu[None, :]
            return -0.5 * dim * jnp.log(2.0 * jnp.pi) - jnp.sum(log_sigma) - 0.5 * jnp.sum(
                (diff / sigma[None, :]) ** 2, axis=-1
            )
        elif cov_type == "low_rank":
            factor = params["factor"]
            diff = z - mu[None, :]
            inv_diag = 1.0 / (sigma ** 2 + 1e-8)

            # Woodbury: Σ^{-1} = D^{-1} - D^{-1} F (I + F^T D^{-1} F)^{-1} F^T D^{-1}
            # Mahalanobis: diff^T Σ^{-1} diff
            scaled_diff = diff * inv_diag[None, :]  # diff @ D^{-1}
            diag_mahal = jnp.sum(diff * scaled_diff, axis=-1)  # diff^T D^{-1} diff

            scaled_factor = factor * inv_diag[:, None]  # D^{-1} F, shape [dim, rank]
            capacitance = jnp.eye(factor.shape[1]) + factor.T @ scaled_factor  # I + F^T D^{-1} F
            v = scaled_diff @ factor  # [n_samples, rank]
            solve = jnp.linalg.solve(capacitance, v.T).T  # [n_samples, rank]
            woodbury_correction = jnp.sum(v * solve, axis=-1)

            mahal = diag_mahal - woodbury_correction

            # log|Σ| = log|D| + log|I + F^T D^{-1} F|
            _, log_det_cap = jnp.linalg.slogdet(capacitance)
            log_det = 2.0 * jnp.sum(log_sigma) + log_det_cap

            return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + log_det + mahal)
        else:  # full_rank
            L = jnp.tril(params["L"])
            diff = z - mu[None, :]
            y = jax.scipy.linalg.solve_triangular(L, diff.T, lower=True)
            mahal = jnp.sum(y ** 2, axis=0)
            log_det = 2.0 * jnp.sum(jnp.log(jnp.maximum(jnp.abs(jnp.diag(L)), 1e-8)))
            return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + log_det + mahal)

    def _compute_elbo(
        self,
        key: jax.random.PRNGKey,
        params: dict[str, jnp.ndarray],
        num_samples: int,
        cov_type: str,
    ) -> jnp.ndarray:
        """ELBO = E_q[log p(x,z)] - E_q[log q(z)]. vmap по сэмплам — батчим potential_fn."""
        z_samples = self._sample_q(key, params, num_samples, cov_type)

        log_joints = jax.vmap(self._backend.potential_fn)(z_samples)

        log_qs = self._log_q(z_samples, params, cov_type)

        elbo = jnp.mean(log_joints - log_qs)
        return elbo

    def _natural_gradient_precondition(
        self,
        params: dict[str, jnp.ndarray],
        grads: dict[str, jnp.ndarray],
        cov_type: str,
        damping: float,
    ) -> dict[str, jnp.ndarray]:
        """Natural gradient: grad_nat = F^{-1} @ grad. F — Fisher. Для diagonal — F = 1/sigma^2."""
        if cov_type != "diagonal":
            return grads  # Full Fisher для low-rank/full — сложно, пока пропускаем

        sigma = jnp.exp(params["log_sigma"])
        # Fisher для mu: 1/sigma^2. Для log_sigma: 2 (для каждого компонента)
        precond_mu = 1.0 / (sigma ** 2 + damping)
        precond_log_sigma = jnp.full_like(params["log_sigma"], 0.5)  # Упрощённо

        return {
            "mu": grads["mu"] * precond_mu,
            "log_sigma": grads["log_sigma"] * precond_log_sigma,
            **{k: v for k, v in grads.items() if k not in ("mu", "log_sigma")},
        }

    def initialize(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> VIState:
        self._backend = backend
        cfg = VIConfig(**{k: v for k, v in config.items() if hasattr(VIConfig, k)})

        dim = backend.total_dim

        params = self._init_params(dim, cfg.covariance_type, cfg.rank)

        if cfg.lr_schedule == "cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=cfg.learning_rate,
                decay_steps=cfg.num_steps,
            )
        elif cfg.lr_schedule == "exponential":
            schedule = optax.exponential_decay(
                init_value=cfg.learning_rate,
                transition_steps=cfg.num_steps // 5,
                decay_rate=0.5,
            )
        else:
            schedule = cfg.learning_rate

        self._optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.clip_grad_norm),
            optax.adam(schedule),
        )

        opt_state = self._optimizer.init(params)

        return VIState(
            step=0,
            rng_key=rng_key,
            params=params,
            opt_state=opt_state,
            num_steps=cfg.num_steps,
            num_elbo_samples=cfg.num_elbo_samples,
            covariance_type=cfg.covariance_type,
            rank=cfg.rank,
            clip_grad_norm=cfg.clip_grad_norm,
            convergence_tol=cfg.convergence_tol,
            patience=cfg.patience,
            use_natural_gradient=cfg.use_natural_gradient,
            natural_grad_damping=cfg.natural_grad_damping,
        )

    def step(self, state: VIState) -> VIState:
        if state.converged:
            return state

        key, elbo_key = jrandom.split(state.rng_key)

        def neg_elbo(params):
            return -self._compute_elbo(
                elbo_key, params, state.num_elbo_samples, state.covariance_type
            )

        loss, grads = jax.value_and_grad(neg_elbo)(state.params)
        elbo = -float(loss)

        # Natural gradient preconditioning если включено
        if state.use_natural_gradient:
            grads = self._natural_gradient_precondition(
                state.params, grads, state.covariance_type, state.natural_grad_damping
            )

        updates, new_opt_state = self._optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        # Track best
        best_elbo = state.best_elbo
        best_params = state.best_params
        steps_no_improve = state.steps_without_improvement

        if elbo > best_elbo:
            best_elbo = elbo
            best_params = new_params
            steps_no_improve = 0
        else:
            steps_no_improve += 1

        # Улучшенная проверка сходимости: relative ELBO change + gradient norm
        grad_norm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads.values())))
        converged = False

        if steps_no_improve >= state.patience:
            converged = True

        if len(state.elbo_history) > 10:
            recent = jnp.array(state.elbo_history[-10:])
            rel_change = jnp.std(recent) / (jnp.abs(jnp.mean(recent)) + 1e-10)
            if rel_change < state.convergence_tol:
                converged = True
            if grad_norm < state.convergence_tol * 10:
                converged = True

        return VIState(
            step=state.step + 1,
            rng_key=key,
            params=new_params,
            opt_state=new_opt_state,
            elbo_history=state.elbo_history + [elbo],
            best_elbo=best_elbo,
            best_params=best_params,
            steps_without_improvement=steps_no_improve,
            converged=converged,
            num_steps=state.num_steps,
            num_elbo_samples=state.num_elbo_samples,
            covariance_type=state.covariance_type,
            rank=state.rank,
            clip_grad_norm=state.clip_grad_norm,
            convergence_tol=state.convergence_tol,
            patience=state.patience,
            use_natural_gradient=state.use_natural_gradient,
            natural_grad_damping=state.natural_grad_damping,
        )

    def get_samples(self, state: VIState) -> dict[str, jnp.ndarray]:
        params = state.best_params or state.params
        if params is None:
            return {}

        key, _ = jrandom.split(state.rng_key)
        num_samples = 1000

        flat_samples = self._sample_q(key, params, num_samples, state.covariance_type)

        latent_names = list(self._backend.get_latent_shapes().keys())
        unflat_batched = jax.vmap(self._backend.unflatten_latents)(flat_samples)
        return {name: unflat_batched[name] for name in latent_names}

    def get_metrics(self, state: VIState) -> dict[str, Any]:
        return {
            "elbo_history": state.elbo_history,
            "best_elbo": state.best_elbo,
            "converged": state.converged,
            "num_steps": state.step,
            "covariance_type": state.covariance_type,
        }

    def run(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> InferenceResult:
        state = self.initialize(backend, rng_key, config)
        num_steps = config.get("num_steps", 5000)
        logger.info("VI started: %d steps, covariance=%s, lr=%s",
                     num_steps, config.get("covariance_type", "diagonal"),
                     config.get("learning_rate", 0.01))

        for i in range(num_steps):
            state = self.step(state)
            if i > 0 and i % 500 == 0:
                logger.debug("VI step %d/%d, ELBO=%.4f", i, num_steps,
                             float(state.metrics.get("elbo", float("nan"))))
            if state.converged:
                logger.info("VI converged at step %d", i)
                break

        num_posterior = config.get("num_posterior_samples", 1000)
        params = state.best_params or state.params
        flat_samples = self._sample_q(
            state.rng_key, params, num_posterior, state.covariance_type
        )

        latent_names = list(self._backend.get_latent_shapes().keys())
        unflat_batched = jax.vmap(self._backend.unflatten_latents)(flat_samples)
        samples = {name: unflat_batched[name] for name in latent_names}

        return InferenceResult(
            samples=samples,
            diagnostics=self.get_metrics(state),
            metadata={"variational_params": state.best_params or state.params},
        )
