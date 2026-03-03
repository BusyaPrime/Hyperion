"""Laplace Approximation для HYPERION.

Самый ленивый способ аппроксимации постериора: MAP + гессиан. Но работает!
Постериор ≈ N(MAP, H^{-1}). Для больших n — асимптотически точно. Для малых — хотя бы быстро.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from hyperion_inference.base import InferenceEngine, InferenceState, InferenceResult


@dataclass
class LaplaceConfig:
    """Конфиг Laplace: MAP optimization, Hessian vs Fisher, line search fallback."""
    num_posterior_samples: int = 1000
    max_optim_steps: int = 1000
    learning_rate: float = 0.01
    convergence_tol: float = 1e-6
    use_full_hessian: bool = True
    use_fisher: bool = False  # Fisher information вместо Hessian (для экспоненциальных семейств)
    use_line_search: bool = True
    line_search_max_iter: int = 20


class LaplaceApproximation(InferenceEngine):
    """Laplace: находим MAP, считаем H (или F), семплим из N(MAP, H^{-1})."""

    def __init__(self):
        self._backend = None

    def initialize(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> InferenceState:
        self._backend = backend
        return InferenceState(step=0, rng_key=rng_key)

    def _find_map(
        self,
        init_position: jnp.ndarray,
        max_steps: int,
        lr: float,
        tol: float,
        use_line_search: bool = True,
        line_search_max_iter: int = 20,
    ) -> tuple[jnp.ndarray, float]:
        """MAP через gradient ascent. Line search: если не приняли шаг — уменьшаем lr и пробуем снова."""
        position = init_position

        for i in range(max_steps):
            log_p, grad = self._backend.potential_and_grad_fn(position)
            grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
            grad_norm = float(jnp.linalg.norm(grad))

            if grad_norm < tol:
                break

            step_size = lr
            if use_line_search:
                # Backtracking line search: уменьшаем шаг пока log_p не вырастет
                for _ in range(line_search_max_iter):
                    proposal = position + step_size * grad
                    log_p_new = float(self._backend.potential_fn(proposal))

                    if jnp.isfinite(log_p_new) and log_p_new > log_p:
                        position = proposal
                        log_p = log_p_new
                        break
                    step_size *= 0.5
                else:
                    # Line search не нашёл улучшение — делаем обычный шаг
                    position = position + lr * grad
                    log_p = float(self._backend.potential_fn(position))
            else:
                position = position + step_size * grad
                log_p = float(self._backend.potential_fn(position))

        log_p = float(self._backend.potential_fn(position))
        return position, log_p

    def _compute_hessian(self, position: jnp.ndarray) -> jnp.ndarray:
        """Гессиан от -log p(x,z) в MAP. JIT-компиляция hessian — без Python overhead."""
        hessian_fn = jax.jit(jax.hessian(lambda z: -self._backend.potential_fn(z)))
        return hessian_fn(position)

    def _compute_fisher(self, position: jnp.ndarray) -> jnp.ndarray:
        """Fisher information: E[grad log p * grad log p^T]. Для гауссиана ≈ Hessian, но иногда стабильнее."""
        grad_val = self._backend.grad_fn(position)
        # Outer product: F = grad @ grad.T. Для одного сэмпла — приближение.
        F = jnp.outer(grad_val, grad_val)
        # Добавляем диагональ для положительной определённости
        F = F + 1e-6 * jnp.eye(F.shape[0])
        return F

    def step(self, state: InferenceState) -> InferenceState:
        return InferenceState(step=state.step + 1, rng_key=state.rng_key)

    def get_samples(self, state: InferenceState) -> dict[str, jnp.ndarray]:
        return {}

    def get_metrics(self, state: InferenceState) -> dict[str, Any]:
        return {}

    def run(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> InferenceResult:
        self._backend = backend
        cfg = LaplaceConfig(**{k: v for k, v in config.items() if hasattr(LaplaceConfig, k)})

        # Инициализация из приора — один сэмпл
        init_key, sample_key = jrandom.split(rng_key)
        prior_samples = backend.sample_prior(init_key)
        init_position = backend.flatten_latents(prior_samples)

        # Ищем MAP с line search fallback
        map_position, map_log_p = self._find_map(
            init_position,
            cfg.max_optim_steps,
            cfg.learning_rate,
            cfg.convergence_tol,
            use_line_search=cfg.use_line_search,
            line_search_max_iter=cfg.line_search_max_iter,
        )

        # Считаем precision: Hessian или Fisher
        if cfg.use_fisher:
            H = self._compute_fisher(map_position)
        elif cfg.use_full_hessian:
            H = self._compute_hessian(map_position)
        else:
            H = None

        # Covariance = H^{-1}
        if H is not None:
            try:
                cov = jnp.linalg.inv(H)
                if not jnp.all(jnp.isfinite(cov)):
                    raise ValueError("Обратный гессиан содержит NaN/Inf")
                cov = 0.5 * (cov + cov.T)

                eigvals = jnp.linalg.eigvalsh(cov)
                min_eig = jnp.min(eigvals)
                if min_eig <= 0:
                    cov = cov + jnp.eye(cov.shape[0]) * (jnp.abs(min_eig) + 1e-4)
            except (ValueError, RuntimeError) as e:
                dim = map_position.shape[0]
                cov = jnp.eye(dim) * 0.01
        else:
            dim = map_position.shape[0]
            cov = jnp.eye(dim) * 0.01

        # Семплим из N(MAP, cov)
        L = jnp.linalg.cholesky(cov)
        eps = jrandom.normal(sample_key, shape=(cfg.num_posterior_samples, map_position.shape[0]))
        flat_samples = map_position[None, :] + eps @ L.T

        # Unflatten в структуру латентных. vmap — батчим unflatten без цикла
        latent_names = list(backend.get_latent_shapes().keys())
        unflat_batched = jax.vmap(backend.unflatten_latents)(flat_samples)
        samples = {name: unflat_batched[name] for name in latent_names}

        return InferenceResult(
            samples=samples,
            diagnostics={
                "map_position": map_position,
                "map_log_prob": map_log_p,
                "covariance": cov,
                "hessian": H if cfg.use_full_hessian else None,
                "used_fisher": cfg.use_fisher,
            },
        )
