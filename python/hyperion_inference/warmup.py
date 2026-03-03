"""Windowed warmup: Stan-style adaptation schedule.

Три фазы:
  1. Initial buffer (75 шагов): только dual averaging для step_size
  2. Adaptation windows (25, 50, 100, ... удваивающиеся): dual averaging + mass matrix
     В конце каждого окна: финализируем mass, сбрасываем Welford
  3. Final buffer (50 шагов): только dual averaging с финальной mass matrix

Mass matrix:
  - Diagonal: inv_mass = 1/var (Welford online variance)
  - Dense: inv_mass = L_inv^T @ L_inv (Welford online covariance + Cholesky)
  - Regularization: Stan shrinkage — scaled_cov * n/(n+5) + 1e-3 * 5/(n+5) * I

Dual averaging: Nesterov (Hoffman & Gelman 2014).
find_reasonable_step_size: бисекция по acceptance rate ~0.5. Реализация НИУ ИТМО.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom


class WarmupState(NamedTuple):
    """Полное состояние адаптации. Все поля — JAX arrays."""
    step_size: jnp.ndarray
    log_step_size_avg: jnp.ndarray
    h_bar: jnp.ndarray
    mu: jnp.ndarray
    inv_mass_diag: jnp.ndarray
    inv_mass_dense: jnp.ndarray
    mass_chol: jnp.ndarray
    use_dense: jnp.ndarray
    welford_count: jnp.ndarray
    welford_mean: jnp.ndarray
    welford_m2: jnp.ndarray
    welford_cov: jnp.ndarray


def make_warmup_state(step_size: float, dim: int, dense: bool = False) -> WarmupState:
    """Создать начальное состояние адаптации."""
    return WarmupState(
        step_size=jnp.array(step_size, dtype=jnp.float32),
        log_step_size_avg=jnp.log(jnp.array(step_size, dtype=jnp.float32)),
        h_bar=jnp.zeros(()),
        mu=jnp.log(jnp.array(10.0 * step_size, dtype=jnp.float32)),
        inv_mass_diag=jnp.ones(dim),
        inv_mass_dense=jnp.eye(dim),
        mass_chol=jnp.eye(dim),
        use_dense=jnp.array(dense),
        welford_count=jnp.zeros((), dtype=jnp.int32),
        welford_mean=jnp.zeros(dim),
        welford_m2=jnp.zeros(dim),
        welford_cov=jnp.zeros((dim, dim)),
    )


# === Dense/diagonal mass matrix helpers ===
# inv_mass.ndim проверяется на уровне Python (trace-time), не JAX-traced.
# Это валидно потому что ndim — статическое свойство формы массива.

def apply_inv_mass(inv_mass: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """M^{-1} @ r — element-wise для diagonal (1D), matmul для dense (2D)."""
    if inv_mass.ndim == 1:
        return inv_mass * r
    return inv_mass @ r


def kinetic_energy(inv_mass: jnp.ndarray, momentum: jnp.ndarray) -> jnp.ndarray:
    """0.5 * r^T M^{-1} r — handles both diagonal and dense mass."""
    if inv_mass.ndim == 1:
        return 0.5 * jnp.sum(inv_mass * momentum ** 2)
    return 0.5 * jnp.dot(momentum, inv_mass @ momentum)


def sample_momentum(
    key: jnp.ndarray, inv_mass: jnp.ndarray, mass_chol: jnp.ndarray,
) -> jnp.ndarray:
    """Семплим r ~ N(0, M). Diagonal: z * sqrt(M_ii). Dense: L @ z, L L^T = M."""
    dim = inv_mass.shape[0]
    z = jrandom.normal(key, shape=(dim,))
    if inv_mass.ndim == 1:
        return z * jnp.sqrt(1.0 / inv_mass)
    return mass_chol @ z


def dual_averaging_update(
    warmup: WarmupState,
    accept_prob: jnp.ndarray,
    step_num: jnp.ndarray,
    target_accept: float = 0.8,
    gamma: float = 0.05,
    t0: float = 10.0,
    kappa: float = 0.75,
) -> WarmupState:
    """Dual averaging по Нестерову для step_size."""
    t = jnp.maximum(step_num, 1)
    w = 1.0 / (t + t0)
    h_bar = (1.0 - w) * warmup.h_bar + w * (target_accept - accept_prob)
    log_step = warmup.mu - jnp.sqrt(t) / gamma * h_bar
    step_size = jnp.clip(jnp.exp(log_step), 1e-10, 1e2)
    m_w = t ** (-kappa)
    log_step_avg = m_w * log_step + (1.0 - m_w) * warmup.log_step_size_avg

    return warmup._replace(
        step_size=step_size, log_step_size_avg=log_step_avg, h_bar=h_bar,
    )


def welford_update(warmup: WarmupState, sample: jnp.ndarray) -> WarmupState:
    """Welford online: diagonal variance + optional full covariance.

    inv_mass_diag обновляется каждый шаг (для простого warmup).
    finalize_mass() добавляет regularization + dense (для windowed).
    """
    count = warmup.welford_count + 1
    delta = sample - warmup.welford_mean
    mean = warmup.welford_mean + delta / count
    delta2 = sample - mean
    m2 = warmup.welford_m2 + delta * delta2
    cov = warmup.welford_cov + jnp.outer(delta, delta2)

    variance = jnp.where(count > 10, m2 / (count - 1), jnp.ones_like(m2))
    inv_mass = 1.0 / jnp.maximum(variance, 1e-3)

    return warmup._replace(
        welford_count=count, welford_mean=mean, welford_m2=m2,
        welford_cov=cov, inv_mass_diag=inv_mass,
    )


def _regularize_diagonal(variance: jnp.ndarray, count: jnp.ndarray) -> jnp.ndarray:
    """Stan-style regularization для diagonal mass: shrinkage к единице."""
    n = count.astype(jnp.float32)
    shrinkage_weight = 5.0 / (n + 5.0)
    scaled = variance * n / (n + 5.0)
    regularized = scaled + shrinkage_weight
    return 1.0 / jnp.maximum(regularized, 1e-6)


def _regularize_dense(
    cov: jnp.ndarray, count: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stan-style regularization для dense mass: shrinkage + Cholesky.

    Σ_reg = n/(n+5) * Σ + 5/(n+5) * 1e-3 * I
    L = cholesky(Σ_reg)          — mass_chol для momentum sampling
    inv_mass = L_inv^T @ L_inv   — precision для leapfrog/kinetic energy

    Returns: (inv_mass_dense, mass_chol)
    """
    dim = cov.shape[0]
    n = count.astype(jnp.float32)
    scaled = cov * n / (n + 5.0)
    shrinkage = 1e-3 * 5.0 / (n + 5.0) * jnp.eye(dim)
    regularized = scaled + shrinkage
    regularized = 0.5 * (regularized + regularized.T)

    L = jnp.linalg.cholesky(regularized)
    L_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(dim), lower=True)
    inv_mass = L_inv.T @ L_inv
    return inv_mass, L


def finalize_mass(warmup: WarmupState) -> WarmupState:
    """Финализация mass matrix в конце adaptation window.

    Вычисляем variance/covariance из Welford, регуляризуем,
    обновляем inv_mass_diag, inv_mass_dense, mass_chol, сбрасываем Welford.
    """
    has_data = warmup.welford_count > 2
    dim = warmup.welford_mean.shape[0]

    variance = jnp.where(
        has_data,
        warmup.welford_m2 / (warmup.welford_count - 1),
        jnp.ones(dim),
    )
    inv_mass_diag = jnp.where(
        has_data,
        _regularize_diagonal(variance, warmup.welford_count),
        jnp.ones(dim),
    )

    cov = jnp.where(
        has_data,
        warmup.welford_cov / (warmup.welford_count - 1),
        jnp.eye(dim),
    )
    inv_mass_dense_new, mass_chol_new = _regularize_dense(cov, warmup.welford_count)
    inv_mass_dense = jnp.where(has_data, inv_mass_dense_new, jnp.eye(dim))
    mass_chol = jnp.where(has_data, mass_chol_new, jnp.eye(dim))

    return warmup._replace(
        inv_mass_diag=inv_mass_diag,
        inv_mass_dense=inv_mass_dense,
        mass_chol=mass_chol,
        welford_count=jnp.zeros((), dtype=jnp.int32),
        welford_mean=jnp.zeros(dim),
        welford_m2=jnp.zeros(dim),
        welford_cov=jnp.zeros((dim, dim)),
    )


def find_reasonable_step_size(
    potential_and_grad_fn: Callable,
    position: jnp.ndarray,
    rng_key: jnp.ndarray,
    inv_mass: jnp.ndarray,
    mass_chol: jnp.ndarray | None = None,
    target_accept: float = 0.5,
) -> jnp.ndarray:
    """Найти начальный step_size через бисекцию по acceptance rate.

    Делаем один leapfrog шаг с разными step_size, ищем тот
    при котором acceptance ≈ target. Стартуем с 1.0, удваиваем/делим.
    Поддерживает и diagonal, и dense mass matrix.
    """
    step_size = jnp.array(1.0)
    dim = position.shape[0]
    lp, grad = potential_and_grad_fn(position)
    grad = jnp.where(jnp.isfinite(grad), grad, jnp.zeros_like(grad))

    if mass_chol is None:
        mass_chol = jnp.eye(dim)
    r = sample_momentum(rng_key, inv_mass, mass_chol)
    ke_old = kinetic_energy(inv_mass, r)

    def _try_step(eps):
        r_ = r + 0.5 * eps * grad
        z_ = position + eps * apply_inv_mass(inv_mass, r_)
        lp_, g_ = potential_and_grad_fn(z_)
        g_ = jnp.where(jnp.isfinite(g_), g_, jnp.zeros_like(g_))
        r_ = r_ + 0.5 * eps * g_
        ke_new = kinetic_energy(inv_mass, r_)
        log_alpha = lp_ - lp + ke_old - ke_new
        return jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)

    log_alpha = _try_step(step_size)
    direction = jnp.where(log_alpha > jnp.log(target_accept), 1.0, -1.0)

    def cond_fn(carry):
        eps, log_a, n = carry
        too_high = (direction > 0) & (log_a > jnp.log(target_accept))
        too_low = (direction < 0) & (log_a < jnp.log(target_accept))
        return (too_high | too_low) & (n < 100) & jnp.isfinite(log_a)

    def body_fn(carry):
        eps, _, n = carry
        eps = eps * jnp.where(direction > 0, 2.0, 0.5)
        eps = jnp.clip(eps, 1e-10, 1e2)
        return (eps, _try_step(eps), n + 1)

    step_size, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (step_size, log_alpha, jnp.int32(0))
    )

    return jnp.clip(step_size, 1e-10, 1e2)


def build_adaptation_schedule(num_warmup: int) -> list[tuple[int, int]]:
    """Stan-style adaptation schedule: initial buffer + windows + final buffer.

    Returns list of (window_start, window_end) for mass adaptation windows.
    Между окнами — только dual averaging. В окне — Welford + dual averaging.
    В конце окна — finalize_mass.
    """
    init_buffer = 75
    final_buffer = 50

    if num_warmup < init_buffer + final_buffer + 25:
        return [(0, num_warmup)]

    adapt_start = init_buffer
    adapt_end = num_warmup - final_buffer

    windows = []
    window_size = 25
    pos = adapt_start
    while pos < adapt_end:
        end = min(pos + window_size, adapt_end)
        windows.append((pos, end))
        pos = end
        window_size = min(window_size * 2, adapt_end - pos) if pos < adapt_end else window_size

    return windows
