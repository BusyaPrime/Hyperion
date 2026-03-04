"""Normalizing Flows для HYPERION.

Цепочка обратимых трансформаций. Превращаем простой гауссиан в что-то сложное.
RealNVP/MAF — масштаб и сдвиг зависят от части входа. Jacobian — диагональный, детерминант легко.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from hyperion_inference.base import InferenceEngine, InferenceState, InferenceResult

logger = logging.getLogger(__name__)


# --- Flow layers ---

def _xavier_init(key: jax.random.PRNGKey, shape: tuple, scale: float = 1.0) -> jnp.ndarray:
    """Xavier/Glorot: std = sqrt(2/(fan_in+fan_out)). Для tanh — норм, градиенты не взрываются."""
    fan_in, fan_out = shape[0], shape[1]
    std = jnp.sqrt(2.0 / (fan_in + fan_out)) * scale
    return jrandom.normal(key, shape) * std


class ActNormLayer:
    """ActNorm — scale+shift с обучаемыми параметрами. Обратимый везде, без batch stats."""

    def __init__(self, dim: int):
        self.dim = dim
        self.params = {
            "log_scale": jnp.zeros(dim),
            "bias": jnp.zeros(dim),
        }

    def forward(self, x: jnp.ndarray, params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        """y = x * exp(log_scale) + bias. Log-det = sum(log_scale)."""
        scale = jnp.exp(params["log_scale"])
        y = x * scale + params["bias"]
        log_det = jnp.sum(params["log_scale"])
        log_det = jnp.broadcast_to(log_det, x.shape[:-1])
        return y, log_det

    def inverse(self, y: jnp.ndarray, params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        """x = (y - bias) / scale."""
        scale = jnp.exp(params["log_scale"])
        x = (y - params["bias"]) / scale
        log_det = -jnp.sum(params["log_scale"])
        log_det = jnp.broadcast_to(log_det, y.shape[:-1])
        return x, log_det


class AffineCouplingLayer:
    """RealNVP-style affine coupling: y_d = x_d * exp(s(x_m)) + t(x_m). Маска делит вход пополам."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        mask: jnp.ndarray,
        key: jax.random.PRNGKey,
        use_xavier: bool = True,
    ):
        self.dim = dim
        self.mask = mask
        k1, k2, k3, k4 = jrandom.split(key, 4)
        scale = 0.01 if not use_xavier else 1.0

        def init_w(k, shp):
            return _xavier_init(k, shp, scale) if use_xavier else jrandom.normal(k, shp) * 0.01

        self.params = {
            "w1": init_w(k1, (dim, hidden_dim)),
            "b1": jnp.zeros(hidden_dim),
            "w2_s": init_w(k2, (hidden_dim, dim)),
            "b2_s": jnp.zeros(dim),
            "w2_t": init_w(k3, (hidden_dim, dim)),
            "b2_t": jnp.zeros(dim),
        }

    def _net(self, x: jnp.ndarray, params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = jnp.tanh(x @ params["w1"] + params["b1"])
        log_s = jnp.tanh(h @ params["w2_s"] + params["b2_s"])
        t = h @ params["w2_t"] + params["b2_t"]
        return log_s, t

    def forward(self, x: jnp.ndarray, params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        x_masked = x * self.mask
        log_s, t = self._net(x_masked, params)
        log_s = log_s * (1 - self.mask)
        t = t * (1 - self.mask)
        y = x * jnp.exp(log_s) + t * (1 - self.mask)
        y = y * (1 - self.mask) + x_masked
        log_det = jnp.sum(log_s, axis=-1)
        return y, log_det

    def inverse(self, y: jnp.ndarray, params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        y_masked = y * self.mask
        log_s, t = self._net(y_masked, params)
        log_s = log_s * (1 - self.mask)
        t = t * (1 - self.mask)
        x_unmasked = (y - t * (1 - self.mask)) * jnp.exp(-log_s) * (1 - self.mask)
        x = x_unmasked + y_masked
        log_det = -jnp.sum(log_s, axis=-1)
        return x, log_det


class MaskedAutoregressiveLayer:
    """MAF: масштаб и сдвиг для x_i зависят только от x_{1:i-1}. Авторегрессия = треугольный Jacobian."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        key: jax.random.PRNGKey,
        use_xavier: bool = True,
    ):
        self.dim = dim
        scale = 0.01 if not use_xavier else 1.0

        def init_w(k, shp):
            return _xavier_init(k, shp, scale) if use_xavier else jrandom.normal(k, shp) * 0.01

        k1, k2, k3, k4 = jrandom.split(key, 4)

        self._input_order = jnp.arange(dim)
        hidden_order = jnp.arange(hidden_dim) % (dim - 1)
        self._mask_in = (self._input_order[None, :] <= hidden_order[:, None]).astype(jnp.float32).T
        self._mask_out = (hidden_order[None, :] < self._input_order[:, None]).astype(jnp.float32).T

        self.params = {
            "w1": init_w(k1, (dim, hidden_dim)),
            "b1": jnp.zeros(hidden_dim),
            "w2_s": init_w(k2, (hidden_dim, dim)),
            "b2_s": jnp.zeros(dim),
            "w2_t": init_w(k3, (hidden_dim, dim)),
            "b2_t": jnp.zeros(dim),
        }

    def forward(self, x: jnp.ndarray, params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = jnp.tanh(x @ (params["w1"] * self._mask_in) + params["b1"])
        log_s = jnp.tanh(h @ (params["w2_s"] * self._mask_out) + params["b2_s"])
        t = h @ (params["w2_t"] * self._mask_out) + params["b2_t"]
        y = x * jnp.exp(log_s) + t
        log_det = jnp.sum(log_s, axis=-1)
        return y, log_det

    def inverse(self, y: jnp.ndarray, params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.zeros_like(y)
        for i in range(self.dim):
            h = jnp.tanh(x @ (params["w1"] * self._mask_in) + params["b1"])
            log_s = jnp.tanh(h @ (params["w2_s"] * self._mask_out) + params["b2_s"])
            t = h @ (params["w2_t"] * self._mask_out) + params["b2_t"]
            x = x.at[..., i].set((y[..., i] - t[..., i]) * jnp.exp(-log_s[..., i]))
        log_det = -jnp.sum(log_s, axis=-1)
        return x, log_det


class NormalizingFlow:
    """Композиция flow-слоёв. Каждый слой — биекция, цепочка — тоже биекция. Log-det суммируется."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 6,
        hidden_dim: int = 64,
        flow_type: str = "realnvp",
        key: jax.random.PRNGKey = jrandom.PRNGKey(0),
        use_batch_norm: bool = False,
        use_xavier: bool = True,
    ):
        self.dim = dim
        self.num_layers = num_layers
        self.layers = []
        self.all_params: list[dict] = []

        keys = jrandom.split(key, num_layers + (1 if use_batch_norm else 0))
        for i in range(num_layers):
            if use_batch_norm and i > 0:
                act = ActNormLayer(dim)
                self.layers.append(act)
                self.all_params.append(act.params)

            if flow_type == "realnvp":
                mask = jnp.array([j % 2 == (i % 2) for j in range(dim)], dtype=jnp.float32)
                layer = AffineCouplingLayer(dim, hidden_dim, mask, keys[i], use_xavier)
            else:
                layer = MaskedAutoregressiveLayer(dim, hidden_dim, keys[i], use_xavier)
            self.layers.append(layer)
            self.all_params.append(layer.params)

    def forward(
        self, z: jnp.ndarray, all_params: list[dict]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        log_det_total = jnp.zeros(z.shape[:-1])
        x = z
        param_idx = 0
        for layer in self.layers:
            params = all_params[param_idx]
            x, log_det = layer.forward(x, params)
            log_det_total = log_det_total + log_det
            param_idx += 1
        return x, log_det_total

    def inverse(
        self, x: jnp.ndarray, all_params: list[dict]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        log_det_total = jnp.zeros(x.shape[:-1])
        z = x
        param_idx = len(self.layers) - 1
        for layer in reversed(self.layers):
            params = all_params[param_idx]
            z, log_det = layer.inverse(z, params)
            log_det_total = log_det_total + log_det
            param_idx -= 1
        return z, log_det_total

    def log_prob(
        self, x: jnp.ndarray, all_params: list[dict]
    ) -> jnp.ndarray:
        """log p(x) = log p(z) - log|det dg/dz|, где z = g^{-1}(x)."""
        z, log_det = self.inverse(x, all_params)
        log_pz = -0.5 * jnp.sum(z ** 2, axis=-1) - 0.5 * self.dim * jnp.log(2.0 * jnp.pi)
        return log_pz + log_det


@dataclass
class FlowsConfig:
    """Конфиг flows: слои, hidden, тип (realnvp/maf), batch norm, Xavier."""
    num_steps: int = 5000
    learning_rate: float = 1e-3
    num_layers: int = 6
    hidden_dim: int = 64
    flow_type: str = "realnvp"
    num_elbo_samples: int = 20
    num_posterior_samples: int = 1000
    clip_grad_norm: float = 10.0
    use_batch_norm: bool = False
    use_xavier_init: bool = True


class FlowsEngine(InferenceEngine):
    """Flows как variational approximation: q = flow(prior). Обучаем ELBO."""

    def __init__(self):
        self._backend = None
        self._flow = None
        self._optimizer = None

    def initialize(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> InferenceState:
        self._backend = backend
        cfg = FlowsConfig(**{k: v for k, v in config.items() if hasattr(FlowsConfig, k)})

        dim = backend.total_dim
        flow_key, rng_key = jrandom.split(rng_key)

        self._flow = NormalizingFlow(
            dim=dim,
            num_layers=cfg.num_layers,
            hidden_dim=cfg.hidden_dim,
            flow_type=cfg.flow_type,
            key=flow_key,
            use_batch_norm=cfg.use_batch_norm,
            use_xavier=cfg.use_xavier_init,
        )

        self._optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.clip_grad_norm),
            optax.adam(cfg.learning_rate),
        )

        params = self._flow.all_params
        self._opt_state = self._optimizer.init(params)

        return InferenceState(
            step=0,
            rng_key=rng_key,
        )

    def _compute_elbo(
        self,
        key: jax.random.PRNGKey,
        flow_params: list[dict],
        num_samples: int,
    ) -> jnp.ndarray:
        """ELBO: семплим z~N(0,I), прогоняем через flow. vmap по сэмплам — батчим potential_fn."""
        dim = self._flow.dim
        eps = jrandom.normal(key, shape=(num_samples, dim))

        z_samples, log_det = self._flow.forward(eps, flow_params)

        log_pz = -0.5 * jnp.sum(eps ** 2, axis=-1) - 0.5 * dim * jnp.log(2.0 * jnp.pi)
        log_qz = log_pz - log_det

        # vmap: potential_fn по батчу — JIT-friendly, без Python loop
        log_joints = jax.vmap(self._backend.potential_fn)(z_samples)

        return jnp.mean(log_joints - log_qz)

    def step(self, state: InferenceState) -> InferenceState:
        key, elbo_key = jrandom.split(state.rng_key)

        def neg_elbo(params):
            return -self._compute_elbo(elbo_key, params, 20)

        loss, grads = jax.value_and_grad(neg_elbo)(self._flow.all_params)
        updates, new_opt_state = self._optimizer.update(
            grads, self._opt_state, self._flow.all_params
        )
        self._flow.all_params = optax.apply_updates(self._flow.all_params, updates)
        self._opt_state = new_opt_state

        return InferenceState(step=state.step + 1, rng_key=key)

    def get_samples(self, state: InferenceState) -> dict[str, jnp.ndarray]:
        dim = self._flow.dim
        key, _ = jrandom.split(state.rng_key)
        eps = jrandom.normal(key, shape=(1000, dim))
        z_samples, _ = self._flow.forward(eps, self._flow.all_params)

        latent_names = list(self._backend.get_latent_shapes().keys())
        unflat_batched = jax.vmap(self._backend.unflatten_latents)(z_samples)
        return {name: unflat_batched[name] for name in latent_names}

    def get_metrics(self, state: InferenceState) -> dict[str, Any]:
        return {"num_steps": state.step}

    def run(
        self,
        backend: Any,
        rng_key: jax.random.PRNGKey,
        config: dict[str, Any],
    ) -> InferenceResult:
        cfg = FlowsConfig(**{k: v for k, v in config.items() if hasattr(FlowsConfig, k)})

        dim = backend.total_dim
        self._backend = backend
        flow_key, rng_key = jrandom.split(rng_key)

        self._flow = NormalizingFlow(
            dim=dim,
            num_layers=cfg.num_layers,
            hidden_dim=cfg.hidden_dim,
            flow_type=cfg.flow_type,
            key=flow_key,
            use_batch_norm=cfg.use_batch_norm,
            use_xavier=cfg.use_xavier_init,
        )

        self._optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.clip_grad_norm),
            optax.adam(cfg.learning_rate),
        )
        self._opt_state = self._optimizer.init(self._flow.all_params)

        logger.info("Flows started: %d steps, %d layers, dim=%d, type=%s",
                     cfg.num_steps, cfg.num_layers, dim, cfg.flow_type)

        elbo_history = []
        for i in range(cfg.num_steps):
            key, elbo_key = jrandom.split(rng_key)
            rng_key = key

            def neg_elbo(params):
                return -self._compute_elbo(elbo_key, params, cfg.num_elbo_samples)

            loss, grads = jax.value_and_grad(neg_elbo)(self._flow.all_params)
            updates, self._opt_state = self._optimizer.update(
                grads, self._opt_state, self._flow.all_params
            )
            self._flow.all_params = optax.apply_updates(self._flow.all_params, updates)
            elbo_history.append(-float(loss))
            if i > 0 and i % 500 == 0:
                logger.debug("Flows step %d/%d, ELBO=%.4f", i, cfg.num_steps, elbo_history[-1])

        logger.info("Flows finished: final ELBO=%.4f", elbo_history[-1] if elbo_history else float("nan"))

        # Генерируем постериорные сэмплы
        sample_key, _ = jrandom.split(rng_key)
        eps = jrandom.normal(sample_key, shape=(cfg.num_posterior_samples, dim))
        z_samples, _ = self._flow.forward(eps, self._flow.all_params)

        latent_names = list(backend.get_latent_shapes().keys())
        unflat_batched = jax.vmap(backend.unflatten_latents)(z_samples)
        samples = {name: unflat_batched[name] for name in latent_names}

        return InferenceResult(
            samples=samples,
            diagnostics={"elbo_history": elbo_history},
        )
