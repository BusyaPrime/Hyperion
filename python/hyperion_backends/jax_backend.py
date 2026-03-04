"""JAX-бэкенд для HYPERION — JIT-компилированный до предела.

Модель трассируется ОДИН РАЗ при initialize(). После этого
potential_fn и potential_and_grad_fn — чистые JIT-скомпилированные
функции, которые работают со скоростью XLA. Никаких Python-объектов
в горячем цикле, никаких dict comprehensions на каждый leapfrog шаг.

Это сердце всей системы: если бэкенд тормозит, тормозит всё.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from hyperion_backends.base import Backend
from hyperion_trace.trace import trace_model, Trace
from hyperion_dsl.transforms import biject_to

logger = logging.getLogger(__name__)


class JAXBackend(Backend):
    """JAX-бэкенд с JIT-скомпилированным potential.

    initialize() трассирует модель один раз, вытаскивает структуру латентов,
    строит биекции constrained↔unconstrained и компилирует potential_fn через jax.jit.
    После этого инференс-движки работают только с potential_fn/potential_and_grad_fn —
    никаких trace_model в горячем цикле. Вся тяжёлая работа на XLA.
    """

    def __init__(self):
        self._model_fn: Any = None
        self._data: dict[str, jnp.ndarray] = {}
        self._rng_key: Any = None
        self._trace: Optional[Trace] = None
        self._latent_shapes: dict[str, tuple[int, ...]] = {}
        self._unconstrained_shapes: dict[str, tuple[int, ...]] = {}
        self._latent_names: list[str] = []
        self._transforms: dict[str, Any] = {}
        self._flat_sizes: dict[str, int] = {}
        self._total_dim: int = 0
        self._initialized: bool = False

        # JIT-скомпилированные функции — заполняются в initialize()
        self._jit_potential: Optional[Callable] = None
        self._jit_potential_and_grad: Optional[Callable] = None
        self._jit_grad: Optional[Callable] = None

        # IR (альтернативный путь) — для оптимизации/анализа
        self._ir_graph = None
        self._ir_potential = None

    def initialize(
        self,
        model_fn: Any,
        data: dict[str, jnp.ndarray],
        rng_key: Any,
    ) -> None:
        """Инициализация: трассим модель один раз, компилим potential.

        Трассировка вытаскивает структуру латентов, биекции constrained↔unconstrained
        строятся автоматически. potential_fn и potential_and_grad_fn JIT-компилируются —
        после этого вызова всё работает за O(XLA), а не O(Python).

        Args:
            model_fn: Функция модели (def model(): ...).
            data: Словарь observed-данных для подстановки.
            rng_key: JAX PRNG key для трассировки.
        """
        self._model_fn = model_fn
        self._data = data
        self._rng_key = rng_key

        if model_fn is None:
            raise ValueError(
                "model_fn не может быть None. Передай реальную модель, не заглушку."
            )

        # === Фаза 1: Трассировка — один раз, чтобы узнать структуру ===
        self._trace = trace_model(
            model_fn, rng_key=rng_key, substitutions=data,
        )

        self._latent_names = self._trace.latent_names
        if not self._latent_names:
            raise ValueError(
                "Модель без латентных переменных. Нечего инферить. "
                "Проверь, что есть sample() без obs."
            )

        for name in self._latent_names:
            entry = self._trace.entries[name]
            shape = ()
            if entry.value is not None and hasattr(entry.value, "shape"):
                shape = tuple(entry.value.shape)
            self._latent_shapes[name] = shape

        # === Фаза 2: Построение биекций ===
        for name in self._latent_names:
            entry = self._trace.entries[name]
            if entry.constraint is not None:
                try:
                    self._transforms[name] = biject_to(entry.constraint)
                except (ValueError, TypeError) as e:
                    logger.debug("biject_to failed for %s (constraint): %s", name, e)
            elif entry.distribution is not None:
                support = getattr(entry.distribution, "support", None)
                if support is not None:
                    try:
                        self._transforms[name] = biject_to(support)
                    except (ValueError, TypeError) as e:
                        logger.debug("biject_to failed for %s (support): %s", name, e)

        # Compute unconstrained shapes and flat_sizes.
        # Transforms like StickBreaking (K→K-1) and Cholesky (n×n→n(n+1)/2)
        # change dimensionality. We probe the inverse to discover unconstrained shape.
        for name in self._latent_names:
            constrained_shape = self._latent_shapes[name]
            if name in self._transforms:
                dummy = jnp.zeros(constrained_shape)
                try:
                    unconstrained = self._transforms[name].inverse(dummy)
                    unc_shape = tuple(unconstrained.shape)
                    self._unconstrained_shapes[name] = unc_shape
                    self._flat_sizes[name] = max(1, unconstrained.size)
                except Exception as e:
                    logger.debug("inverse probe failed for %s: %s", name, e)
                    self._unconstrained_shapes[name] = constrained_shape
                    size = 1
                    for s in constrained_shape:
                        size *= s
                    self._flat_sizes[name] = size
            else:
                self._unconstrained_shapes[name] = constrained_shape
                size = 1
                for s in constrained_shape:
                    size *= s
                self._flat_sizes[name] = size

        self._total_dim = sum(self._flat_sizes.values())

        # === Фаза 3: Компиляция potential_fn ===
        # Замыкание захватывает всю структуру модели. JAX JIT скомпилирует
        # числовую часть в XLA, Python-обёртка отработает при первом вызове (tracing).
        latent_names = list(self._latent_names)
        flat_sizes = dict(self._flat_sizes)
        unconstrained_shapes = dict(self._unconstrained_shapes)
        transforms = dict(self._transforms)
        model_fn_ref = self._model_fn
        data_ref = dict(self._data)
        rng_key_ref = self._rng_key

        def _raw_potential(flat_params: jnp.ndarray) -> jnp.ndarray:
            """Чистая функция: R^d → scalar log p(data, z).

            Внутри: unflatten (unconstrained shape) → forward transform → trace → sum log_prob.
            JAX видит только jnp-операции и компилит в XLA.
            """
            params = {}
            offset = 0
            for name in latent_names:
                size = flat_sizes[name]
                unc_shape = unconstrained_shapes[name]
                params[name] = flat_params[offset:offset + size].reshape(unc_shape)
                offset += size

            constrained = {}
            log_det = jnp.zeros(())
            for name, value in params.items():
                if name in transforms:
                    t = transforms[name]
                    c = t.forward(value)
                    ld = t.log_abs_det_jacobian(value, c)
                    if ld.ndim > 0:
                        ld = jnp.sum(ld)
                    log_det = log_det + ld
                    constrained[name] = c
                else:
                    constrained[name] = value

            subs = {**data_ref, **constrained}
            trace = trace_model(model_fn_ref, rng_key=rng_key_ref, substitutions=subs)
            return trace.log_joint() + log_det

        # JIT-компилируем. Первый вызов — трейсинг (медленно).
        # Все последующие — чистый XLA (быстро).
        self._jit_potential = jax.jit(_raw_potential)
        self._jit_potential_and_grad = jax.jit(jax.value_and_grad(_raw_potential))
        self._jit_grad = jax.jit(jax.grad(_raw_potential))

        # Прогреваем JIT — первый вызов компилирует XLA граф
        try:
            dummy = self.flatten_latents(self.sample_prior(rng_key))
            _ = self._jit_potential(dummy)
        except Exception as e:
            logger.debug("JIT warmup skipped: %s", e)

        # === Phase 4: IR compilation (optional, for optimization/analysis) ===
        try:
            from hyperion_ir.compiler import ModelCompiler
            compiler = ModelCompiler(optimize=True)
            self._ir_graph = compiler.compile(
                model_fn, rng_key=rng_key, substitutions=data,
            )
        except Exception as e:
            logger.debug("IR compilation skipped: %s", e)
            self._ir_graph = None

        # === Phase 5: IR-based potential (alternative path) ===
        self._ir_potential = self._build_ir_potential()

        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "Бэкенд не инициализирован. Вызови initialize() перед использованием."
            )

    @property
    def ir_graph(self):
        """Скомпилированный IR граф модели. None если компиляция не удалась."""
        return self._ir_graph

    @property
    def ir_potential_fn(self):
        """IR-based potential_fn (альтернатива trace_model). None если IR недоступен."""
        return self._ir_potential

    def _build_ir_potential(self):
        """Построить potential_fn из IR вместо trace_model.

        IR-based potential не вызывает trace_model на каждой итерации,
        а использует предкомпилированные log_prob_fn из IRNode.
        Работает только для моделей без Python control flow в log_prob.
        """
        if self._ir_graph is None:
            return None

        ir = self._ir_graph
        if ir._has_dynamic_observed:
            return None
        latent_names = list(self._latent_names)
        flat_sizes = dict(self._flat_sizes)
        unconstrained_shapes = dict(self._unconstrained_shapes)
        transforms = dict(self._transforms)
        observed_values = {}

        for name in ir.observed_names:
            if name in self._data:
                observed_values[name] = jnp.asarray(self._data[name], dtype=jnp.float32)

        def _ir_potential(flat_params):
            params = {}
            offset = 0
            for name in latent_names:
                size = flat_sizes[name]
                unc_shape = unconstrained_shapes[name]
                params[name] = flat_params[offset:offset + size].reshape(unc_shape)
                offset += size

            constrained = {}
            log_det = jnp.zeros(())
            for name, value in params.items():
                if name in transforms:
                    t = transforms[name]
                    c = t.forward(value)
                    ld = t.log_abs_det_jacobian(value, c)
                    if ld.ndim > 0:
                        ld = jnp.sum(ld)
                    log_det = log_det + ld
                    constrained[name] = c
                else:
                    constrained[name] = value

            return ir.compute_log_joint(constrained, observed_values) + log_det

        return jax.jit(_ir_potential)

    @property
    def potential_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """JIT-скомпилированная log p(data, z). Основной интерфейс для HMC/NUTS.

        Принимает flat unconstrained вектор, возвращает скаляр. Никакого trace_model
        в горячем цикле — только XLA.
        """
        self._ensure_initialized()
        return self._jit_potential

    @property
    def potential_and_grad_fn(self) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
        """JIT-скомпилированная (log_p, grad). Для HMC/NUTS — одним вызовом.

        value_and_grad под капотом: один проход по графу вместо двух.
        """
        self._ensure_initialized()
        return self._jit_potential_and_grad

    @property
    def grad_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """JIT-скомпилированный градиент."""
        self._ensure_initialized()
        return self._jit_grad

    @property
    def total_dim(self) -> int:
        """Размерность unconstrained пространства — суммарное число скаляров в flat-векторе."""
        return self._total_dim

    def get_latent_shapes(self) -> dict[str, tuple[int, ...]]:
        """Форма каждого латента в constrained пространстве. Для reshape и отладки."""
        return dict(self._latent_shapes)

    def sample_prior(self, rng_key: Any) -> dict[str, jnp.ndarray]:
        """Семплим из приора через трассировку с data-only substitutions."""
        trace = trace_model(
            self._model_fn, rng_key=rng_key, substitutions=self._data,
        )
        return {name: trace.entries[name].value for name in trace.latent_names}

    def flatten_latents(self, latent_values: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """dict (constrained) → flat vector (unconstrained).

        sample_prior отдаёт constrained значения (положительные для HalfNormal,
        simplex для Dirichlet). inverse transform переводит в unconstrained —
        размерность может поменяться (simplex K → R^{K-1}, Cholesky n×n → n(n+1)/2).

        Args:
            latent_values: Словарь латентов в constrained пространстве.

        Returns:
            Одномерный вектор длины total_dim.
        """
        parts = []
        for name in self._latent_names:
            v = latent_values.get(name)
            if v is None:
                raise KeyError(f"Латент '{name}' отсутствует. Ожидались: {self._latent_names}")
            v = jnp.asarray(v)
            if name in self._transforms:
                v = self._transforms[name].inverse(v)
            parts.append(v.ravel())
        return jnp.concatenate(parts) if parts else jnp.array([])

    def unflatten_latents(self, flat: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """flat vector (unconstrained) → dict (constrained).

        Берёт сэмплы из HMC/NUTS (unconstrained), решейпит по unconstrained shapes,
        применяет forward transform. Размерность может поменяться (R^{K-1} → simplex K).

        Args:
            flat: Вектор длины total_dim.

        Returns:
            Словарь латентов в constrained пространстве.
        """
        result = {}
        offset = 0
        for name in self._latent_names:
            size = self._flat_sizes[name]
            unc_shape = self._unconstrained_shapes[name]
            val = flat[offset:offset + size].reshape(unc_shape)
            if name in self._transforms:
                val = self._transforms[name].forward(val)
            result[name] = val
            offset += size
        return result

    # === Backward-compatible API (для старого кода) ===

    def log_joint_flat(self, flat: jnp.ndarray) -> jnp.ndarray:
        """Обратная совместимость: log_joint от flat вектора."""
        self._ensure_initialized()
        return self._jit_potential(flat)

    def grad_log_joint_flat(self, flat: jnp.ndarray) -> jnp.ndarray:
        """Обратная совместимость: градиент от flat вектора."""
        self._ensure_initialized()
        return self._jit_grad(flat)

    def log_joint_and_grad_flat(self, flat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Обратная совместимость: (value, grad) от flat вектора."""
        self._ensure_initialized()
        return self._jit_potential_and_grad(flat)
