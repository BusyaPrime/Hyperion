"""Хендлер трассировки — перехватывает вызовы DSL-примитивов и записывает их.

Когда модель выполняется под trace_model(), этот хендлер
ловит каждый sample/plate/param/factor и складывает в Trace.
По сути — это прокси между DSL и данными трассировки.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

import jax
import jax.numpy as jnp


class TraceHandler:
    """Хендлер, который перехватывает вызовы DSL-примитивов и пишет их в trace.

    Не трогай руками — работает через глобальный контекст.
    """

    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        substitutions: Optional[dict[str, Any]] = None,
    ):
        self._rng_key = rng_key
        self._substitutions = substitutions or {}
        self._plate_stack: list[tuple[str, int, int]] = []

        from hyperion_trace.trace import Trace
        self.trace = Trace()

    def _next_key(self) -> jax.random.PRNGKey:
        """Дёргаем следующий subkey из PRNG — классический split."""
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    def _current_plates(self) -> list[str]:
        """Список имён активных пластин (для контекста)."""
        return [name for name, _, _ in self._plate_stack]

    def process_sample(
        self,
        name: str,
        dist: Any,
        obs: Optional[Any] = None,
        constraint: Optional[Any] = None,
    ) -> Any:
        from hyperion_trace.trace import TraceEntry, NodeType

        is_observed = obs is not None or name in self._substitutions
        if name in self._substitutions:
            value = jnp.asarray(self._substitutions[name], dtype=jnp.float32)
        elif obs is not None:
            value = jnp.asarray(obs, dtype=jnp.float32)
        else:
            key = self._next_key()
            value = dist.sample(key)

        log_p = dist.log_prob(value)

        entry = TraceEntry(
            name=name,
            node_type=NodeType.SAMPLE,
            distribution=dist,
            value=value,
            observed=is_observed,
            obs_value=obs if obs is not None else (value if name in self._substitutions else None),
            constraint=constraint,
            log_prob=log_p,
            plates=self._current_plates(),
        )
        self.trace.add_entry(entry)
        return value

    @contextmanager
    def process_plate(self, name: str, size: int, dim: int = -1):
        """Контекст-менеджер для plate: пушим в стек, юзаем, попим."""
        self._plate_stack.append((name, size, dim))
        self.trace.plate_stack.append((name, size, dim))
        try:
            yield jnp.arange(size)
        finally:
            self._plate_stack.pop()
            self.trace.plate_stack.pop()

    def process_param(
        self,
        name: str,
        init_value: Any,
        constraint: Optional[Any] = None,
    ) -> Any:
        from hyperion_trace.trace import TraceEntry, NodeType

        if name in self._substitutions:
            value = jnp.asarray(self._substitutions[name], dtype=jnp.float32)
        else:
            value = jnp.asarray(init_value, dtype=jnp.float32) if init_value is not None else None

        entry = TraceEntry(
            name=name,
            node_type=NodeType.PARAM,
            value=value,
            constraint=constraint,
            plates=self._current_plates(),
        )
        self.trace.add_entry(entry)
        return value

    def process_deterministic(self, name: str, value: Any) -> Any:
        """Детерминистическая нода — просто прокидываем значение в trace."""
        from hyperion_trace.trace import TraceEntry, NodeType

        entry = TraceEntry(
            name=name,
            node_type=NodeType.DETERMINISTIC,
            value=value,
            plates=self._current_plates(),
        )
        self.trace.add_entry(entry)
        return value

    def process_factor(self, name: str, log_weight: Any) -> None:
        """Фактор — добавляем log-вес в joint (например для soft constraints)."""
        from hyperion_trace.trace import TraceEntry, NodeType

        entry = TraceEntry(
            name=name,
            node_type=NodeType.FACTOR,
            value=log_weight,
            plates=self._current_plates(),
        )
        self.trace.add_entry(entry)
