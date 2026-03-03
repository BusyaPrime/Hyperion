"""Структуры данных трассировки и трассировка модели.

Парсим выполнение модели и складываем всё в Trace —
потом по нему строим граф, IR и прочие штуки.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import jax
import jax.numpy as jnp

from hyperion_trace.handler import TraceHandler
from hyperion_dsl.primitives import _push_trace_handler, _pop_trace_handler


class NodeType(Enum):
    """Типы нод в trace: sample, param, deterministic, factor."""
    SAMPLE = auto()
    PARAM = auto()
    DETERMINISTIC = auto()
    FACTOR = auto()


@dataclass
class TraceEntry:
    """Одна запись в execution trace — sample, param, factor и т.д."""

    name: str
    node_type: NodeType
    distribution: Optional[Any] = None
    value: Optional[Any] = None
    observed: bool = False
    obs_value: Optional[Any] = None
    constraint: Optional[Any] = None
    log_prob: Optional[jnp.ndarray] = None
    plates: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)


@dataclass
class Trace:
    """Полный execution trace вероятностной модели.

    Содержит все sample/param/factor в порядке выполнения.
    Отсюда строим граф зависимостей и IR.
    """

    entries: dict[str, TraceEntry] = field(default_factory=dict)
    plate_stack: list[tuple[str, int, int]] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)

    def add_entry(self, entry: TraceEntry) -> None:
        if entry.name in self.entries:
            raise ValueError(f"Дубликат записи в trace: {entry.name}")
        self.entries[entry.name] = entry
        self.execution_order.append(entry.name)

    def __getitem__(self, name: str) -> TraceEntry:
        return self.entries[name]

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.execution_order)

    @property
    def latent_names(self) -> list[str]:
        """Имена латентных переменных (sample без obs)."""
        return [
            name for name, entry in self.entries.items()
            if entry.node_type == NodeType.SAMPLE and not entry.observed
        ]

    @property
    def observed_names(self) -> list[str]:
        """Имена observed переменных."""
        return [
            name for name, entry in self.entries.items()
            if entry.node_type == NodeType.SAMPLE and entry.observed
        ]

    @property
    def param_names(self) -> list[str]:
        """Имена параметров."""
        return [
            name for name, entry in self.entries.items()
            if entry.node_type == NodeType.PARAM
        ]

    def log_joint(self) -> jnp.ndarray:
        """Суммируем log_prob по всем sample + value по factor."""
        total = jnp.float32(0.0)
        for name in self.execution_order:
            entry = self.entries[name]
            if entry.log_prob is not None:
                lp = entry.log_prob
                if lp.ndim > 0:
                    lp = jnp.sum(lp)
                total = total + lp
            elif entry.node_type == NodeType.FACTOR and entry.value is not None:
                v = entry.value
                if hasattr(v, 'ndim') and v.ndim > 0:
                    v = jnp.sum(v)
                total = total + v
        return total

    def summary(self) -> dict[str, Any]:
        """Краткая сводка по trace — для дебага и логирования."""
        return {
            "num_entries": len(self.entries),
            "latent_vars": self.latent_names,
            "observed_vars": self.observed_names,
            "params": self.param_names,
            "execution_order": self.execution_order,
        }


def trace_model(
    model_fn,
    *args,
    rng_key: Optional[jax.random.PRNGKey] = None,
    substitutions: Optional[dict[str, Any]] = None,
    **kwargs,
) -> Trace:
    """Выполняем модель и захватываем trace.

    Пушим TraceHandler в контекст, прогоняем model_fn,
    всё что sample/param/factor пишется в trace.

    Args:
        model_fn: Функция модели (или HyperionModel).
        *args: Позиционные аргументы модели.
        rng_key: JAX PRNG key для сэмплинга.
        substitutions: Словарь имя -> фиксированное значение (для conditioning).
        **kwargs: Keyword-аргументы модели.

    Returns:
        Trace со всеми записями.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    handler = TraceHandler(rng_key=rng_key, substitutions=substitutions or {})
    _push_trace_handler(handler)
    try:
        model_fn(*args, **kwargs)
    finally:
        _pop_trace_handler()

    return handler.trace
