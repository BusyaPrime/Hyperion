"""Composable effect handlers для HYPERION PPL.

Четыре базовых handler-а, composable через стекирование:
  - TraceMessenger: записывает все sample/param/factor в Trace
  - SubstituteMessenger: подставляет фиксированные значения
  - ReplayMessenger: воспроизводит значения из другого trace
  - BlockMessenger: блокирует видимость определённых сайтов

Composable: вложенные handler-ы обрабатывают сообщения по цепочке.
Вдохновлено Pyro/NumPyro effect system.

Использование:
    # TraceMessenger должен быть внешним (дном цепочки)
    with TraceMessenger() as trace:
        with SubstituteMessenger(data={"obs": y_data}):
            model()
    # trace.trace содержит записи, obs подставлен

    with TraceMessenger() as trace:
        with BlockMessenger(hide=["sigma"]):
            model()
    # sigma не будет в trace
"""

from __future__ import annotations

from abc import ABC
from contextlib import contextmanager
from typing import Any, Optional

import jax
import jax.numpy as jnp

from hyperion_dsl.primitives import (
    _get_trace_handler,
    _push_trace_handler,
    _pop_trace_handler,
)


class Messenger(ABC):
    """Базовый handler — перехватывает DSL-примитивы.

    Каждый messenger реализует process_message() для каждого типа.
    Вложенные messenger-ы вызываются по цепочке через _inner.
    """

    def __init__(self):
        self._inner = None

    def __enter__(self):
        self._inner = _get_trace_handler()
        _push_trace_handler(self)
        return self

    def __exit__(self, *args):
        _pop_trace_handler()

    def process_sample(self, name, dist, obs=None, constraint=None):
        if self._inner is not None:
            return self._inner.process_sample(name, dist, obs=obs, constraint=constraint)
        raise RuntimeError("Нет inner handler — sample() не может быть обработан")

    def process_plate(self, name, size, dim=-1):
        if self._inner is not None:
            return self._inner.process_plate(name, size, dim=dim)
        raise RuntimeError("Нет inner handler — plate() не может быть обработан")

    def process_param(self, name, init_value, constraint=None):
        if self._inner is not None:
            return self._inner.process_param(name, init_value, constraint=constraint)
        raise RuntimeError("Нет inner handler — param() не может быть обработан")

    def process_deterministic(self, name, value):
        if self._inner is not None:
            return self._inner.process_deterministic(name, value)
        return value

    def process_factor(self, name, log_weight):
        if self._inner is not None:
            return self._inner.process_factor(name, log_weight)


class TraceMessenger(Messenger):
    """Записывает все sample/param/factor в Trace. Базовый handler для трассировки.

    Является «дном» цепочки — когда _inner is None, выполняет сэмплинг и запись.
    Используется как: with TraceMessenger() as t: model()  # t.trace содержит записи.
    """

    def __init__(self, rng_key: Optional[jax.random.PRNGKey] = None):
        super().__init__()
        self._rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        from hyperion_trace.trace import Trace

        self.trace = Trace()
        self._plate_stack: list[tuple[str, int, int]] = []

    def _next_key(self) -> jax.random.PRNGKey:
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    def _current_plates(self) -> list[str]:
        return [name for name, _, _ in self._plate_stack]

    def process_sample(self, name, dist, obs=None, constraint=None):
        if self._inner is not None:
            value = self._inner.process_sample(name, dist, obs=obs, constraint=constraint)
            self._record_sample(name, dist, value, obs, constraint)
            return value
        return self._do_sample_and_record(name, dist, obs, constraint)

    def _record_sample(self, name, dist, value, obs, constraint):
        from hyperion_trace.trace import TraceEntry, NodeType

        is_observed = obs is not None
        log_p = dist.log_prob(value)
        entry = TraceEntry(
            name=name,
            node_type=NodeType.SAMPLE,
            distribution=dist,
            value=value,
            observed=is_observed,
            obs_value=obs,
            constraint=constraint,
            log_prob=log_p,
            plates=self._current_plates(),
        )
        self.trace.add_entry(entry)

    def _do_sample_and_record(self, name, dist, obs=None, constraint=None):
        from hyperion_trace.trace import TraceEntry, NodeType

        is_observed = obs is not None
        if obs is not None:
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
            obs_value=obs,
            constraint=constraint,
            log_prob=log_p,
            plates=self._current_plates(),
        )
        self.trace.add_entry(entry)
        return value

    @contextmanager
    def process_plate(self, name, size, dim=-1):
        if self._inner is not None:
            with self._inner.process_plate(name, size, dim=dim) as ctx:
                self._plate_stack.append((name, size, dim))
                self.trace.plate_stack.append((name, size, dim))
                try:
                    yield ctx
                finally:
                    self._plate_stack.pop()
                    self.trace.plate_stack.pop()
        else:
            self._plate_stack.append((name, size, dim))
            self.trace.plate_stack.append((name, size, dim))
            try:
                yield jnp.arange(size)
            finally:
                self._plate_stack.pop()
                self.trace.plate_stack.pop()

    def process_param(self, name, init_value, constraint=None):
        if self._inner is not None:
            return self._inner.process_param(name, init_value, constraint=constraint)
        value = (
            jnp.asarray(init_value, dtype=jnp.float32)
            if init_value is not None
            else None
        )
        from hyperion_trace.trace import TraceEntry, NodeType

        entry = TraceEntry(
            name=name,
            node_type=NodeType.PARAM,
            value=value,
            constraint=constraint,
            plates=self._current_plates(),
        )
        self.trace.add_entry(entry)
        return value

    def process_deterministic(self, name, value):
        if self._inner is not None:
            return self._inner.process_deterministic(name, value)
        from hyperion_trace.trace import TraceEntry, NodeType

        entry = TraceEntry(
            name=name,
            node_type=NodeType.DETERMINISTIC,
            value=value,
            plates=self._current_plates(),
        )
        self.trace.add_entry(entry)
        return value

    def process_factor(self, name, log_weight):
        if self._inner is not None:
            return self._inner.process_factor(name, log_weight)
        from hyperion_trace.trace import TraceEntry, NodeType

        entry = TraceEntry(
            name=name,
            node_type=NodeType.FACTOR,
            value=log_weight,
            plates=self._current_plates(),
        )
        self.trace.add_entry(entry)


class SubstituteMessenger(Messenger):
    """Подставляет значения из data dict. Аналог numpyro.handlers.substitute.

    Если имя sample-сайта есть в data — подставляем значение вместо сэмплинга.
    """

    def __init__(self, data: dict[str, Any]):
        super().__init__()
        self._data = data

    def process_sample(self, name, dist, obs=None, constraint=None):
        new_obs = obs
        if name in self._data and obs is None:
            new_obs = self._data[name]
        return super().process_sample(name, dist, obs=new_obs, constraint=constraint)


class ReplayMessenger(Messenger):
    """Воспроизводит значения из записанного trace. Аналог numpyro.handlers.replay.

    Полезно для SVI: replay latent values из guide в model.
    """

    def __init__(self, trace):
        super().__init__()
        self._guide_trace = trace

    def process_sample(self, name, dist, obs=None, constraint=None):
        if name in self._guide_trace and obs is None:
            entry = self._guide_trace[name]
            from hyperion_trace.trace import NodeType
            is_sample = entry.node_type == NodeType.SAMPLE
            if is_sample and not entry.observed and entry.value is not None:
                return super().process_sample(
                    name, dist, obs=entry.value, constraint=constraint
                )
        return super().process_sample(name, dist, obs=obs, constraint=constraint)


class BlockMessenger(Messenger):
    """Блокирует видимость определённых sample-сайтов. Аналог numpyro.handlers.block.

    hide_fn: функция (name) -> bool, True = блокировать
    hide: список имён для блокировки (альтернатива hide_fn)
    """

    def __init__(
        self,
        hide: Optional[list[str]] = None,
        hide_fn: Optional[Any] = None,
    ):
        super().__init__()
        if hide is not None:
            self._hide_fn = lambda name: name in hide
        elif hide_fn is not None:
            self._hide_fn = hide_fn
        else:
            raise ValueError(
                "BlockMessenger requires either 'hide' (list of names) or 'hide_fn' (callable). "
                "Without arguments it would block ALL sites, which is almost certainly a bug."
            )

    def process_sample(self, name, dist, obs=None, constraint=None):
        if self._hide_fn(name):
            key = jax.random.PRNGKey(sum(ord(c) for c in name) % (2**31))
            return dist.sample(key)
        return super().process_sample(name, dist, obs=obs, constraint=constraint)

    def process_factor(self, name, log_weight):
        if self._hide_fn(name):
            return None
        return super().process_factor(name, log_weight)
