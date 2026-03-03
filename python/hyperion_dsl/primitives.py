"""Ядро DSL HYPERION — примитивы для описания вероятностных моделей.

Тут живут базовые кирпичики: sample, plate, param, deterministic, factor.
Если ты не понимаешь что тут происходит — начни с sample() и двигайся вниз.
Вся магия работает через контекстный стек трейс-хендлеров (thread-local).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Optional

import jax.numpy as jnp

# Тред-локальный стек для хендлеров трассировки.
# Каждый раз когда модель выполняется под trace_model(),
# сюда пушится хендлер который ловит все sample/plate/param вызовы.
# Если стек пустой — примитивы работают в "тупом" режиме без записи.
_TRACE_CONTEXT: threading.local = threading.local()


def _get_trace_handler() -> Optional[Any]:
    """Достать текущий хендлер из стека. None если стек пуст (модель вызвана вне трассировки)."""
    stack = getattr(_TRACE_CONTEXT, "stack", None)
    if stack and len(stack) > 0:
        return stack[-1]
    return None


def _push_trace_handler(handler: Any) -> None:
    """Запушить хендлер в стек. Вызывается из trace_model() перед запуском модели."""
    if not hasattr(_TRACE_CONTEXT, "stack"):
        _TRACE_CONTEXT.stack = []
    _TRACE_CONTEXT.stack.append(handler)


def _pop_trace_handler() -> Any:
    """Попнуть хендлер из стека. Если попаешь из пустого — сам виноват."""
    return _TRACE_CONTEXT.stack.pop()


def sample(
    name: str,
    dist: Any,
    obs: Optional[Any] = None,
    constraint: Optional[Any] = None,
) -> Any:
    """Засемплить случайную величину из распределения.

    Главный примитив DSL — каждый вызов регистрирует случайную переменную
    в графе модели. Работает только внутри trace_model(); снаружи — RuntimeError.

    Args:
        name: уникальное имя сайта. Повторы имён — баг, не фича.
        dist: распределение (из hyperion_dsl.distributions), из которого семплим.
        obs: если передано — наблюдаемая переменная (likelihood), иначе латентная.
        constraint: ограничение на область значений (positive, bounded и т.д.).

    Returns:
        Семпл или obs, если obs передан. В трассировке — то, что вернул хендлер.
    """
    handler = _get_trace_handler()
    if handler is not None:
        return handler.process_sample(name, dist, obs=obs, constraint=constraint)
    # Вне трассировки — просто семплим или возвращаем obs
    if obs is not None:
        return obs
    raise RuntimeError(
        f"sample('{name}') вызван вне контекста trace_model(). "
        "Оберни вызов модели в trace_model() или используй obs=."
    )


@contextmanager
def plate(name: str, size: int, dim: int = -1):
    """Контекст-менеджер для i.i.d. повторений (plate в графических моделях).

    Внутри plate переменные считаются условно независимыми. Размер plate
    задаёт batch-размерность — все sample() внутри получат эту ось.

    Args:
        name: имя plate (для отладки и структуры графа).
        size: число повторений (N в типичном случае).
        dim: ось, по которой разворачивается plate (по умолчанию -1).

    Yields:
        Контекст; внутри — jnp.arange(size) или аналогичный индекс.

    Пример:
        with plate("data", N):
            x = sample("x", Normal(mu, sigma))  # N независимых x
    """
    handler = _get_trace_handler()
    if handler is not None:
        with handler.process_plate(name, size, dim=dim) as ctx:
            yield ctx
    else:
        yield jnp.arange(size)


def param(
    name: str,
    init_value: Any = None,
    constraint: Optional[Any] = None,
) -> Any:
    """Зарегистрировать оптимизируемый параметр (для VI, flows и т.п.).

    Не случайная переменная, а детерминированный параметр, который
    подбирается при инференсе — как веса нейросети, только байесовские.

    Args:
        name: уникальное имя параметра.
        init_value: начальное значение (если None — хендлер может задать своё).
        constraint: ограничение на область (positive, unit_interval и т.д.).

    Returns:
        Текущее значение параметра. Вне трассировки — init_value.
    """
    handler = _get_trace_handler()
    if handler is not None:
        return handler.process_param(name, init_value, constraint=constraint)
    return init_value


def deterministic(name: str, value: Any) -> Any:
    """Зарегистрировать детерминированный узел в графе модели.

    Полезно для отслеживания промежуточных величин и для инференса —
    например, mu = alpha + beta * x перед sample("y", Normal(mu, sigma)).

    Args:
        name: имя узла (для отладки и инспекции графа).
        value: детерминированное значение (тензор или скаляр).

    Returns:
        value без изменений. В трассировке хендлер может обернуть/записать.
    """
    handler = _get_trace_handler()
    if handler is not None:
        return handler.process_deterministic(name, value)
    return value


def factor(name: str, log_weight: Any) -> None:
    """Добавить ручной лог-фактор к совместному распределению.

    Прямо добавляет log_weight к лог-плотности модели. Нужно для кастомных
    лайклихудов и конструкций, которые не укладываются в sample() —
    например, мягкие ограничения, penalty-термы, частичные наблюдения.

    Args:
        name: имя фактора (для отладки).
        log_weight: лог-вес (скаляр или тензор), добавляется к log p(x).

    Returns:
        None. Побочный эффект — изменение графа/логарифма совместной плотности.
    """
    handler = _get_trace_handler()
    if handler is not None:
        handler.process_factor(name, log_weight)
