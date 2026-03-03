"""Декоратор @model — превращает обычную Python-функцию в модель HYPERION.

Использование:
    @model
    def my_cool_model(data):
        mu = sample("mu", Normal(0, 1))
        ...

    @model(name="кастомное_имя")
    def another_model(data):
        ...

Модель — это callable обёртка, которая хранит метаданные,
хеш исходного кода и сигнатуру. Нужно для воспроизводимости,
сериализации и чтобы можно было сравнивать модели между собой.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from typing import Any, Callable, Optional

from pydantic import BaseModel as PydanticBaseModel


class ModelInfo(PydanticBaseModel):
    """Метаданные зарегистрированной модели.

    Иммутабельный (frozen) объект с основной инфой:
    имя, хеш исходника, количество и имена аргументов.
    """
    name: str
    source_hash: str
    num_args: int
    arg_names: list[str]

    model_config = {"frozen": True}


class HyperionModel:
    """Обёртка над Python-функцией, превращающая её в модель HYPERION.

    Callable — вызывается как обычная функция. Хранит метаданные (имя,
    хеш исходника, сигнатуру) для воспроизводимости и сериализации.
    Используется трассировщиками и инференс-движками.

    Args:
        fn: Python-функция, описывающая модель (с sample, plate, param внутри).
        name: кастомное имя модели; если None — берётся fn.__name__.
    """

    def __init__(self, fn: Callable[..., Any], name: Optional[str] = None):
        self._fn = fn
        self._name = name or fn.__name__
        try:
            self._source = inspect.getsource(fn)
        except OSError:
            # Если исходник недоступен (например, lambda или интерактивный режим)
            self._source = f"<недоступен для {self._name}>"
        self._sig = inspect.signature(fn)
        functools.update_wrapper(self, fn)

    @property
    def name(self) -> str:
        return self._name

    @property
    def info(self) -> ModelInfo:
        """Метаданные модели: имя, хеш исходника, число и имена аргументов."""
        return ModelInfo(
            name=self._name,
            source_hash=hashlib.sha256(self._source.encode()).hexdigest()[:16],
            num_args=len(self._sig.parameters),
            arg_names=list(self._sig.parameters.keys()),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"HyperionModel({self._name})"


def model(fn: Optional[Callable[..., Any]] = None, *, name: Optional[str] = None):
    """Декоратор для регистрации функции как модели HYPERION.

    Оборачивает функцию в HyperionModel. Поддерживает оба варианта вызова:
    @model и @model(name="..."). Результат — callable с метаданными.

    Args:
        fn: декорируемая функция (передаётся автоматически при @model).
        name: опциональное имя модели; если None — fn.__name__.

    Returns:
        HyperionModel — обёртка, вызываемая как fn, с полями .name и .info.
    """
    if fn is not None:
        return HyperionModel(fn, name=name)

    def wrapper(fn: Callable[..., Any]) -> HyperionModel:
        return HyperionModel(fn, name=name)

    return wrapper
