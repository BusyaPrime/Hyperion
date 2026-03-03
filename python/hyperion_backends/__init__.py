"""Бэкенды HYPERION — абстракция над вычислениями.

JAXBackend — основной. Всё остальное — на будущее.
"""

from hyperion_backends.base import Backend
from hyperion_backends.jax_backend import JAXBackend

__all__ = ["Backend", "JAXBackend"]
