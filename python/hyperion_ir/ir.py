"""Intermediate Representation для скомпилированных вероятностных моделей.

Сюда флэтим trace/graph — получаем IRGraph с нодами, parents/children,
log_prob_fn и прочим. Дальше оптимизатор и компилятор работают с IR.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


class IRNodeType(Enum):
    """Типы нод в IR: sample, observed, param, deterministic, factor, constant."""
    SAMPLE = auto()
    OBSERVED = auto()
    PARAM = auto()
    DETERMINISTIC = auto()
    FACTOR = auto()
    CONSTANT = auto()


@dataclass
class IRNode:
    """Одна нода в IR-графе.

    Содержит тип, родителей/детей, shape, distribution, log_prob_fn.
    """

    name: str
    node_type: IRNodeType
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    shape: Optional[tuple[int, ...]] = None
    dtype: str = "float32"

    distribution_type: Optional[str] = None
    distribution_params: Optional[dict[str, Any]] = None
    constraint: Optional[str] = None
    plates: list[str] = field(default_factory=list)

    value: Optional[Any] = None
    log_prob_fn: Optional[Callable] = None

    _hash: Optional[str] = None

    @property
    def is_latent(self) -> bool:
        return self.node_type == IRNodeType.SAMPLE

    @property
    def is_observed(self) -> bool:
        return self.node_type == IRNodeType.OBSERVED

    def content_hash(self) -> str:
        """Хеш содержимого ноды — для CSE (common subexpression elimination)."""
        if self._hash is None:
            params_repr = None
            if self.distribution_params is not None:
                params_repr = {
                    k: float(v) if hasattr(v, '__float__') else str(v)
                    for k, v in self.distribution_params.items()
                }
            data = json.dumps({
                "type": self.node_type.name,
                "dist": self.distribution_type,
                "dist_params": params_repr,
                "shape": list(self.shape) if self.shape else None,
                "parents": sorted(self.parents),
                "plates": self.plates,
            }, sort_keys=True)
            self._hash = hashlib.sha256(data.encode()).hexdigest()[:12]
        return self._hash

    def __repr__(self) -> str:
        return f"IRNode({self.name}, type={self.node_type.name}, shape={self.shape})"


class IRGraph:
    """Скомпилированное IR вероятностной модели.

    Ноды + execution_order + metadata.
    Умеет compute_log_joint, grad_log_joint, topological_order.

    compute_log_joint — быстрый путь для статических моделей (observed params
    не зависят от latents). Для dynamic models используй compute_log_joint_dynamic.
    """

    def __init__(self):
        self.nodes: dict[str, IRNode] = {}
        self.execution_order: list[str] = []
        self.metadata: dict[str, Any] = {}
        self._model_fn: Optional[Any] = None
        self._data: Optional[dict[str, Any]] = None
        self._has_dynamic_observed: bool = False

    def add_node(self, node: IRNode) -> None:
        self.nodes[node.name] = node
        if node.name not in self.execution_order:
            self.execution_order.append(node.name)

    def remove_node(self, name: str) -> None:
        """Удаляем ноду и чистим ссылки у родителей/детей. Не трогай руками."""
        if name in self.nodes:
            node = self.nodes[name]
            for parent_name in node.parents:
                if parent_name in self.nodes:
                    parent = self.nodes[parent_name]
                    if name in parent.children:
                        parent.children.remove(name)
            for child_name in node.children:
                if child_name in self.nodes:
                    child = self.nodes[child_name]
                    if name in child.parents:
                        child.parents.remove(name)
            del self.nodes[name]
            if name in self.execution_order:
                self.execution_order.remove(name)

    @property
    def latent_nodes(self) -> list[IRNode]:
        return [n for n in self.nodes.values() if n.is_latent]

    @property
    def observed_nodes(self) -> list[IRNode]:
        return [n for n in self.nodes.values() if n.is_observed]

    @property
    def latent_names(self) -> list[str]:
        return [n.name for n in self.latent_nodes]

    @property
    def observed_names(self) -> list[str]:
        return [n.name for n in self.observed_nodes]

    def topological_order(self) -> list[str]:
        """Топологическая сортировка по parents — тут водятся драконы если циклы."""
        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            node = self.nodes[name]
            for parent in node.parents:
                if parent in self.nodes:
                    visit(parent)
            order.append(name)

        for name in self.nodes:
            visit(name)
        return order

    def compute_log_joint(
        self,
        latent_values: dict[str, jnp.ndarray],
        observed_values: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Суммируем log_prob по latent + observed + factor."""
        total = jnp.float32(0.0)
        for name in self.topological_order():
            node = self.nodes[name]
            if node.node_type == IRNodeType.FACTOR:
                if node.value is not None:
                    fv = node.value
                else:
                    continue
                if hasattr(fv, 'ndim') and fv.ndim > 0:
                    fv = jnp.sum(fv)
                total = total + fv
            elif node.log_prob_fn is not None:
                if node.is_latent and name in latent_values:
                    lp = node.log_prob_fn(latent_values[name])
                elif node.is_observed and name in observed_values:
                    lp = node.log_prob_fn(observed_values[name])
                else:
                    continue
                if hasattr(lp, 'ndim') and lp.ndim > 0:
                    lp = jnp.sum(lp)
                total = total + lp
        return total

    def compute_log_joint_dynamic(
        self,
        latent_values: dict[str, jnp.ndarray],
        observed_values: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Log joint через re-trace — корректен для dynamic models.

        Используется когда observed nodes зависят от latent параметров
        (e.g. y ~ Normal(x, 1.0) где x — latent).
        """
        if self._model_fn is None:
            raise RuntimeError(
                "compute_log_joint_dynamic requires model_fn. "
                "Use ModelCompiler.compile() to build IRGraph."
            )
        from hyperion_trace.trace import trace_model
        subs = {**observed_values, **latent_values}
        trace = trace_model(self._model_fn, substitutions=subs)
        return trace.log_joint()

    def grad_log_joint(
        self,
        latent_values: dict[str, jnp.ndarray],
        observed_values: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        """Градиент log_joint по латентным — через jax.grad.

        Автоматически выбирает static или dynamic compute_log_joint.
        """
        if self._has_dynamic_observed:
            def lj_fn(latents):
                return self.compute_log_joint_dynamic(latents, observed_values)
        else:
            def lj_fn(latents):
                return self.compute_log_joint(latents, observed_values)

        return jax.grad(lj_fn)(latent_values)

    def to_dict(self) -> dict[str, Any]:
        """Сериализация для дебага/логов."""
        return {
            "nodes": {
                name: {
                    "type": node.node_type.name,
                    "parents": node.parents,
                    "children": node.children,
                    "shape": list(node.shape) if node.shape else None,
                    "distribution": node.distribution_type,
                    "plates": node.plates,
                }
                for name, node in self.nodes.items()
            },
            "execution_order": self.execution_order,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_trace(trace, model_fn=None, data=None) -> IRGraph:
        """Собираем IRGraph из Trace — парсим entries, вешаем log_prob_fn, связываем рёбра."""
        graph = IRGraph()
        graph._model_fn = model_fn
        graph._data = data

        latent_names_set = set(trace.latent_names)

        for name in trace.execution_order:
            entry = trace.entries[name]

            if entry.node_type.name == "SAMPLE" and entry.observed:
                node_type = IRNodeType.OBSERVED
            else:
                node_type = IRNodeType[entry.node_type.name]

            shape = None
            if entry.value is not None and hasattr(entry.value, 'shape'):
                shape = tuple(entry.value.shape)

            dist_type = None
            if entry.distribution is not None:
                dist_type = type(entry.distribution).__name__

            log_prob_fn = None
            if entry.distribution is not None:
                dist = entry.distribution
                log_prob_fn = dist.log_prob

            node = IRNode(
                name=name,
                node_type=node_type,
                shape=shape,
                distribution_type=dist_type,
                plates=list(entry.plates),
                constraint=repr(entry.constraint) if entry.constraint else None,
                log_prob_fn=log_prob_fn,
            )

            if entry.node_type.name == "FACTOR":
                node.value = entry.value

            graph.add_node(node)

        for name in graph.execution_order:
            entry = trace.entries[name]
            if entry.parents:
                node = graph.nodes[name]
                for parent_name in entry.parents:
                    if parent_name in graph.nodes:
                        node.parents.append(parent_name)
                        graph.nodes[parent_name].children.append(name)

        if graph._model_fn is not None:
            for node in graph.observed_nodes:
                if any(p in latent_names_set for p in node.parents):
                    graph._has_dynamic_observed = True
                    break

        return graph

    def __repr__(self) -> str:
        return (
            f"IRGraph(nodes={len(self.nodes)}, "
            f"latent={len(self.latent_nodes)}, "
            f"observed={len(self.observed_nodes)})"
        )
