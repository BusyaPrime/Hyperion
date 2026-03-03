"""Строим DAG зависимостей из execution trace модели.

Парсим trace, вытаскиваем ноды и рёбра,
получаем граф для визуализации и передачи в IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx


@dataclass
class NodeInfo:
    """Метаданные для ноды графа — тип, распределение, plates и т.д."""

    name: str
    node_type: str
    distribution_type: Optional[str] = None
    observed: bool = False
    shape: Optional[tuple] = None
    plates: list[str] = field(default_factory=list)
    constraint: Optional[str] = None


class ModelGraph:
    """DAG — граф зависимостей вероятностной модели.

    Ноды = sample/param/deterministic/factor,
    рёбра = кто от кого зависит (parent -> child).
    """

    def __init__(self):
        self.dag = nx.DiGraph()
        self._node_info: dict[str, NodeInfo] = {}

    def add_node(self, info: NodeInfo) -> None:
        self.dag.add_node(info.name)
        self._node_info[info.name] = info

    def add_edge(self, parent: str, child: str) -> None:
        self.dag.add_edge(parent, child)

    def get_node_info(self, name: str) -> NodeInfo:
        return self._node_info[name]

    @property
    def nodes(self) -> list[str]:
        return list(self.dag.nodes)

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self.dag.edges)

    def parents(self, name: str) -> list[str]:
        return list(self.dag.predecessors(name))

    def children(self, name: str) -> list[str]:
        return list(self.dag.successors(name))

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self.dag))

    def roots(self) -> list[str]:
        """Ноды без входов — параметры, константы."""
        return [n for n in self.dag.nodes if self.dag.in_degree(n) == 0]

    def leaves(self) -> list[str]:
        """Ноды без выходов — наблюдаемые, финальные sample."""
        return [n for n in self.dag.nodes if self.dag.out_degree(n) == 0]

    @property
    def latent_nodes(self) -> list[str]:
        return [
            n for n, info in self._node_info.items()
            if info.node_type == "SAMPLE" and not info.observed
        ]

    @property
    def observed_nodes(self) -> list[str]:
        return [
            n for n, info in self._node_info.items()
            if info.node_type == "SAMPLE" and info.observed
        ]

    def to_dict(self) -> dict[str, Any]:
        """Сериализация для JSON/логов — не трогай руками."""
        return {
            "nodes": {
                name: {
                    "type": info.node_type,
                    "distribution": info.distribution_type,
                    "observed": info.observed,
                    "plates": info.plates,
                    "shape": info.shape,
                }
                for name, info in self._node_info.items()
            },
            "edges": self.edges,
        }

    def __repr__(self) -> str:
        return (
            f"ModelGraph(nodes={len(self.dag.nodes)}, "
            f"edges={len(self.dag.edges)}, "
            f"latent={len(self.latent_nodes)}, "
            f"observed={len(self.observed_nodes)})"
        )


class GraphBuilder:
    """Строит ModelGraph из Trace.

    Флэтим trace в граф: добавляем ноды, инферим рёбра.
    """

    def build(self, trace) -> ModelGraph:
        graph = ModelGraph()

        for name in trace.execution_order:
            entry = trace.entries[name]
            dist_type = None
            if entry.distribution is not None:
                dist_type = type(entry.distribution).__name__

            shape = None
            if entry.value is not None and hasattr(entry.value, 'shape'):
                shape = tuple(entry.value.shape)

            constraint_str = None
            if entry.constraint is not None:
                constraint_str = repr(entry.constraint)

            info = NodeInfo(
                name=name,
                node_type=entry.node_type.name,
                distribution_type=dist_type,
                observed=entry.observed,
                shape=shape,
                plates=list(entry.plates),
                constraint=constraint_str,
            )
            graph.add_node(info)

        self._infer_edges(trace, graph)
        return graph

    def _infer_edges(self, trace, graph: ModelGraph) -> None:
        """Инферим рёбра зависимостей: по execution order и data flow.

        Эвристика:
        1. Если у distribution параметры ссылаются на values из более ранних trace entries — добавляем parent edge.
        2. Явные entry.parents (если заполнены) — тоже добавляем.
        3. Дополнительно: если value — это результат операции над другими values — ищем по shape/type.
        """
        known_values: dict[str, int] = {}
        for name in trace.execution_order:
            entry = trace.entries[name]
            if entry.value is not None:
                known_values[name] = id(entry.value)

            # Связи через distribution params
            if entry.distribution is not None:
                dist = entry.distribution
                for attr_name in vars(dist):
                    attr_val = getattr(dist, attr_name)
                    if hasattr(attr_val, '__jax_array__') or hasattr(attr_val, 'shape'):
                        attr_id = id(attr_val)
                        for prev_name, prev_id in known_values.items():
                            if prev_name != name and attr_id == prev_id:
                                graph.add_edge(prev_name, name)

            # Явные parents (если кто-то их заполнил)
            if entry.parents:
                for parent in entry.parents:
                    if parent in graph._node_info:
                        graph.add_edge(parent, name)

            # Дополнительная эвристика: factor/value часто зависит от sample
            # Если value — это array и мы его видели в dist params — уже учли выше
            # Пропускаем дубликаты рёбер (add_edge идемпотентен)
