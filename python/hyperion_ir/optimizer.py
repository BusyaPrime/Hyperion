"""Оптимизационные проходы для IR графа HYPERION.

Проходы:
  1. dead_node_elimination — убираем ноды без потомков (если они не observed)
  2. constant_folding — сворачиваем константы (где можем)
  3. common_subexpression_elimination — мержим одинаковые ноды по хешу
  4. topological_reorder — перестраиваем порядок исполнения

Каждый проход — идемпотентный: применять дважды = применить один раз.
Порядок проходов имеет значение (dead nodes после CSE, например).
"""

from __future__ import annotations

import logging
from typing import Optional

from hyperion_ir.ir import IRGraph, IRNodeType

logger = logging.getLogger(__name__)


class IROptimizer:
    """Применяет оптимизационные проходы к IRGraph."""

    def __init__(self, passes: Optional[list[str]] = None):
        self._passes = passes or [
            "dead_node_elimination",
            "constant_folding",
            "common_subexpression_elimination",
            "topological_reorder",
        ]

    def optimize(self, graph: IRGraph) -> IRGraph:
        """Прогоняем все проходы по очереди."""
        for pass_name in self._passes:
            method = getattr(self, f"_pass_{pass_name}", None)
            if method is None:
                logger.warning(f"Неизвестный проход оптимизации: {pass_name}")
                continue
            graph = method(graph)
            logger.debug(f"После {pass_name}: {len(graph.nodes)} нод")
        return graph

    def _pass_dead_node_elimination(self, graph: IRGraph) -> IRGraph:
        """Убиваем детерминистические ноды без детей и не observed.

        Мёртвый код — не нужен. Выкидываем пока есть что выкидывать.
        """
        changed = True
        while changed:
            changed = False
            to_remove = []
            for name, node in graph.nodes.items():
                if (
                    node.node_type == IRNodeType.DETERMINISTIC
                    and not node.children
                ):
                    to_remove.append(name)
            for name in to_remove:
                graph.remove_node(name)
                changed = True
                logger.debug(f"Убили мёртвую ноду: {name}")
        return graph

    def _pass_constant_folding(self, graph: IRGraph) -> IRGraph:
        """Сворачиваем константы где возможно.

        Если CONSTANT нода известна на этапе компиляции и все
        потребители — deterministic, можно пре-вычислить.
        Пока только логируем кандидатов — полный folding требует eval.
        """
        for name, node in list(graph.nodes.items()):
            if node.node_type == IRNodeType.CONSTANT and node.value is not None:
                for child_name in list(node.children):
                    child = graph.nodes.get(child_name)
                    if child and child.node_type == IRNodeType.DETERMINISTIC:
                        logger.debug(f"Кандидат на constant folding: {name} -> {child_name}")
        return graph

    def _pass_common_subexpression_elimination(self, graph: IRGraph) -> IRGraph:
        """Мержим ноды с одинаковым content_hash — CSE.

        Дубликаты вычислений — зло. Один раз посчитали — переиспользуем.
        """
        hash_map: dict[str, str] = {}
        to_merge: list[tuple[str, str]] = []

        for name, node in graph.nodes.items():
            if node.node_type in (IRNodeType.DETERMINISTIC, IRNodeType.CONSTANT):
                h = node.content_hash()
                if h in hash_map:
                    to_merge.append((name, hash_map[h]))
                else:
                    hash_map[h] = name

        for duplicate, original in to_merge:
            dup_node = graph.nodes.get(duplicate)
            if dup_node is None:
                continue
            for child_name in list(dup_node.children):
                child = graph.nodes.get(child_name)
                if child:
                    child.parents = [
                        original if p == duplicate else p for p in child.parents
                    ]
                    if child_name not in graph.nodes[original].children:
                        graph.nodes[original].children.append(child_name)
            graph.remove_node(duplicate)
            logger.debug(f"CSE: смержили {duplicate} в {original}")

        return graph

    def _pass_topological_reorder(self, graph: IRGraph) -> IRGraph:
        """Перестраиваем execution_order по топологической сортировке."""
        graph.execution_order = graph.topological_order()
        return graph
