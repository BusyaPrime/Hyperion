"""Компилятор: Model -> Trace -> IRGraph pipeline.

Цепочка: trace_model() -> GraphBuilder -> IRGraph -> IROptimizer.
На выходе — оптимизированный IR готовый для inference.
"""

from __future__ import annotations

from typing import Any, Optional

import jax

from hyperion_trace.trace import trace_model
from hyperion_graph.graph_builder import GraphBuilder
from hyperion_ir.ir import IRGraph
from hyperion_ir.optimizer import IROptimizer


class ModelCompiler:
    """Компилирует HYPERION модель в оптимизированный IR.

    Трассируем, строим граф, конвертируем в IR, прогоняем оптимизатор.
    """

    def __init__(self, optimize: bool = True, passes: Optional[list[str]] = None):
        self._optimize = optimize
        self._optimizer = IROptimizer(passes=passes) if optimize else None
        self._graph_builder = GraphBuilder()

    def compile(
        self,
        model_fn,
        *args,
        rng_key: Optional[jax.random.PRNGKey] = None,
        substitutions: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> IRGraph:
        """Полный пайплайн компиляции: trace -> graph -> IR -> optimize."""
        trace = trace_model(
            model_fn, *args,
            rng_key=rng_key,
            substitutions=substitutions,
            **kwargs,
        )

        model_graph = self._graph_builder.build(trace)

        ir = IRGraph.from_trace(trace, model_fn=model_fn, data=substitutions)

        for src, dst in model_graph.edges:
            if src in ir.nodes and dst in ir.nodes:
                if src not in ir.nodes[dst].parents:
                    ir.nodes[dst].parents.append(src)
                if dst not in ir.nodes[src].children:
                    ir.nodes[src].children.append(dst)

        latent_set = set(ir.latent_names)
        for node in ir.observed_nodes:
            if any(p in latent_set for p in node.parents):
                ir._has_dynamic_observed = True
                break

        ir.metadata["model_name"] = getattr(model_fn, "name", model_fn.__name__)
        ir.metadata["graph_summary"] = model_graph.to_dict()

        if self._optimizer is not None:
            ir = self._optimizer.optimize(ir)

        return ir
