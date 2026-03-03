"""HYPERION IR — Intermediate Representation и оптимизатор.

Экспортируем IRNode, IRGraph, IRNodeType, IROptimizer.
"""

from hyperion_ir.ir import IRNode, IRGraph, IRNodeType
from hyperion_ir.optimizer import IROptimizer

__all__ = ["IRNode", "IRGraph", "IRNodeType", "IROptimizer"]
