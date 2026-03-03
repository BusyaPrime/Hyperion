"""HYPERION Diagnostics — статистическая диагностика для inference.

ESS, R-hat, автокорреляция, PPC, отчёты. Всё что нужно чтобы понять: сходится ли, можно ли доверять.
"""

from hyperion_diagnostics.metrics import (
    effective_sample_size,
    effective_sample_size_multichain,
    r_hat,
    split_r_hat,
    autocorrelation,
    acceptance_rate,
    divergence_count,
    compute_all_diagnostics,
    summary_table,
    print_summary,
)
from hyperion_diagnostics.ppc import posterior_predictive_check
from hyperion_diagnostics.report import DiagnosticsReport, generate_report

__all__ = [
    "effective_sample_size",
    "effective_sample_size_multichain",
    "r_hat",
    "split_r_hat",
    "autocorrelation",
    "acceptance_rate",
    "divergence_count",
    "compute_all_diagnostics",
    "summary_table",
    "print_summary",
    "posterior_predictive_check",
    "DiagnosticsReport",
    "generate_report",
]
