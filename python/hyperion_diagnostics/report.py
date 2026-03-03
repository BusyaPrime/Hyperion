"""Генерация диагностических отчётов для HYPERION.

Собираем summary stats, convergence metrics, автоматические варнинги.
Форматируем в JSON и Markdown — для людей и для парсеров.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np


@dataclass
class DiagnosticsReport:
    """Структурированный отчёт: модель, метод, конфиг, summary, convergence, варнинги, выводы."""

    model_name: str = ""
    inference_method: str = ""
    timestamp: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    summary_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    convergence_metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    conclusions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """В dict для JSON. default=str — для numpy/datetime."""
        return {
            "model_name": self.model_name,
            "inference_method": self.inference_method,
            "timestamp": self.timestamp,
            "config": self.config,
            "summary_stats": self.summary_stats,
            "convergence_metrics": self.convergence_metrics,
            "warnings": self.warnings,
            "conclusions": self.conclusions,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_markdown(self) -> str:
        """Markdown-отчёт. Таблица для summary, списки для варнингов. Красиво и читаемо."""
        lines = [
            f"# Diagnostics Report: {self.model_name}",
            f"**Method:** {self.inference_method}",
            f"**Timestamp:** {self.timestamp}",
            "",
            "## Configuration",
        ]
        for k, v in self.config.items():
            lines.append(f"- **{k}:** {v}")

        lines.append("")
        lines.append("## Parameter Summary")
        lines.append("")

        if self.summary_stats:
            headers = None
            for param, stats in self.summary_stats.items():
                if headers is None:
                    headers = list(stats.keys())
                    lines.append("| Parameter | " + " | ".join(headers) + " |")
                    lines.append("|" + "|".join(["---"] * (len(headers) + 1)) + "|")
                vals = " | ".join(
                    f"{stats.get(h, 'N/A'):.4f}" if isinstance(stats.get(h), float)
                    else str(stats.get(h, 'N/A'))
                    for h in headers
                )
                lines.append(f"| {param} | {vals} |")

        lines.append("")
        lines.append("## Convergence Metrics")
        for k, v in self.convergence_metrics.items():
            if isinstance(v, float):
                lines.append(f"- **{k}:** {v:.4f}")
            else:
                lines.append(f"- **{k}:** {v}")

        if self.warnings:
            lines.append("")
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- ⚠ {w}")

        if self.conclusions:
            lines.append("")
            lines.append("## Conclusions")
            for c in self.conclusions:
                lines.append(f"- {c}")

        return "\n".join(lines)


def generate_report(
    inference_result: Any,
    model_name: str = "unknown",
    inference_method: str = "unknown",
    config: Optional[dict[str, Any]] = None,
) -> DiagnosticsReport:
    """Генерируем отчёт из InferenceResult.

    Summary table, convergence metrics, автоматические варнинги по порогам.
    """
    from hyperion_diagnostics.metrics import (
        compute_all_diagnostics,
        summary_table,
    )

    report = DiagnosticsReport(
        model_name=model_name,
        inference_method=inference_method,
        timestamp=datetime.now().isoformat(),
        config=config or {},
    )

    # Summary statistics
    samples_np = {
        name: np.asarray(vals)
        for name, vals in inference_result.samples.items()
    }
    by_chain = getattr(inference_result, 'samples_by_chain', None)
    by_chain_np = None
    if by_chain is not None:
        by_chain_np = {name: np.asarray(v) for name, v in by_chain.items()}
    report.summary_stats = summary_table(samples_np, samples_by_chain=by_chain_np)

    # Convergence metrics
    report.convergence_metrics = compute_all_diagnostics(inference_result)

    # Автоматические варнинги по порогам
    for key, val in report.convergence_metrics.items():
        if key.endswith("/ess") and isinstance(val, float) and val < 100:
            report.warnings.append(
                f"Low ESS for {key}: {val:.1f} (recommended > 100)"
            )
        if key == "accept_rate" and isinstance(val, float) and val < 0.5:
            report.warnings.append(
                f"Low acceptance rate: {val:.3f} (recommended > 0.65)"
            )
        if key == "num_divergences" and isinstance(val, int) and val > 0:
            report.warnings.append(
                f"{val} divergent transitions detected"
            )
        if key == "bfmi" and isinstance(val, float) and val < 0.3:
            report.warnings.append(
                f"Low BFMI: {val:.3f} (values < 0.3 suggest problems)"
            )

    # Conclusions
    if not report.warnings:
        report.conclusions.append(
            "No convergence issues detected. Results appear reliable."
        )
    else:
        report.conclusions.append(
            f"{len(report.warnings)} potential issue(s) detected. "
            "Review warnings before trusting results."
        )

    return report
