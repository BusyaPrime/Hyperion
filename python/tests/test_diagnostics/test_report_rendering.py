from __future__ import annotations

from hyperion_diagnostics.report import DiagnosticsReport


def test_report_markdown_preserves_section_order() -> None:
    report = DiagnosticsReport(
        model_name="ordered_model",
        inference_method="nuts",
        timestamp="2026-05-09T10:00:00",
        config={"num_samples": 100},
        summary_stats={"mu": {"mean": 0.0, "std": 1.0}},
        convergence_metrics={"mu/ess": 120.0},
        warnings=["Low BFMI"],
        conclusions=["Review diagnostics."],
    )

    markdown = report.to_markdown()

    assert markdown.index("## Configuration") < markdown.index("## Parameter Summary")
    assert markdown.index("## Parameter Summary") < markdown.index("## Convergence Metrics")
    assert markdown.index("## Convergence Metrics") < markdown.index("## Warnings")
    assert markdown.index("## Warnings") < markdown.index("## Conclusions")


def test_report_markdown_omits_empty_optional_sections() -> None:
    report = DiagnosticsReport(
        model_name="minimal_model",
        inference_method="laplace",
        timestamp="2026-05-09T10:00:00",
    )

    markdown = report.to_markdown()

    assert "## Warnings" not in markdown
    assert "## Conclusions" not in markdown
