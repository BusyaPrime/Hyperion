from __future__ import annotations

import json

import numpy as np

from hyperion_diagnostics.report import DiagnosticsReport
from hyperion_diagnostics.report import generate_report
from hyperion_inference.base import InferenceResult


def test_diagnostics_report_json_serializes_core_fields() -> None:
    report = DiagnosticsReport(
        model_name="normal_model",
        inference_method="hmc",
        timestamp="2026-05-09T10:00:00",
        config={"num_samples": 100},
        summary_stats={"mu": {"mean": 0.1, "std": 1.2}},
        convergence_metrics={"accept_rate": 0.8},
    )

    payload = json.loads(report.to_json())

    assert payload["model_name"] == "normal_model"
    assert payload["inference_method"] == "hmc"
    assert payload["config"]["num_samples"] == 100
    assert payload["summary_stats"]["mu"]["mean"] == 0.1


def test_diagnostics_report_markdown_renders_configuration_and_summary_table() -> None:
    report = DiagnosticsReport(
        model_name="coin_model",
        inference_method="nuts",
        timestamp="2026-05-09T10:00:00",
        config={"num_warmup": 50, "num_samples": 100},
        summary_stats={
            "mu": {
                "mean": 2.0,
                "std": 0.5,
                "median": 2.1,
                "ci_5.0%": 1.2,
                "ci_95.0%": 2.8,
                "ess": 88.0,
            }
        },
        convergence_metrics={"mu/ess": 88.0},
    )

    markdown = report.to_markdown()

    assert "# Diagnostics Report: coin_model" in markdown
    assert "- **num_warmup:** 50" in markdown
    assert "| Parameter | mean | std | median | ci_5.0% | ci_95.0% | ess |" in markdown
    assert "| mu | 2.0000 | 0.5000 | 2.1000 | 1.2000 | 2.8000 | 88.0000 |" in markdown


def test_generate_report_adds_clean_conclusion_when_diagnostics_pass() -> None:
    result = InferenceResult(
        samples={"mu": np.linspace(-1.0, 1.0, 200)},
        diagnostics={"accept_rate": 0.75, "num_divergences": 0},
    )

    report = generate_report(
        result,
        model_name="normal_model",
        inference_method="hmc",
        config={"num_samples": 200},
    )

    assert report.model_name == "normal_model"
    assert report.warnings == []
    assert report.conclusions == ["No convergence issues detected. Results appear reliable."]
    assert "mu" in report.summary_stats


def test_generate_report_warns_on_low_effective_sample_size() -> None:
    result = InferenceResult(
        samples={"mu": np.ones(150)},
        diagnostics={"accept_rate": 0.8, "num_divergences": 0},
    )

    report = generate_report(result, model_name="sticky_model", inference_method="hmc")

    assert any("Low ESS for mu/ess" in warning for warning in report.warnings)
    assert report.conclusions == [
        "1 potential issue(s) detected. Review warnings before trusting results."
    ]
