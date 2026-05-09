from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from hyperion_diagnostics.report import DiagnosticsReport
from hyperion_diagnostics.report import generate_report


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
    result = SimpleNamespace(
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
    result = SimpleNamespace(
        samples={"mu": np.ones(150)},
        diagnostics={"accept_rate": 0.8, "num_divergences": 0},
    )

    report = generate_report(result, model_name="sticky_model", inference_method="hmc")

    assert any("Low ESS for mu/ess" in warning for warning in report.warnings)
    assert report.conclusions == [
        "1 potential issue(s) detected. Review warnings before trusting results."
    ]


def test_generate_report_warns_on_sampler_diagnostics() -> None:
    rng = np.random.default_rng(7)
    result = SimpleNamespace(
        samples={"mu": rng.normal(size=500)},
        diagnostics={
            "accept_rate": 0.4,
            "num_divergences": 2,
            "energy": np.cumsum(rng.normal(0.0, 0.01, size=500)),
        },
    )

    report = generate_report(result, model_name="difficult_model", inference_method="nuts")

    assert any("Low acceptance rate" in warning for warning in report.warnings)
    assert any("divergent transitions" in warning for warning in report.warnings)
    assert any("Low BFMI" in warning for warning in report.warnings)


def test_generate_report_includes_multichain_vector_parameters() -> None:
    rng = np.random.default_rng(11)
    chains = rng.normal(size=(3, 120, 2))
    result = SimpleNamespace(
        samples={"beta": chains.reshape(-1, 2)},
        diagnostics={"accept_rate": 0.9, "num_divergences": 0},
        num_chains=3,
        samples_by_chain={"beta": chains},
    )

    report = generate_report(result, model_name="linear_model", inference_method="hmc")

    assert "beta[0]" in report.summary_stats
    assert "beta[1]" in report.summary_stats
    assert "beta[0]/r_hat" in report.convergence_metrics
    assert "beta[1]/split_r_hat" in report.convergence_metrics


def test_diagnostics_report_markdown_renders_warnings_and_conclusions() -> None:
    report = DiagnosticsReport(
        model_name="warning_model",
        inference_method="vi",
        timestamp="2026-05-09T10:00:00",
        warnings=["Low ESS for z/ess: 30.0 (recommended > 100)"],
        conclusions=["Review warnings before trusting results."],
    )

    markdown = report.to_markdown()

    assert "## Warnings" in markdown
    assert "Low ESS for z/ess" in markdown
    assert "## Conclusions" in markdown
    assert "- Review warnings before trusting results." in markdown
