from __future__ import annotations

import json

from hyperion_diagnostics.report import DiagnosticsReport


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
