"""Stage 18 -- tests for src/monitoring.py.

Covers: normal (non-drifting) batches, deliberately shifted batches,
missingness handling, invalid/malformed inputs, small/insufficient
sample sizes, empty batches, PSI correctness, operational log
parsing (including real Stage 17 log fixtures), and threshold
decisions. No `final_test` access anywhere in this file.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring import (
    MIN_BATCH_SAMPLE_SIZE,
    MIN_REFERENCE_SAMPLE_SIZE,
    build_reference_distribution,
    build_reference_score_distribution,
    evaluate_input_drift,
    evaluate_operational_health,
    evaluate_score_drift,
    parse_log_file,
    population_stability_index,
    score_batch_distribution,
    summarize_operational_records,
)


# ---------------------------------------------------------------------------
# PSI correctness
# ---------------------------------------------------------------------------

def test_psi_is_zero_for_identical_distributions():
    props = [0.1] * 10
    assert population_stability_index(props, props) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_positive_and_symmetric_in_magnitude_for_a_known_shift():
    ref = [0.5, 0.5]
    batch = [0.1, 0.9]
    psi_forward = population_stability_index(ref, batch)
    psi_backward = population_stability_index(batch, ref)
    assert psi_forward > 0.25  # a large, known shift should clear the conventional threshold
    assert psi_forward == pytest.approx(psi_backward, rel=1e-6)  # PSI is symmetric by construction


def test_psi_handles_a_zero_proportion_bin_without_error():
    ref = [1.0, 0.0, 0.0]
    batch = [0.5, 0.25, 0.25]
    psi = population_stability_index(ref, batch)
    assert np.isfinite(psi)


# ---------------------------------------------------------------------------
# Reference distribution construction
# ---------------------------------------------------------------------------

def test_build_reference_distribution_on_a_normal_sample():
    rng = np.random.default_rng(42)
    values = pd.Series(rng.normal(loc=100, scale=20, size=1000))
    ref = build_reference_distribution(values, feature="synthetic", partition="train")
    assert ref["status"] == "ok"
    assert ref["reference_sample_size_finite"] == 1000
    assert len(ref["bin_proportions"]) >= 2
    assert pytest.approx(sum(ref["bin_proportions"]), abs=1e-9) == 1.0


def test_build_reference_distribution_too_small_is_flagged_not_silently_built():
    values = pd.Series(np.arange(MIN_REFERENCE_SAMPLE_SIZE - 1, dtype="float64"))
    ref = build_reference_distribution(values, feature="synthetic", partition="train")
    assert ref["status"] == "insufficient_reference_sample"
    assert ref["bin_edges"] is None


def test_build_reference_distribution_handles_missing_values_explicitly():
    values = pd.Series([1.0] * 300 + [np.nan] * 50)
    ref = build_reference_distribution(values, feature="synthetic", partition="train")
    assert ref["n_missing_or_non_finite"] == 50
    assert ref["reference_sample_size_finite"] == 300


def test_build_reference_distribution_handles_constant_feature():
    values = pd.Series([7.0] * (MIN_REFERENCE_SAMPLE_SIZE + 10))
    ref = build_reference_distribution(values, feature="synthetic", partition="train")
    assert ref["status"] == "constant_feature"
    assert ref["constant_value"] == 7.0


def test_build_reference_distribution_handles_non_finite_values():
    values = pd.Series([1.0, 2.0, np.inf, -np.inf] * 100)
    ref = build_reference_distribution(values, feature="synthetic", partition="train")
    assert ref["n_missing_or_non_finite"] == 200  # two inf values per 4-tuple, repeated 100x


def test_fingerprint_is_deterministic_for_identical_data():
    values = pd.Series(np.arange(500, dtype="float64"))
    ref1 = build_reference_distribution(values, feature="f", partition="train")
    ref2 = build_reference_distribution(values, feature="f", partition="train")
    assert ref1["population_fingerprint"] == ref2["population_fingerprint"]


# ---------------------------------------------------------------------------
# Batch scoring against a reference
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_reference():
    rng = np.random.default_rng(1)
    values = pd.Series(rng.normal(loc=50, scale=10, size=1000))
    return build_reference_distribution(values, feature="x", partition="train")


def test_score_batch_distribution_on_a_matching_batch(normal_reference):
    rng = np.random.default_rng(2)
    batch = pd.Series(rng.normal(loc=50, scale=10, size=200))
    result = score_batch_distribution(batch, normal_reference)
    assert result["status"] == "ok"
    assert pytest.approx(sum(result["bin_proportions"]), abs=1e-9) == 1.0


def test_score_batch_distribution_empty_batch_is_malformed_not_a_drift_alert(normal_reference):
    result = score_batch_distribution(pd.Series([], dtype="float64"), normal_reference)
    assert result["status"] == "malformed_input"


def test_score_batch_distribution_too_small_batch_is_insufficient(normal_reference):
    batch = pd.Series([1.0] * (MIN_BATCH_SAMPLE_SIZE - 1))
    result = score_batch_distribution(batch, normal_reference)
    assert result["status"] == "insufficient_batch_sample"


def test_score_batch_distribution_against_unusable_reference_is_flagged():
    bad_reference = {"feature": "x", "status": "insufficient_reference_sample"}
    batch = pd.Series(np.arange(100, dtype="float64"))
    result = score_batch_distribution(batch, bad_reference)
    assert result["status"] == "reference_unusable"


def test_score_batch_distribution_out_of_range_values_are_counted(normal_reference):
    # A batch entirely far outside the reference's observed range -- this must land in the
    # open-ended extreme bin, NOT be dropped as malformed, since this is itself the most
    # extreme possible drift signal.
    batch = pd.Series([10_000.0] * 100)
    result = score_batch_distribution(batch, normal_reference)
    assert result["n_out_of_reference_range"] == 100
    assert result["status"] == "ok"
    assert result["bin_proportions"][-1] == pytest.approx(1.0)  # all 100 values in the last, open-ended bin


# ---------------------------------------------------------------------------
# End-to-end input drift evaluation
# ---------------------------------------------------------------------------

def test_evaluate_input_drift_no_alert_on_a_control_batch_from_the_same_distribution(normal_reference):
    rng = np.random.default_rng(3)
    control_batch = pd.Series(rng.normal(loc=50, scale=10, size=500))
    result = evaluate_input_drift(normal_reference, control_batch)
    assert result["status"] == "ok"


def test_evaluate_input_drift_fires_on_a_deliberately_shifted_batch(normal_reference):
    rng = np.random.default_rng(4)
    shifted_batch = pd.Series(rng.normal(loc=50, scale=10, size=500) * 5.0 + 500)
    result = evaluate_input_drift(normal_reference, shifted_batch)
    assert result["status"] == "drift_alert"
    alert = result["alert"]
    for required_field in ("alert_id", "signal", "observed_statistic", "threshold", "sample_size",
                            "reference_provenance", "batch_provenance", "severity", "recommended_action", "detected_at"):
        assert required_field in alert
    assert alert["observed_statistic"] > alert["threshold"]


# ---------------------------------------------------------------------------
# Prediction (score) distribution monitoring
# ---------------------------------------------------------------------------

def test_build_reference_score_distribution_and_detect_no_drift_on_matching_scores():
    rng = np.random.default_rng(5)
    scores = rng.beta(2, 8, size=1000)  # right-skewed, like a fraud-score distribution
    ref = build_reference_score_distribution(scores, artifact_sha256="abc123")
    control_batch = rng.beta(2, 8, size=200)
    result = evaluate_score_drift(ref, control_batch, current_artifact_sha256="abc123")
    assert result["status"] == "ok"


def test_evaluate_score_drift_fires_on_a_shifted_score_batch():
    rng = np.random.default_rng(6)
    scores = rng.beta(2, 8, size=1000)
    ref = build_reference_score_distribution(scores, artifact_sha256="abc123")
    shifted = rng.beta(8, 2, size=300)  # flipped skew -- a real distributional shift
    result = evaluate_score_drift(ref, shifted, current_artifact_sha256="abc123")
    assert result["status"] == "drift_alert"
    assert "alert" in result


def test_evaluate_score_drift_refuses_mismatched_provenance():
    rng = np.random.default_rng(7)
    scores = rng.beta(2, 8, size=1000)
    ref = build_reference_score_distribution(scores, artifact_sha256="abc123")
    batch = rng.beta(2, 8, size=200)
    result = evaluate_score_drift(ref, batch, current_artifact_sha256="DIFFERENT_CHECKSUM")
    assert result["status"] == "provenance_mismatch"


def test_evaluate_score_drift_empty_batch_is_malformed():
    rng = np.random.default_rng(8)
    scores = rng.beta(2, 8, size=1000)
    ref = build_reference_score_distribution(scores, artifact_sha256="abc123")
    result = evaluate_score_drift(ref, np.array([]), current_artifact_sha256="abc123")
    assert result["status"] == "malformed_input"


# ---------------------------------------------------------------------------
# Operational log parsing (real fixtures, incl. actual Stage 17 log files)
# ---------------------------------------------------------------------------

def test_parse_log_file_skips_uvicorn_plain_text_lines(tmp_path):
    log_path = tmp_path / "test.log"
    log_path.write_text(
        'INFO:     Started server process [1]\n'
        '{"event": "startup", "artifact_loaded": true, "startup_seconds": 0.5}\n'
        'INFO:     127.0.0.1:1 - "GET /health HTTP/1.1" 200 OK\n'
        '{"request_id": "a", "status": 200, "duration_seconds": 0.05, "artifact_sha256": "abc"}\n'
    )
    parsed = parse_log_file(log_path)
    assert parsed["n_skipped_non_json"] == 2
    assert len(parsed["records"]) == 2


def test_parse_real_stage17_version_a_log_if_present():
    log_path = __file__.replace("tests/test_monitoring.py", "deploy/run/version_a.log")
    from pathlib import Path
    if not Path(log_path).exists():
        pytest.skip("Real Stage 17 version_a.log not present on this machine.")
    parsed = parse_log_file(log_path)
    assert any(r.get("status") == 200 for r in parsed["records"])


def test_summarize_operational_records_computes_error_rate_and_latency():
    records = [{"status": 200, "duration_seconds": 0.01 * i, "artifact_sha256": "abc"} for i in range(1, 21)]
    records += [{"status": 500, "duration_seconds": 0.02, "error_category": "X"}]
    summary = summarize_operational_records(records, expected_artifact_sha256="abc")
    assert summary["n_records_with_status_and_duration"] == 21
    assert summary["n_5xx_server_error"] == 1
    assert summary["error_rate_5xx"] == pytest.approx(1 / 21)
    assert summary["latency_insufficient_sample"] is False if "latency_insufficient_sample" in summary else True


def test_summarize_operational_records_flags_insufficient_latency_sample_when_small():
    records = [{"status": 200, "duration_seconds": 0.01}]
    summary = summarize_operational_records(records)
    assert summary["latency_insufficient_sample"] is True
    assert summary["latency_median_seconds"] is None


def test_summarize_operational_records_detects_artifact_mismatch():
    records = [{"status": 200, "duration_seconds": 0.01, "artifact_sha256": "WRONG"}] * 25
    summary = summarize_operational_records(records, expected_artifact_sha256="EXPECTED")
    assert summary["artifact_sha256_mismatches"] == ["WRONG"]


def test_summarize_operational_records_empty_list_does_not_crash():
    summary = summarize_operational_records([])
    assert summary["n_records_with_status_and_duration"] == 0
    assert summary["error_rate_5xx"] is None


def test_evaluate_operational_health_fires_on_any_server_error():
    summary = summarize_operational_records([{"status": 500, "duration_seconds": 0.01}])
    alerts = evaluate_operational_health(summary, expected_artifact_sha256="EXPECTED")
    assert any(a["alert_id"] == "operational_server_error" for a in alerts)


def test_evaluate_operational_health_fires_on_artifact_mismatch():
    records = [{"status": 200, "duration_seconds": 0.01, "artifact_sha256": "WRONG"}] * 5
    summary = summarize_operational_records(records, expected_artifact_sha256="EXPECTED")
    alerts = evaluate_operational_health(summary, expected_artifact_sha256="EXPECTED")
    assert any(a["alert_id"] == "operational_artifact_identity_mismatch" for a in alerts)


def test_evaluate_operational_health_no_alerts_on_a_clean_window():
    records = [{"status": 200, "duration_seconds": 0.01, "artifact_sha256": "EXPECTED"} for _ in range(25)]
    summary = summarize_operational_records(records, expected_artifact_sha256="EXPECTED")
    alerts = evaluate_operational_health(summary, expected_artifact_sha256="EXPECTED")
    assert alerts == []


def test_config_file_is_valid_json_and_has_required_rule_fields():
    from src.monitoring import MONITORING_CONFIG_PATH
    with open(MONITORING_CONFIG_PATH) as f:
        config = json.load(f)
    required = {"rule_id", "signal", "calculation", "required_sample_size", "reference_source",
                "threshold", "threshold_rationale", "expected_false_alert_limitations", "operator_response"}
    for rule in config["rules"]:
        missing = required - set(rule.keys())
        assert not missing, f"rule {rule.get('rule_id')} missing fields: {missing}"
