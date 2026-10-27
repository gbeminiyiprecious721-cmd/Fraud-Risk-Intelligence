"""Stage 18 -- real local monitoring demonstration.

Builds real reference distributions from the approved `train` partition
and the real, trusted Stage 13 artifact (no refit anywhere), then runs
a genuine control-vs-drift demonstration for both input-distribution
and prediction-score monitoring, plus an operational-monitoring
demonstration using a REAL fresh local deployment (control, via the
existing Stage 17 `deploy/` scripts) and REAL existing Stage 17 failure
evidence (`deploy/run/version_b.log`, falling back to a documented
reproduction of that exact, already-reported real log content if the
file is not present on this machine).

Never accesses `final_test`. Never retrains, retunes, or recalibrates
the model. No new deployment service is created -- the operational
control demonstration reuses Stage 17's existing `start_release.sh`/
`stop_release.sh`/`health_check.sh` scripts exactly as they already are.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.monitoring import (  # noqa: E402
    MONITORING_DIR,
    REFERENCE_DISTRIBUTIONS_PATH,
    REFERENCE_SCORE_DISTRIBUTION_PATH,
    MONITORED_INPUT_FEATURES,
    build_reference_distribution,
    build_reference_score_distribution,
    evaluate_input_drift,
    evaluate_operational_health,
    evaluate_score_drift,
    load_monitoring_config,
    parse_log_file,
    save_json,
    summarize_operational_records,
)
from src.preprocessing import load_partition_df  # noqa: E402
from src.production_pipeline import load_artifact, predict_fraud_probability  # noqa: E402
from src.api import DEFAULT_ARTIFACT_PATH  # noqa: E402

ASSIGNMENT_PATH = ROOT / "docs" / "_stage4_split_assignment.csv"
TRAIN_TXN_PATH = ROOT / "data" / "raw" / "ieee-fraud-detection" / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
DEPLOY_DIR = ROOT / "deploy"

# Reproduces the EXACT, already-reported real log content from Stage 17 (docs/_stage17_deployment_demo_raw.json
# / docs/STAGE17_DEPLOYMENT_REPORT.md Section 7) -- used ONLY if deploy/run/version_b.log is no longer present
# on this machine. Not an invented fixture; a verbatim reproduction of already-approved real evidence.
STAGE17_VERSION_B_LOG_REPRODUCTION = (
    'INFO:     Started server process [30145]\n'
    'INFO:     Waiting for application startup.\n'
    '{"event": "startup", "artifact_loaded": false, "error_category": "ArtifactIntegrityError"}\n'
    'INFO:     Application startup complete.\n'
    'INFO:     Uvicorn running on http://127.0.0.1:53338 (Press CTRL+C to quit)\n'
    '{"request_id": "8eb68690-3faa-4bb1-85e9-665fc4a6ed11", "api_contract_version": "v1", "status": 503, '
    '"error_category": "ArtifactUnavailable", "duration_seconds": 0.0}\n'
    'INFO:     127.0.0.1:53369 - "POST /v1/predict HTTP/1.1" 503 Service Unavailable\n'
)


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}", flush=True)


def find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> dict:
    report: dict = {}
    artifact, manifest = load_artifact(DEFAULT_ARTIFACT_PATH)
    config = load_monitoring_config()

    # --- 1. Build real reference distributions for the monitored input features ---
    section("1. Build reference input distributions from the real train partition")
    reference_distributions = {}
    train_features_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN_PATH, "train", list(MONITORED_INPUT_FEATURES))
    for feature in MONITORED_INPUT_FEATURES:
        ref = build_reference_distribution(train_features_df[feature], feature=feature, partition="train")
        reference_distributions[feature] = ref
        print(f"{feature}: status={ref['status']} n_finite={ref.get('reference_sample_size_finite')} "
              f"n_missing={ref.get('n_missing_or_non_finite')}")
        assert ref["status"] in ("ok", "constant_feature"), f"Unexpected reference status for {feature}"
    save_json(REFERENCE_DISTRIBUTIONS_PATH, reference_distributions)
    report["reference_distributions_summary"] = {
        f: {"status": reference_distributions[f]["status"],
            "reference_sample_size_finite": reference_distributions[f]["reference_sample_size_finite"],
            "n_missing_or_non_finite": reference_distributions[f]["n_missing_or_non_finite"]}
        for f in MONITORED_INPUT_FEATURES
    }

    # --- 2. Build a real reference score distribution from the trusted artifact ---
    section("2. Build reference prediction-score distribution (real artifact, no refit)")
    reference_rows = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN_PATH, "train", list(artifact.schema.raw_columns)).sample(
        n=1000, random_state=10
    )
    reference_scores = predict_fraud_probability(artifact, reference_rows[list(artifact.schema.raw_columns)])
    score_reference = build_reference_score_distribution(reference_scores, artifact_sha256=manifest["artifact_sha256"])
    save_json(REFERENCE_SCORE_DISTRIBUTION_PATH, score_reference)
    print(f"score reference: status={score_reference['status']} n={score_reference['reference_sample_size']} "
          f"mean={score_reference.get('mean')}")
    assert score_reference["status"] == "ok"
    report["reference_score_distribution_summary"] = {
        "status": score_reference["status"], "reference_sample_size": score_reference["reference_sample_size"],
        "mean": score_reference["mean"], "artifact_sha256": score_reference["artifact_sha256"],
    }

    # --- 3. Control batch: independent real sample, same distribution -- expect NO alert ---
    section("3. Control batch (independent real train sample) -- expect no drift alert")
    control_rows = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN_PATH, "train", list(artifact.schema.raw_columns)).sample(
        n=500, random_state=99
    )
    control_results = {}
    for feature in MONITORED_INPUT_FEATURES:
        result = evaluate_input_drift(reference_distributions[feature], control_rows[feature])
        control_results[feature] = {"status": result["status"], "psi": result.get("psi")}
        print(f"control {feature}: status={result['status']} psi={result.get('psi')}")
        assert result["status"] == "ok", f"Control batch unexpectedly alerted for {feature}"
    control_scores = predict_fraud_probability(artifact, control_rows[list(artifact.schema.raw_columns)])
    control_score_result = evaluate_score_drift(score_reference, control_scores, current_artifact_sha256=manifest["artifact_sha256"])
    print(f"control score distribution: status={control_score_result['status']} psi={control_score_result.get('psi')}")
    assert control_score_result["status"] == "ok", "Control batch unexpectedly alerted on prediction-score drift"
    report["control_batch"] = {"input_drift": control_results, "score_drift": {"status": control_score_result["status"], "psi": control_score_result.get("psi")}}

    # --- 4. Deliberately shifted batch -- expect the drift mechanism to actually fire ---
    section("4. Deliberately shifted batch (TransactionAmt x20) -- expect a real drift alert")
    shifted_rows = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN_PATH, "train", list(artifact.schema.raw_columns)).sample(
        n=500, random_state=7
    ).copy()
    shifted_rows["TransactionAmt"] = shifted_rows["TransactionAmt"] * 20.0  # one deliberate, documented defect
    drift_result = evaluate_input_drift(reference_distributions["TransactionAmt"], shifted_rows["TransactionAmt"])
    print(f"shifted TransactionAmt: status={drift_result['status']} psi={drift_result.get('psi')}")
    assert drift_result["status"] == "drift_alert", "The deliberately shifted batch did not raise the expected input-drift alert"
    input_alert = drift_result["alert"]
    for required in ("alert_id", "signal", "observed_statistic", "threshold", "sample_size",
                      "reference_provenance", "batch_provenance", "severity", "recommended_action", "detected_at"):
        assert required in input_alert, f"alert record missing required field {required}"

    shifted_scores = predict_fraud_probability(artifact, shifted_rows[list(artifact.schema.raw_columns)])
    shifted_score_result = evaluate_score_drift(score_reference, shifted_scores, current_artifact_sha256=manifest["artifact_sha256"])
    print(f"shifted-batch score distribution: status={shifted_score_result['status']} psi={shifted_score_result.get('psi')}")
    # Reported truthfully either way -- a feature shift does not always guarantee a score-distribution
    # shift large enough to cross the PSI threshold; this is itself a real, informative outcome to document.
    report["drift_demonstration"] = {
        "input_drift_alert": input_alert,
        "score_drift_result": {"status": shifted_score_result["status"], "psi": shifted_score_result.get("psi")},
    }

    # --- 5. Operational monitoring: a real, fresh control run + real Stage 17 failure evidence ---
    section("5. Operational monitoring -- real control run (fresh Version A) + real Stage 17 failure evidence")
    port = find_free_port()
    control_log_path = DEPLOY_DIR / "run" / "monitoring_control.log"
    r = subprocess.run([str(DEPLOY_DIR / "start_release.sh"), "monitoring_control", str(port)], cwd=str(ROOT), capture_output=True, text=True)
    print(r.stdout.strip())
    assert r.returncode == 0
    h = subprocess.run([str(DEPLOY_DIR / "health_check.sh"), str(port), "20"], cwd=str(ROOT), capture_output=True, text=True)
    assert h.returncode == 0, "Fresh control release did not become ready"

    base_url = f"http://127.0.0.1:{port}"
    valid_payload = {}
    for col in artifact.schema.raw_columns:
        if col in artifact.schema.numeric_columns:
            valid_payload[col] = 100.0
        elif col == "ProductCD":
            valid_payload[col] = "W"
        elif col == "card4":
            valid_payload[col] = "visa"
        elif col == "card6":
            valid_payload[col] = "debit"
        else:
            valid_payload[col] = "unknown"
    valid_payload["TransactionAmt"] = 125.0
    for _ in range(25):
        resp = httpx.post(f"{base_url}/v1/predict", json=valid_payload, timeout=5.0)
        assert resp.status_code == 200
    for _ in range(3):
        httpx.get(f"{base_url}/health", timeout=5.0)

    stop = subprocess.run([str(DEPLOY_DIR / "stop_release.sh"), "monitoring_control"], cwd=str(ROOT), capture_output=True, text=True)
    print(stop.stdout.strip())
    assert stop.returncode == 0

    control_log_source = DEPLOY_DIR / "run" / "monitoring_control.log"
    control_parsed = parse_log_file(control_log_source)
    control_summary = summarize_operational_records(control_parsed["records"], expected_artifact_sha256=manifest["artifact_sha256"])
    control_alerts = evaluate_operational_health(control_summary, expected_artifact_sha256=manifest["artifact_sha256"])
    print(f"control operational summary: {control_summary}")
    assert control_alerts == [], f"Fresh, healthy control run unexpectedly raised alerts: {control_alerts}"

    version_b_log_path = DEPLOY_DIR / "run" / "version_b.log"
    used_reproduction = False
    if not version_b_log_path.exists():
        used_reproduction = True
        version_b_log_path = ROOT / "deploy" / "run" / "_stage17_version_b_log_reproduction.log"
        version_b_log_path.parent.mkdir(parents=True, exist_ok=True)
        version_b_log_path.write_text(STAGE17_VERSION_B_LOG_REPRODUCTION)
    failure_parsed = parse_log_file(version_b_log_path)
    failure_summary = summarize_operational_records(failure_parsed["records"], expected_artifact_sha256=manifest["artifact_sha256"])
    failure_alerts = evaluate_operational_health(failure_summary, expected_artifact_sha256=manifest["artifact_sha256"])
    print(f"failure operational summary: {failure_summary}")
    print(f"failure alerts: {[a['alert_id'] for a in failure_alerts]}")
    assert any(a["alert_id"] == "operational_server_error" for a in failure_alerts), \
        "Real Stage 17 failure evidence did not raise the expected operational_server_error alert"

    report["operational_monitoring"] = {
        "control": {"summary": control_summary, "alerts": control_alerts},
        "failure": {
            "source": "real deploy/run/version_b.log" if not used_reproduction else "reproduction of already-reported real Stage 17 log content (original file no longer present on this machine)",
            "summary": failure_summary,
            "alerts": failure_alerts,
        },
    }

    out_path = DOCS_DIR / "_stage18_monitoring_demo_raw.json"
    save_json(out_path, report)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
