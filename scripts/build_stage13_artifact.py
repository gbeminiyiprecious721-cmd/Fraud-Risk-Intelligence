"""Stage 13 -- build the ONE versioned production artifact.

Fits the frozen Stage 9/10 Random Forest configuration + the approved
Stage 5 preprocessing ONCE, on the complete `train` partition (the same
effective training procedure Stage 10 used to obtain the model whose
AP=0.42784 was reported against `final_test`). Persists the result via
`src.production_pipeline.save_artifact`.

**This is a RECONSTRUCTION, not a recovery of Stage 10's original
fitted Python object** -- that object was never serialized anywhere
and no longer exists in memory. This script builds a NEW fit of the
identical frozen configuration on the identical training partition;
Stage 13's regression test (`tests/test_production_pipeline.py`)
independently verifies that this reconstruction is equivalent (same
config, same ordered feature schema, `np.allclose` predictions) to a
freshly-fit reference built the same way -- NOT that it is bit-for-bit
Stage 10's original object, which is not a claim this script can or
does make. Stage 10's historical metrics (AP 0.42784 on `final_test`)
are referenced below for context only and are NOT re-measured or
re-claimed as a new evaluation of this artifact.

TRAIN-ONLY. Never reads `validation` or `final_test`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.model_evaluation import FROZEN_RF_CONFIG, verify_frozen_config_unchanged  # noqa: E402
from src.model_selection import align_labels_to_partition_df, load_labels  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    load_included_columns,
    load_partition_df,
    measure_train_missingness,
    select_indicator_columns,
)
from src.production_pipeline import (  # noqa: E402
    environment_compatibility_info,
    fit_production_pipeline,
    git_provenance_info,
    save_artifact,
)
from src.experiment_tracking import track_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

SEARCH_ID = "stage13-production-pipeline"
ARTIFACT_ID = "frozen-rf-fraud-risk"
ARTIFACT_VERSION = "1.0.0"

STAGE9_REFERENCE = "Stage 9 (docs/STAGE9_TUNING_DESIGN_REPORT.md): candidate_index=4, mean chronological-CV AP 0.51236, audited and selected as the provisional tuned configuration."
STAGE10_REFERENCE = "Stage 10 (docs/STAGE10_EVALUATION_PROTOCOL.md): the frozen configuration's one authorised final_test evaluation, AP 0.42784, ROC-AUC 0.86637, 95% CI [0.40842, 0.44534] -- NOT re-measured or re-claimed by this artifact."


def fingerprint_ids(ids: pd.Series) -> str:
    """Identical logic to the `fingerprint_ids` helper already used in
    `scripts/run_stage9_hyperparameter_search.py` -- a sha256 of the
    sorted TransactionID list, for consistency with that precedent."""
    joined = "\x1f".join(str(i) for i in sorted(ids.tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}", flush=True)


def main() -> dict:
    section("0. final_test refusal + frozen-config integrity")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    try:
        get_partition(assignment, ["train", "final_test"])
        final_test_refused = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        final_test_refused = True
    assert verify_frozen_config_unchanged()

    with track_run(13, "stage13_build_production_artifact", "Fit and persist the ONE versioned Stage 13 production artifact (frozen RF + approved preprocessing), train-only.", search_id=SEARCH_ID) as record:
        section("1. Load the complete train partition (same procedure as Stage 10)")
        columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
        numeric_columns, categorical_columns = columns["numeric"], columns["categorical"]
        missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, numeric_columns)
        indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)

        train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", indicator_split["without_indicator"] + indicator_split["with_indicator"] + categorical_columns)
        labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
        train_df = align_labels_to_partition_df(train_df, labels)
        training_fingerprint = fingerprint_ids(train_df["TransactionID"])
        print(f"train: {len(train_df)} rows, fingerprint={training_fingerprint[:16]}...")

        section("2. Fit the production pipeline ONCE (this script's only fit)")
        t0 = time.perf_counter()
        artifact = fit_production_pipeline(train_df, indicator_split["without_indicator"], indicator_split["with_indicator"], categorical_columns)
        fit_seconds = time.perf_counter() - t0
        print(f"fit_seconds={fit_seconds:.3f}  fitted_feature_count={len(artifact.schema.transformed_feature_names)}")
        record["model_family"] = "RandomForestClassifier"
        record["hyperparameters"] = FROZEN_RF_CONFIG
        record["feature_count"] = len(artifact.schema.transformed_feature_names)
        record["training_time_seconds"] = round(fit_seconds, 3)

        section("3. Persist the artifact + manifest")
        git_info = git_provenance_info()
        commit, dirty = git_info["git_commit"], git_info["git_dirty"]
        metadata = {
            "artifact_id": ARTIFACT_ID,
            "artifact_version": ARTIFACT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frozen_estimator_config": FROZEN_RF_CONFIG,
            "training_partition": {"name": "train", "n_rows": len(train_df), "row_fingerprint": training_fingerprint},
            "git_commit": commit,
            "git_dirty": dirty,
            "environment": environment_compatibility_info(),
            "raw_feature_schema": list(artifact.schema.raw_columns),
            "numeric_feature_schema": list(artifact.schema.numeric_columns),
            "transformed_feature_schema": list(artifact.schema.transformed_feature_names),
            "fit_seconds": round(fit_seconds, 3),
            "stage9_reference": STAGE9_REFERENCE,
            "stage10_reference": STAGE10_REFERENCE,
            "reconstruction_disclaimer": (
                "This artifact is a RECONSTRUCTION of the frozen Stage 9/10 configuration, fit fresh on the "
                "same training partition -- NOT the identical historical Python object Stage 10 evaluated "
                "(that object was never serialized and no longer exists). Equivalence to an independently-"
                "fit reference is verified by tests/test_production_pipeline.py's full-training regression "
                "test, not assumed."
            ),
            "final_test_accessed": False,
        }
        t0 = time.perf_counter()
        artifact_path = ARTIFACTS_DIR / "frozen_rf_fraud_risk_v1.joblib"
        manifest = save_artifact(artifact, artifact_path, metadata)
        save_seconds = time.perf_counter() - t0
        print(f"save_seconds={save_seconds:.3f}  artifact_size_bytes={manifest['artifact_size_bytes']}  sha256={manifest['artifact_sha256'][:16]}...")

        section("4. Reload and verify the save/load round-trip (load timing + prediction equivalence)")
        from src.production_pipeline import load_artifact, predict_fraud_probability

        t0 = time.perf_counter()
        reloaded_artifact, reloaded_manifest = load_artifact(artifact_path)
        load_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        proba_before = predict_fraud_probability(artifact, train_df.iloc[:1000])
        inference_seconds_1000_rows = time.perf_counter() - t0
        proba_after = predict_fraud_probability(reloaded_artifact, train_df.iloc[:1000])
        import numpy as np
        predictions_match = bool(np.allclose(proba_before, proba_after))
        print(f"load_seconds={load_seconds:.3f}  inference_seconds_for_1000_rows={inference_seconds_1000_rows:.3f}  predictions_match_after_reload={predictions_match}")
        assert predictions_match, "Save/load round-trip predictions diverged -- stopping, not forcing a pass."

        record["metrics"] = {
            "fit_seconds": round(fit_seconds, 3),
            "save_seconds": round(save_seconds, 3),
            "load_seconds": round(load_seconds, 3),
            "inference_seconds_for_1000_rows": round(inference_seconds_1000_rows, 3),
            "artifact_size_bytes": manifest["artifact_size_bytes"],
            "predictions_match_after_reload": predictions_match,
        }
        record["validation_consulted"] = False

    report = {
        "manifest": manifest,
        "final_test_refused": final_test_refused,
        "fit_seconds": fit_seconds,
        "save_seconds": save_seconds,
        "load_seconds": load_seconds,
        "inference_seconds_for_1000_rows": inference_seconds_1000_rows,
        "predictions_match_after_reload": predictions_match,
    }
    out_path = DOCS_DIR / "_stage13_artifact_build_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
