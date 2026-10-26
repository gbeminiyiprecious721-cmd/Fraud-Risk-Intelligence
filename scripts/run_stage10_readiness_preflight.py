"""Stage 10 — readiness-gate PREFLIGHT rehearsal.

Runs the ENTIRE evaluation pipeline for real -- feature alignment,
preprocessing, label alignment, AP/ranking metrics, segmentation,
bootstrap CI, cost-threshold sweep, artifact serialization -- end to end,
using `train` (to fit) and `validation` (to score) ONLY. `final_test` is
never read anywhere in this script; the only `allow_final_test` call site
in this project is the separately-gated `run_stage10_final_evaluation.py`,
never this one.

This is explicitly a REHEARSAL, not the real Stage 10 result: scoring on
`validation` here proves the pipeline works, but `validation` is not a
clean, never-consulted signal for this project (Stage 7 touched it 15
times) and this rehearsal's own numbers are NOT the project's Stage 10
evaluation -- they are preflight evidence only, logged under a distinct,
clearly-named search_id, never under `FINAL_EVALUATION_SEARCH_ID`.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.preprocessing import build_preprocessing_pipeline, load_included_columns, load_partition_df, measure_train_missingness, select_indicator_columns, DEFAULT_INDICATOR_THRESHOLD  # noqa: E402
from src.model_selection import align_labels_to_partition_df, load_labels, no_skill_baseline  # noqa: E402
from src.model_evaluation import (  # noqa: E402
    build_frozen_model,
    bootstrap_ap_ci,
    compare_to_no_skill_baseline,
    compute_amount_segment_boundaries,
    assign_amount_segment,
    segment_report,
    sweep_cost_minimizing_thresholds,
    verify_frozen_config_unchanged,
    FROZEN_CONFIG_FINGERPRINT,
)
from src.experiment_tracking import track_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

PREFLIGHT_SEARCH_ID = "stage10-readiness-preflight-rehearsal"


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {"search_id": PREFLIGHT_SEARCH_ID, "note": "REHEARSAL on validation -- NOT the real Stage 10 final_test evaluation."}

    with track_run(10, "stage10_readiness_preflight", "Full pipeline rehearsal on validation (not final_test) -- proves readiness only.", search_id=PREFLIGHT_SEARCH_ID) as record:
        section("0. final_test refusal re-check + frozen-config integrity")
        assignment = pd.read_csv(ASSIGNMENT_PATH)
        try:
            get_partition(assignment, ["train", "final_test"])
            report["final_test_refused"] = False
        except FinalTestAccessError as exc:
            print(f"Correctly refused: {exc}")
            report["final_test_refused"] = True
        config_ok = verify_frozen_config_unchanged()
        print(f"Frozen config fingerprint check: {config_ok} ({FROZEN_CONFIG_FINGERPRINT[:16]}...)")
        report["frozen_config_fingerprint_ok"] = config_ok
        assert config_ok

        section("1. Feature schema (approved Stage 5 structure, unmodified) + label alignment")
        columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
        numeric_columns, categorical_columns = columns["numeric"], columns["categorical"]
        missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, numeric_columns)
        indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
        all_cols = indicator_split["without_indicator"] + indicator_split["with_indicator"] + categorical_columns

        train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols + ["TransactionAmt", "ProductCD"])
        train_labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
        train_df = align_labels_to_partition_df(train_df, train_labels)

        val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols + ["TransactionAmt", "ProductCD"])
        val_labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")
        val_df = align_labels_to_partition_df(val_df, val_labels)
        print(f"train: {len(train_df)} rows, validation: {len(val_df)} rows -- label alignment verified by align_labels_to_partition_df's own one_to_one merge check.")
        report["train_n_rows"] = len(train_df)
        report["validation_n_rows"] = len(val_df)

        section("2. Fit approved Stage 5 preprocessing (train-only) + the FROZEN model on train")
        ct = build_preprocessing_pipeline(indicator_split["without_indicator"], indicator_split["with_indicator"], categorical_columns)
        t0 = time.perf_counter()
        X_train = ct.fit_transform(train_df[all_cols])
        preprocessing_seconds = time.perf_counter() - t0
        X_val = ct.transform(val_df[all_cols])
        print(f"preprocessing_seconds={preprocessing_seconds:.3f}  fitted_feature_count={X_train.shape[1]}  (train) vs {X_val.shape[1]} (validation, SAME fitted transform -- exact schema match required)")
        assert X_train.shape[1] == X_val.shape[1], "Feature schema mismatch between train-fitted transform and its own validation output -- should be impossible, flag immediately if seen."
        report["fitted_feature_count"] = X_train.shape[1]

        model = build_frozen_model()
        t0 = time.perf_counter()
        model.fit(X_train, train_df["isFraud"].to_numpy())
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        val_scores = model.predict_proba(X_val)[:, 1]
        predict_seconds = time.perf_counter() - t0
        print(f"fit_seconds={fit_seconds:.3f}  predict_seconds={predict_seconds:.3f}")
        record["model_family"] = "RandomForestClassifier"
        record["hyperparameters"] = model.get_params()
        record["feature_count"] = X_train.shape[1]
        record["training_time_seconds"] = round(fit_seconds, 3)
        record["inference_time_seconds"] = round(predict_seconds, 3)

        section("3. Artifact serialization round-trip (joblib) -- proves the fitted model can be persisted and reloaded identically")
        with tempfile.TemporaryDirectory() as tmp:
            # joblib.load here only ever reads a file this same process just
            # wrote, in its own private temp dir -- never an external or
            # user-supplied artifact, so the usual pickle-deserialization
            # risk from untrusted sources does not apply.
            artifact_path = Path(tmp) / "frozen_model_rehearsal.joblib"
            joblib.dump(model, artifact_path)
            reloaded = joblib.load(artifact_path)
            reloaded_scores = reloaded.predict_proba(X_val)[:, 1]
            import numpy as np
            # Exact equality is the WRONG check here, per Stage 7's own
            # documented finding: RandomForestClassifier(n_jobs=-1)'s
            # predict_proba involves cross-thread summation whose order
            # (and therefore its machine-epsilon-level rounding) is not
            # guaranteed identical between two separate calls, even on
            # identical fitted trees -- not a serialization bug.
            serialization_ok = bool(np.allclose(val_scores, reloaded_scores, rtol=1e-9, atol=1e-12))
            max_abs_diff = float(np.max(np.abs(val_scores - reloaded_scores)))
            print(f"Serialize -> reload -> predictions match within tolerance: {serialization_ok} (max abs diff: {max_abs_diff:.2e})")
            report["serialization_round_trip_ok"] = serialization_ok
            report["serialization_max_abs_diff"] = max_abs_diff
            assert serialization_ok

        section("4. Evaluation metrics + Stage 1 success-criterion #2 (beat the no-skill baseline)")
        from src.model_selection import evaluate_predictions
        y_val = val_df["isFraud"].to_numpy()
        model_metrics = evaluate_predictions(y_val, val_scores)
        baseline = no_skill_baseline(train_df["isFraud"].to_numpy(), y_val)
        comparison = compare_to_no_skill_baseline(model_metrics, baseline)
        print(f"Model AP={model_metrics['average_precision']}  No-skill AP={baseline['average_precision']}  {comparison}")
        report["model_metrics"] = model_metrics
        report["no_skill_baseline"] = baseline
        report["stage1_baseline_comparison"] = comparison
        record["metrics"] = model_metrics
        record["validation_consulted"] = True
        record["validation_purpose"] = "REHEARSAL ONLY -- proves pipeline correctness, is NOT the project's Stage 10 final_test evaluation and must never be reported as such"

        section("5. Segmentation -- ProductCD and TransactionAmt, boundaries from TRAIN only")
        productcd_report = segment_report(y_val, val_scores, val_df["ProductCD"].to_numpy())
        amount_boundaries = compute_amount_segment_boundaries(train_df["TransactionAmt"].to_numpy())
        amount_segments = assign_amount_segment(val_df["TransactionAmt"].to_numpy(), amount_boundaries)
        amount_report = segment_report(y_val, val_scores, amount_segments)
        print(f"ProductCD segments: {list(productcd_report.keys())}")
        print(f"TransactionAmt boundaries (from TRAIN only): {amount_boundaries}")
        for seg, entry in productcd_report.items():
            print(f"  ProductCD={seg}: n={entry['n_samples']} n_pos={entry['n_positive']} sufficient={entry['sufficient_sample']}" + (f" AP={entry['metrics']['average_precision']}" if entry['sufficient_sample'] else ""))
        report["productcd_segment_report"] = productcd_report
        report["amount_segment_boundaries"] = amount_boundaries
        report["amount_segment_report"] = amount_report

        section("6. Bootstrap confidence interval on AP")
        ci = bootstrap_ap_ci(y_val, val_scores)
        print(f"AP point estimate {ci['point_estimate']:.5f}, {int(ci['ci_level']*100)}% CI [{ci['lower']:.5f}, {ci['upper']:.5f}] ({ci['n_resamples']} resamples, seed={ci['seed']})")
        report["bootstrap_ci"] = ci

        section("7. Cost-sensitive threshold sweep (Stage 1's own framework, swept not invented)")
        cost_sweep = sweep_cost_minimizing_thresholds(y_val, val_scores, val_df["TransactionAmt"].to_numpy())
        for scenario, by_assumption in cost_sweep.items():
            for assumption, res in by_assumption.items():
                print(f"  {scenario} / {assumption}: best_threshold={res['best_threshold']}  best_cost=${res['best_cost']:.2f}")
        report["cost_threshold_sweep"] = cost_sweep

        record["evidence_paths"] = []

    out_path = DOCS_DIR / "_stage10_readiness_preflight_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured preflight report written to {out_path}")
    print("\nREMINDER: this is a REHEARSAL on validation. It does NOT constitute, and must never be reported as, the project's Stage 10 final_test evaluation.")
    return report


if __name__ == "__main__":
    main()
