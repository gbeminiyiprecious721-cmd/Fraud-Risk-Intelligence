"""Stage 10 — the ONE-TIME final_test evaluation.

THIS SCRIPT IS BUILT BUT NOT APPROVED TO RUN YET. It requires an explicit
CLI argument carrying the exact unlock token, and even then refuses if a
completed evaluation already exists under `FINAL_EVALUATION_SEARCH_ID`.

Call order, deliberately structured so no final_test-touching code can
execute before BOTH gates pass:
    1. Parse CLI args (no I/O).
    2. `require_unlock_token(token)` -- cheap, no dataset access at all.
    3. `check_final_test_not_already_evaluated()` -- reads only the
       (dataset-free) experiment registry, never a data file.
    4. ONLY THEN: the imports and calls below this point in `main()` that
       actually touch `final_test` via `allow_final_test=True`.

Once authorised, this script performs the scoring and persists evidence
in ONE controlled operation -- it does not loop, retry, or offer a way to
re-run under a different configuration without a brand-new, separately
approved `search_id` chosen by the user.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_evaluation import (  # noqa: E402
    FINAL_EVALUATION_SEARCH_ID,
    REQUIRED_UNLOCK_TOKEN,
    bootstrap_ap_ci,
    build_frozen_model,
    check_final_test_not_already_evaluated,
    compare_to_no_skill_baseline,
    compute_amount_segment_boundaries,
    assign_amount_segment,
    require_unlock_token,
    segment_report,
    sweep_cost_minimizing_thresholds,
    verify_frozen_config_unchanged,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main(token: str) -> dict:
    # --- GATE 1: token check. No data access above or before this line. ---
    require_unlock_token(token)
    # --- GATE 2: registry duplicate check. Still no data access. ---
    check_final_test_not_already_evaluated()

    # Only past both gates do we import anything capable of reading
    # final_test, and only now do we track the run (so a refusal at either
    # gate above never creates a registry record at all -- a refused
    # attempt is not an experiment).
    from src.leakage_boundaries import get_partition  # noqa: E402 (deliberately deferred import)
    from src.preprocessing import build_preprocessing_pipeline, load_included_columns, load_partition_df, measure_train_missingness, select_indicator_columns, DEFAULT_INDICATOR_THRESHOLD  # noqa: E402
    from src.model_selection import align_labels_to_partition_df, load_labels, no_skill_baseline, evaluate_predictions  # noqa: E402
    from src.experiment_tracking import track_run  # noqa: E402

    report: dict = {"search_id": FINAL_EVALUATION_SEARCH_ID}

    with track_run(10, "stage10_final_test_evaluation", "THE one-time Stage 10 evaluation on final_test.", search_id=FINAL_EVALUATION_SEARCH_ID) as record:
        section("0. Frozen-config integrity")
        assert verify_frozen_config_unchanged()

        section("1. Load train (fit) + final_test (score, ONE TIME) under the approved Stage 5 pipeline")
        columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
        numeric_columns, categorical_columns = columns["numeric"], columns["categorical"]
        missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, numeric_columns)
        indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
        all_cols = indicator_split["without_indicator"] + indicator_split["with_indicator"] + categorical_columns

        train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols + ["TransactionAmt", "ProductCD"])
        train_labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
        train_df = align_labels_to_partition_df(train_df, train_labels)

        # The ONLY allow_final_test=True call sites in this entire project.
        assignment = pd.read_csv(ASSIGNMENT_PATH)
        final_test_ids = set(get_partition(assignment, ["final_test"], allow_final_test=True)["TransactionID"])
        test_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "final_test", all_cols + ["TransactionAmt", "ProductCD"], allow_final_test=True)
        test_labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "final_test", allow_final_test=True)
        test_df = align_labels_to_partition_df(test_df, test_labels)
        assert set(test_df["TransactionID"]) == final_test_ids

        section("2. Fit approved Stage 5 preprocessing (train-only) + the FROZEN model")
        ct = build_preprocessing_pipeline(indicator_split["without_indicator"], indicator_split["with_indicator"], categorical_columns)
        t0 = time.perf_counter()
        X_train = ct.fit_transform(train_df[all_cols])
        preprocessing_seconds = time.perf_counter() - t0
        X_test = ct.transform(test_df[all_cols])

        model = build_frozen_model()
        t0 = time.perf_counter()
        model.fit(X_train, train_df["isFraud"].to_numpy())
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        test_scores = model.predict_proba(X_test)[:, 1]
        predict_seconds = time.perf_counter() - t0

        section("3. THE evaluation -- primary/secondary metrics, segments, bootstrap CI, cost sweep")
        y_test = test_df["isFraud"].to_numpy()
        model_metrics = evaluate_predictions(y_test, test_scores)
        baseline = no_skill_baseline(train_df["isFraud"].to_numpy(), y_test)
        comparison = compare_to_no_skill_baseline(model_metrics, baseline)
        productcd_report = segment_report(y_test, test_scores, test_df["ProductCD"].to_numpy())
        amount_boundaries = compute_amount_segment_boundaries(train_df["TransactionAmt"].to_numpy())
        amount_segments = assign_amount_segment(test_df["TransactionAmt"].to_numpy(), amount_boundaries)
        amount_report = segment_report(y_test, test_scores, amount_segments)
        ci = bootstrap_ap_ci(y_test, test_scores)
        cost_sweep = sweep_cost_minimizing_thresholds(y_test, test_scores, test_df["TransactionAmt"].to_numpy())

        report.update({
            "final_test_n_rows": len(test_df), "fitted_feature_count": X_train.shape[1],
            "preprocessing_seconds": round(preprocessing_seconds, 3), "fit_seconds": round(fit_seconds, 3), "predict_seconds": round(predict_seconds, 3),
            "model_metrics": model_metrics, "no_skill_baseline": baseline, "stage1_baseline_comparison": comparison,
            "productcd_segment_report": productcd_report, "amount_segment_boundaries": amount_boundaries, "amount_segment_report": amount_report,
            "bootstrap_ci": ci, "cost_threshold_sweep": cost_sweep,
        })
        record["model_family"] = "RandomForestClassifier"
        record["hyperparameters"] = model.get_params()
        record["feature_count"] = X_train.shape[1]
        record["training_time_seconds"] = round(fit_seconds, 3)
        record["inference_time_seconds"] = round(predict_seconds, 3)
        record["metrics"] = model_metrics
        record["validation_consulted"] = False
        record["partitions"] = {"final_test": {"id": "final_test (complete, THE one-time use)", "n_rows": len(test_df)}}

        out_path = DOCS_DIR / "_stage10_final_test_evaluation_raw.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        record["evidence_paths"] = [f"docs/_stage10_final_test_evaluation_raw.json"]
        print(f"\nFull structured report written to {out_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-approve-one-time-final-test-evaluation", dest="token", required=True,
        help=f"Must equal exactly {REQUIRED_UNLOCK_TOKEN!r} -- this is a deliberate, explicit confirmation, not a default.",
    )
    args = parser.parse_args()
    main(args.token)
