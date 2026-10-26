"""Stage 7 — controlled full-train Random Forest benchmark.

ONE Random Forest, on the COMPLETE approved 438,125-row `train` partition
(not the 15,000-row pilot subset), with the IDENTICAL configuration as
the pilot -- no change:

    RandomForestClassifier(n_estimators=100, max_depth=10,
                            min_samples_leaf=20, n_jobs=-1, random_state=42)

Approved (unscaled) Stage 5 preprocessing, fit on `train` only via
`load_partition_df()` (chunked reads -- never materializes the full
unfiltered file, same discipline as every Stage 5+ driver). Scored on
the real `validation` partition only. `final_test` is never requested.

Meant to run under an isolated process with:
  - An OS-level wall-clock timeout (the harness's own process timeout,
    not Python's signal.alarm).
  - A conservative `ulimit -v` virtual-memory cap, set by the invoking
    shell command, not by this script -- see the accompanying report for
    the exact value and its justification (Stage 5's own historical
    peak for fitting this same pipeline on the same 438,125 rows).

No retry logic of any kind: if this process is killed (timeout or OOM),
it simply stops -- the caller is responsible for not re-invoking it
automatically, per instruction.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    build_preprocessing_pipeline,
    load_included_columns,
    load_partition_df,
    measure_train_missingness,
    select_indicator_columns,
)
from src.model_selection import align_labels_to_partition_df, build_bounded_random_forest, evaluate_predictions, load_labels  # noqa: E402
from src.experiment_tracking import track_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {}

    # Stage 8 automatic logging: wraps the EXISTING code below unchanged --
    # see the matching note in run_stage7_tree_pilots.py. This script is
    # the most resource-sensitive in Stage 7 (16GB+ peak, ~43s wall time);
    # if it is ever killed by the harness's OS-level timeout partway
    # through, track_run's exception handling still logs that as a
    # "failed" run with whatever partial fields were set, never silently.
    with track_run(7, "rf_full_train_benchmark", "Full 438,125-row train partition Random Forest benchmark (live re-run).") as record:
        section("0. Load the COMPLETE approved train partition (not the 15,000-row pilot)")
        columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
        missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
        indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
        numeric_no_indicator = indicator_split["without_indicator"]
        numeric_with_indicator = indicator_split["with_indicator"]
        categorical = columns["categorical"]
        all_cols = numeric_no_indicator + numeric_with_indicator + categorical

        train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols)
        # Explicit alignment by TransactionID -- NOT by assuming load_partition_df's
        # file-order output happens to match load_labels()'s sort-by-TransactionID
        # output. They do, in fact, agree here (the raw CSV is itself monotonically
        # increasing by TransactionID, verified separately), but align_labels_to_partition_df
        # makes that fact irrelevant rather than relying on it silently.
        labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
        train_df = align_labels_to_partition_df(train_df, labels)
        y_train = train_df["isFraud"].to_numpy()
        print(f"Full train partition: {len(train_df)} rows (expected 438,125)")
        report["train_n_rows"] = len(train_df)

        section("0b. Confirm final_test still refused")
        assignment = pd.read_csv(ASSIGNMENT_PATH)
        try:
            get_partition(assignment, ["train", "final_test"])
            report["final_test_refused"] = False
        except FinalTestAccessError as exc:
            print(f"Correctly refused: {exc}")
            report["final_test_refused"] = True

        section("1. Fit the APPROVED (unscaled) Stage 5 preprocessing pipeline on the full train partition")
        ct = build_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
        t0 = time.perf_counter()
        X_train = ct.fit_transform(train_df[all_cols])
        preprocessing_seconds = time.perf_counter() - t0
        print(f"Preprocessing fit_transform: {preprocessing_seconds:.3f}s, shape {X_train.shape}")
        report["preprocessing_seconds"] = round(preprocessing_seconds, 3)
        report["matrix_shape"] = list(X_train.shape)
        report["fitted_feature_count"] = int(X_train.shape[1])
        record["preprocessing_config"] = {"pipeline": "build_preprocessing_pipeline (approved, unscaled)"}
        record["feature_count"] = report["fitted_feature_count"]
        record["partitions"] = {"train": {"id": "train (complete)", "n_rows": report["train_n_rows"], "fraud_prevalence": round(float(y_train.mean()), 5)}}

        section("2. Fit RandomForestClassifier -- IDENTICAL config to the pilot, no change")
        model = build_bounded_random_forest()
        print("RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=42)")
        record["model_family"] = "RandomForestClassifier"
        record["hyperparameters"] = {k: v for k, v in model.get_params().items() if k in ("n_estimators", "max_depth", "min_samples_leaf", "random_state", "n_jobs")}
        record["random_seed"] = model.get_params().get("random_state")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        print(f"fit_seconds={fit_seconds:.3f}")
        report["fit_seconds"] = round(fit_seconds, 3)
        record["training_time_seconds"] = report["fit_seconds"]

        depths = [est.get_depth() for est in model.estimators_]
        leaves = [est.get_n_leaves() for est in model.estimators_]
        print(f"Trees fitted: {len(model.estimators_)}  depth range: {min(depths)}-{max(depths)}  leaf range: {min(leaves)}-{max(leaves)}")
        report["n_trees_fitted"] = len(model.estimators_)
        report["tree_depth_range"] = [min(depths), max(depths)]
        report["tree_leaf_range"] = [min(leaves), max(leaves)]

        section("3. Evaluate on the REAL validation partition only")
        val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
        y_val = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")
        t0 = time.perf_counter()
        X_val = ct.transform(val_df[all_cols])
        val_transform_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = model.predict_proba(X_val)[:, 1]
        predict_seconds = time.perf_counter() - t0
        metrics = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
        print(f"validation transform: {val_transform_seconds:.3f}s, predict: {predict_seconds:.3f}s")
        print(f"AP={metrics['average_precision']}  ROC-AUC={metrics['roc_auc']}")
        print(f"P@1%={metrics['k_frac_0.01']['precision_at_k']}  R@1%={metrics['k_frac_0.01']['recall_at_k']}")
        print(f"P@5%={metrics['k_frac_0.05']['precision_at_k']}  R@5%={metrics['k_frac_0.05']['recall_at_k']}")
        print("This is Stage 7 diagnostic/model-selection evidence, not a final Stage 10 evaluation -- final_test untouched.")

        report["validation_transform_seconds"] = round(val_transform_seconds, 3)
        report["predict_seconds"] = round(predict_seconds, 3)
        report["metrics"] = metrics
        report["train_fraud_prevalence"] = round(float(y_train.mean()), 5)
        record["inference_time_seconds"] = report["predict_seconds"]
        record["metrics"] = metrics
        record["partitions"]["validation"] = {"id": "validation", "n_rows": metrics["n_rows"]}
        record["validation_consulted"] = True
        record["validation_purpose"] = "the headline reported Stage 7 result"
        record["influenced_subsequent_decisions"] = True

        out_path = DOCS_DIR / "_stage7_rf_full_train_raw.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull structured report written to {out_path}")
        record["evidence_paths"] = ["docs/_stage7_rf_full_train_raw.json"]

    return report


if __name__ == "__main__":
    main()
