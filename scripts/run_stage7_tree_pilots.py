"""Stage 7 — bounded tree-based pilots (Random Forest, Gradient Boosting).

Each `--model` value is meant to be invoked as a SEPARATE process (under
`/usr/bin/time -l`) for clean, isolated peak-memory measurement -- same
discipline as every other Stage 7 driver script in this project (see
docs/PREPROCESSING_REPORT.md §13.2 for the documented lesson about
same-process high-water-mark contamination).

Uses the APPROVED Stage 5 preprocessing (src.preprocessing.build_preprocessing_pipeline
-- unscaled, full per-column indicators), NOT the experimental
collapsed-indicator variant built for the Logistic Regression
investigation. Fits on the IDENTICAL 15,000-row earliest-chronological
`train` subset used by every prior Stage 7 script. Scores on the real
`validation` partition. `final_test` is never requested.

Usage:
    .venv/bin/python3 scripts/run_stage7_tree_pilots.py --model rf
    .venv/bin/python3 scripts/run_stage7_tree_pilots.py --model gb
"""

from __future__ import annotations

import argparse
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
from src.model_selection import (  # noqa: E402
    build_bounded_gradient_boosting,
    build_bounded_random_forest,
    build_chronological_pilot_subset,
    evaluate_predictions,
    load_labels,
    run_basic_model_correctness_checks,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

PILOT_N_ROWS = 15_000  # identical to every prior Stage 7 driver script


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main(model_name: str) -> dict:
    report: dict = {"model": model_name}

    section("0. Rebuild the SAME 15,000-row chronological pilot subset")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=PILOT_N_ROWS)
    y_train = pilot_df["isFraud"].to_numpy()
    print(f"Pilot subset: {len(pilot_df)} rows, fraud prevalence {y_train.mean():.4%} (identical subset, every prior Stage 7 script)")
    report["pilot_n_rows"] = len(pilot_df)
    report["pilot_fraud_prevalence"] = round(float(y_train.mean()), 5)

    section("0b. Confirm final_test still refused")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    try:
        get_partition(assignment, ["train", "final_test"])
        report["final_test_refused"] = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        report["final_test_refused"] = True

    section("1. APPROVED Stage 5 preprocessing (unscaled, full indicators -- not the LogReg-specific experimental variant)")
    ct = build_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
    t0 = time.perf_counter()
    X_train = ct.fit_transform(pilot_df[all_cols])
    preprocessing_seconds = time.perf_counter() - t0
    print(f"Preprocessing (fit_transform) runtime: {preprocessing_seconds:.3f}s, shape {X_train.shape}")
    report["preprocessing_seconds"] = round(preprocessing_seconds, 3)
    report["matrix_shape"] = list(X_train.shape)

    val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    y_val = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")
    X_val = ct.transform(val_df[all_cols])

    section(f"2. {model_name.upper()} pilot -- one bounded, non-tuned configuration")
    if model_name == "rf":
        model = build_bounded_random_forest()
        print("RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=42)")
    elif model_name == "gb":
        model = build_bounded_gradient_boosting()
        print("GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)")
        print("Known limitation, stated plainly: sequential, single-threaded (no n_jobs) -- unlike the Random Forest pilot.")
    else:
        raise ValueError(f"unknown model {model_name!r}")

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    print(f"fit_seconds={fit_seconds:.3f}")
    report["fit_seconds"] = round(fit_seconds, 3)
    report["config"] = {k: v for k, v in model.get_params().items() if k in ("n_estimators", "max_depth", "min_samples_leaf", "random_state", "n_jobs")}

    section("3. Correctness checks (structural sanity, not performance) -- gate for proceeding to the next pilot")
    checks = run_basic_model_correctness_checks(model, X_val)
    print(checks)
    report["correctness_checks"] = checks

    section("4. Inference time + evaluation on the REAL validation partition")
    t0 = time.perf_counter()
    scores = model.predict_proba(X_val)[:, 1]
    predict_seconds = time.perf_counter() - t0
    metrics = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
    print(f"predict_seconds={predict_seconds:.3f}")
    print(f"AP={metrics['average_precision']}  ROC-AUC={metrics['roc_auc']}  "
          f"P@1%={metrics['k_frac_0.01']['precision_at_k']}  R@1%={metrics['k_frac_0.01']['recall_at_k']}  "
          f"P@5%={metrics['k_frac_0.05']['precision_at_k']}  R@5%={metrics['k_frac_0.05']['recall_at_k']}")
    print("Reminder: this is a DIAGNOSTIC pilot result (15,000-row training subset), not final model-selection evidence.")
    report["predict_seconds"] = round(predict_seconds, 3)
    report["metrics"] = metrics

    out_path = DOCS_DIR / f"_stage7_tree_pilot_{model_name}_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rf", "gb"])
    args = parser.parse_args()
    main(args.model)
