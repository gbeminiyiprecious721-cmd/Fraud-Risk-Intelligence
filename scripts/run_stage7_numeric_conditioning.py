"""Stage 7 — controlled numerical-conditioning verification.

Reuses the IDENTICAL 15,000-row chronological `train` subset as the two
prior Stage 7 driver scripts. Compares:

  A. The APPROVED indicator-collapse representation (unchanged) + unweighted SAGA.
  B. The SAME representation + a training-derived numeric-value-redundancy
     reduction (new this round), + unweighted SAGA -- isolating exactly one
     variable (the value-column set) between A and B.

`class_weight=None` is used in BOTH arms deliberately, not as a silent
choice: the previous round's "class_weight='balanced' causes oscillation"
finding was evaluated against the wrong objective (see §2 of this round's
report) and is corrected here by removing class-weighting from this
specific comparison entirely, so the two arms differ in exactly one way.

TRAIN-ONLY throughout. Validation AP is computed once, at the very end,
for continuity with prior rounds' reports -- explicitly NOT used to pick
between A and B anywhere in this script. `final_test` is never requested.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.feature_engineering import collapsed_indicator_representatives, find_duplicate_indicator_groups  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    load_included_columns,
    load_partition_df,
    measure_train_missingness,
    select_indicator_columns,
)
from src.model_selection import (  # noqa: E402
    RANDOM_STATE,
    build_chronological_pilot_subset,
    build_scaled_pipeline_with_collapsed_indicators,
    build_scaled_pipeline_with_collapsed_indicators_and_value_reduction,
    evaluate_predictions,
    find_duplicate_value_columns,
    find_highly_correlated_value_groups,
    load_labels,
    select_reduced_numeric_columns,
    sparse_magnitude_summary,
    track_saga_training_loss,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

PILOT_N_ROWS = 15_000  # identical to the two prior Stage 7 driver scripts


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def fit_and_measure(ct, pilot_df, all_cols, y_train, label):
    t0 = time.perf_counter()
    X_train = ct.fit_transform(pilot_df[all_cols])
    pipeline_s = time.perf_counter() - t0

    model = LogisticRegression(solver="saga", penalty="l2", C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    mag = sparse_magnitude_summary(X_train)
    print(f"[{label}] pipeline_fit_transform={pipeline_s:.3f}s  shape={mag['shape']}  density={mag['density']}  "
          f"dynamic_range={mag['dynamic_range_orders_of_magnitude']}")
    print(f"[{label}] saga fit_seconds={fit_s:.3f}  n_iter_={int(model.n_iter_[0])}  converged={converged}")

    return {
        "ct": ct,
        "X_train": X_train,
        "model": model,
        "pipeline_fit_seconds": round(pipeline_s, 3),
        "fit_seconds": round(fit_s, 3),
        "n_iter": int(model.n_iter_[0]),
        "converged": converged,
        "sparse_magnitude": mag,
    }


def main() -> dict:
    report: dict = {}

    section("0. Rebuild the SAME 15,000-row chronological pilot subset")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    numeric_value_cols = numeric_no_indicator + numeric_with_indicator
    all_cols = numeric_value_cols + categorical

    pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=PILOT_N_ROWS)
    y_train = pilot_df["isFraud"].to_numpy()
    print(f"Pilot subset: {len(pilot_df)} rows, fraud prevalence {y_train.mean():.4%} (identical subset, both prior scripts)")

    section("0b. Confirm final_test still refused from this script too")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    try:
        get_partition(assignment, ["train", "final_test"])
        report["final_test_refused"] = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        report["final_test_refused"] = True

    section("1. Train-only numeric redundancy detection (exact duplicates + highly-correlated clusters)")
    imputer = SimpleImputer(strategy="median")
    imputed_values = imputer.fit_transform(pilot_df[numeric_value_cols])
    exact_groups = find_duplicate_value_columns(imputed_values, numeric_value_cols)
    corr_groups = find_highly_correlated_value_groups(imputed_values, numeric_value_cols, threshold=0.99)
    reduction = select_reduced_numeric_columns(numeric_value_cols, exact_groups, corr_groups)
    n_distinct_affected = len(set(reduction["dropped_to_representative"].keys()) | set(reduction["dropped_to_representative"].values()))
    print(f"Original numeric value columns: {reduction['n_original']}")
    print(f"Exact-duplicate groups: {len(exact_groups)}  |  Highly-correlated (>=0.99) clusters: {len(corr_groups)}")
    print(f"Kept: {reduction['n_kept']}  Dropped: {reduction['n_dropped']}  Distinct columns involved (kept+dropped): {n_distinct_affected}")
    print(f"Sample of drop mapping: {dict(list(reduction['dropped_to_representative'].items())[:8])}")
    report["numeric_reduction"] = {
        "n_original": reduction["n_original"],
        "n_kept": reduction["n_kept"],
        "n_dropped": reduction["n_dropped"],
        "n_exact_duplicate_groups": len(exact_groups),
        "n_correlated_clusters": len(corr_groups),
        "full_drop_mapping": reduction["dropped_to_representative"],
    }

    section("2. Indicator-collapse representatives -- UNCHANGED, computed exactly as the approved prototype does")
    indicator_groups = find_duplicate_indicator_groups(pilot_df, numeric_with_indicator)
    representatives = collapsed_indicator_representatives(indicator_groups)
    print(f"{len(numeric_with_indicator)} indicator columns -> {len(representatives)} representatives (same method as Stage 6/7, not altered)")

    section("3. ARM A: existing collapsed-indicator pipeline (full numeric value set) + unweighted SAGA")
    ct_a = build_scaled_pipeline_with_collapsed_indicators(numeric_no_indicator, numeric_with_indicator, categorical, representatives)
    result_a = fit_and_measure(ct_a, pilot_df, all_cols, y_train, "Arm A (baseline)")

    section("4. ARM B: SAME pipeline + training-derived numeric-value reduction + unweighted SAGA")
    ct_b = build_scaled_pipeline_with_collapsed_indicators_and_value_reduction(
        numeric_no_indicator, numeric_with_indicator, categorical, representatives, reduction["keep"]
    )
    result_b = fit_and_measure(ct_b, pilot_df, all_cols, y_train, "Arm B (reduced)")

    section("5. Train-only loss trajectory, both arms (identical class_weight=None -- isolates ONLY the column-set change)")
    traces = {}
    for label, X in [("arm_a", result_a["X_train"]), ("arm_b", result_b["X_train"])]:
        trace = track_saga_training_loss(X, y_train, class_weight=None, n_steps=10, iters_per_step=100)
        for point in trace:
            print(f"  [{label}] iters={point['cumulative_iters']:5d}  train_log_loss={point['train_log_loss']:.6f}  "
                  f"coef_change={point['coef_l2_change_from_previous_step']}")
        traces[label] = trace
    report["loss_trajectories"] = traces

    section("6. Exploratory, predefined, NOT tuned: C=0.1 and C=10 on Arm B only, train-only diagnostic")
    print("Small, predefined regularization check -- reported as exploratory, not a search for the best validation score.")
    exploratory = {}
    for C in (0.1, 10.0):
        model = LogisticRegression(solver="saga", penalty="l2", C=C, max_iter=1000, random_state=RANDOM_STATE, class_weight=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            t0 = time.perf_counter()
            model.fit(result_b["X_train"], y_train)
            fit_s = time.perf_counter() - t0
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        print(f"  Arm B, C={C}: fit_seconds={fit_s:.3f} n_iter_={int(model.n_iter_[0])} converged={converged}")
        exploratory[f"C_{C}"] = {"fit_seconds": round(fit_s, 3), "n_iter": int(model.n_iter_[0]), "converged": converged}
    report["exploratory_regularization_check_arm_b"] = exploratory

    section("7. Validation AP -- CONTINUITY ONLY, not used to choose between Arm A and Arm B")
    val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    y_val = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")
    for label, result, ct in [("arm_a", result_a, ct_a), ("arm_b", result_b, ct_b)]:
        X_val = ct.transform(val_df[all_cols])
        scores = result["model"].predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
        print(f"{label}: validation AP (continuity only) = {metrics['average_precision']}")
        result["validation_ap_continuity_only"] = metrics["average_precision"]

    report["arm_a"] = {k: v for k, v in result_a.items() if k not in ("ct", "X_train", "model")}
    report["arm_b"] = {k: v for k, v in result_b.items() if k not in ("ct", "X_train", "model")}

    out_path = DOCS_DIR / "_stage7_numeric_conditioning_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
