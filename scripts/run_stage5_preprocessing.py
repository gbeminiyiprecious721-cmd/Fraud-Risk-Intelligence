"""Stage 5 driver — builds, fits (train only), and verifies the preprocessing
pipeline against the real data; measures memory/density/timing.

Run: .venv/bin/python3 scripts/run_stage5_preprocessing.py

Read-only against data/raw/. All partition access goes through
src.leakage_boundaries.get_partition() — never a raw filter of the
assignment file, and final_test is never requested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    check_sparse_input_compatibility,
    estimate_indicator_memory_cost,
    fit_on_train_only,
    load_included_columns,
    measure_output,
    measure_train_missingness,
    select_indicator_columns,
    transform_partition,
    verify_column_accounting,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {}

    section("1. Column scope from the feature register")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    print(f"Numeric: {len(columns['numeric'])}  Categorical: {len(columns['categorical'])}")
    report["column_scope"] = {k: len(v) for k, v in columns.items()}

    section("2. Train-only missingness for numeric columns (decides indicator placement)")
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    print(missingness.head(10))
    print("...")
    print(missingness.tail(5))
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    print(f"\nWith indicator (>= {DEFAULT_INDICATOR_THRESHOLD*100}% missing in train): {len(indicator_split['with_indicator'])}")
    print(f"Without indicator: {len(indicator_split['without_indicator'])}")
    report["indicator_split_counts"] = {k: len(v) for k, v in indicator_split.items()}
    report["train_missingness_top10"] = missingness.head(10).to_dict()

    section("3. Memory implications of indicator columns at different thresholds (review-required investigation)")
    train_ids = get_partition(__import__("pandas").read_csv(ASSIGNMENT_PATH), ["train"])
    n_train_rows = len(train_ids)
    memory_scenarios = {}
    for label, thresh in [("every column with any missingness (0%)", 0.0), ("chosen threshold (5%)", 5.0), ("conservative (20%)", 20.0)]:
        split = select_indicator_columns(missingness, threshold_pct=thresh)
        cost = estimate_indicator_memory_cost(n_train_rows, len(split["with_indicator"]))
        memory_scenarios[label] = {"n_indicator_columns": len(split["with_indicator"]), **cost}
        print(f"{label}: {len(split['with_indicator'])} indicator columns -> {cost['estimated_mb']} MB (train rows only, float64 estimate)")
    report["indicator_memory_scenarios"] = memory_scenarios

    section("4. Categorical cardinality (informs item 6 — straightforward approach + noted alternatives)")
    import pandas as pd
    cat_df = pd.read_csv(TRAIN_TXN, usecols=columns["categorical"])
    cardinality = cat_df.nunique().sort_values(ascending=False)
    print(cardinality)
    report["categorical_cardinality"] = cardinality.to_dict()
    naive_ohe_width = int(cardinality.sum() + len(columns["categorical"]))  # +1 per column for the "missing" sentinel category
    print(f"\nEstimated naive one-hot width (incl. 'missing' sentinel per column): {naive_ohe_width}")
    report["naive_onehot_width_estimate"] = naive_ohe_width

    section("5. Fit on train ONLY")
    fitted = fit_on_train_only(
        ASSIGNMENT_PATH, TRAIN_TXN,
        indicator_split["without_indicator"], indicator_split["with_indicator"], columns["categorical"],
    )
    print(f"Fit wall time: {fitted.fit_wall_seconds:.3f}s")
    report["fit_wall_seconds"] = fitted.fit_wall_seconds

    section("6. Transform train and validation (NOT final_test) — measure output")
    for partition in ("train", "validation"):
        output, elapsed = transform_partition(fitted, ASSIGNMENT_PATH, TRAIN_TXN, partition)
        measured = measure_output(output, partition)
        measured["transform_wall_seconds"] = round(elapsed, 3)
        print(measured)
        report[f"output_{partition}"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in measured.items()}

    section("7a. Leakage-guard demonstration on REAL data (not just the synthetic test)")
    from sklearn.impute import SimpleImputer
    probe_col = "TransactionAmt"
    assignment_df = pd.read_csv(ASSIGNMENT_PATH)
    train_rows = get_partition(assignment_df, ["train"])
    val_rows = get_partition(assignment_df, ["validation"])
    txn_amt = pd.read_csv(TRAIN_TXN, usecols=["TransactionID", probe_col])
    train_vals = txn_amt.loc[txn_amt["TransactionID"].isin(train_rows["TransactionID"]), [probe_col]]
    combined_vals = txn_amt.loc[
        txn_amt["TransactionID"].isin(set(train_rows["TransactionID"]) | set(val_rows["TransactionID"])), [probe_col]
    ]
    train_only_median = SimpleImputer(strategy="median").fit(train_vals).statistics_[0]
    combined_median = SimpleImputer(strategy="median").fit(combined_vals).statistics_[0]
    print(f"{probe_col} median — train only: {train_only_median}   train+validation combined: {combined_median}")
    print(f"Difference: {abs(train_only_median - combined_median)}")
    report["leakage_guard_real_data_demo"] = {
        "probe_column": probe_col,
        "train_only_median": float(train_only_median),
        "train_plus_validation_median": float(combined_median),
        "differs": bool(train_only_median != combined_median),
    }

    section("7b. Confirm final_test is refused without explicit override (demonstrated, not just asserted)")
    try:
        transform_partition(fitted, ASSIGNMENT_PATH, TRAIN_TXN, "final_test")  # no allow_final_test
        report["final_test_refused_by_default"] = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        report["final_test_refused_by_default"] = True

    section("8. Column accounting against the register (review item 6)")
    register_df = pd.read_csv(REGISTER_PATH)
    accounting = verify_column_accounting(
        register_df,
        numeric=indicator_split["without_indicator"] + indicator_split["with_indicator"],
        categorical=columns["categorical"],
    )
    print(accounting)
    report["column_accounting"] = accounting

    section("9. Missingness-indicator redundancy (review item 3)")
    indicator_cols = indicator_split["with_indicator"]
    indicator_mask = pd.read_csv(TRAIN_TXN, usecols=["TransactionID"] + indicator_cols)
    train_only_mask = indicator_mask.loc[
        indicator_mask["TransactionID"].isin(set(get_partition(pd.read_csv(ASSIGNMENT_PATH), ["train"])["TransactionID"])),
        indicator_cols,
    ].isna()
    seen = {}
    duplicate_groups = []
    for col in indicator_cols:
        h = train_only_mask[col].to_numpy().tobytes()
        seen.setdefault(h, []).append(col)
    for names in seen.values():
        if len(names) > 1:
            duplicate_groups.append(names)
    n_duplicated = sum(len(g) for g in duplicate_groups)
    print(f"{len(indicator_cols)} indicator columns -> {len(duplicate_groups)} exact-duplicate-mask groups covering {n_duplicated} columns")
    print(f"Unique missingness patterns actually present: {len(seen)}")
    report["indicator_redundancy"] = {
        "n_indicator_columns": len(indicator_cols),
        "n_exact_duplicate_groups": len(duplicate_groups),
        "n_columns_in_duplicate_groups": n_duplicated,
        "n_unique_missingness_patterns": len(seen),
    }

    section("10. Sparse-input compatibility of candidate Stage 7 estimator families (review item 7, tags only, no .fit())")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import LinearSVC

    candidates = [LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, DecisionTreeClassifier, LinearSVC]
    sparse_compat = check_sparse_input_compatibility(candidates)
    print(sparse_compat)
    report["sparse_input_compatibility"] = sparse_compat

    out_path = DOCS_DIR / "_stage5_preprocessing_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")

    return report


if __name__ == "__main__":
    main()
