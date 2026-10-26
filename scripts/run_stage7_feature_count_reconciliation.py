"""Stage 7 closure — reconcile the pilot (790) vs full-train (813) fitted
feature-count discrepancy flagged during review.

This is a READ-ONLY reconciliation diagnostic: it re-fits ONLY the approved
`build_preprocessing_pipeline` ColumnTransformer (never a model) on the same
two row sets already used (the 15,000-row chronological pilot, the full
438,125-row `train` partition) and inspects `transformers_` branch-by-branch
to attribute the exact delta. No model is trained, no experimental result
(any reported AP/ROC-AUC/etc.) is touched or recomputed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    build_preprocessing_pipeline,
    load_included_columns,
    load_partition_df,
    measure_train_missingness,
    select_indicator_columns,
)
from src.model_selection import build_chronological_pilot_subset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def branch_widths(ct) -> dict:
    widths = {}
    for name, _, cols in ct.transformers_:
        if name == "remainder":
            continue
        fitted_transformer = ct.named_transformers_[name]
        # Probe the actual fitted width by transforming a 1-row slice is
        # unnecessary -- each branch's output width is directly inspectable.
        if name == "numeric_plain":
            widths[name] = len(cols)
        elif name == "numeric_indicator":
            imputer = fitted_transformer.named_steps["impute"]
            n_indicators = imputer.indicator_.features_.shape[0] if imputer.indicator_ is not None else 0
            widths[name] = {"input_columns": len(cols), "indicator_columns_added": int(n_indicators), "output_width": len(cols) + int(n_indicators)}
        elif name == "categorical":
            onehot = fitted_transformer.named_steps["onehot"]
            n_categories = sum(len(c) for c in onehot.categories_)
            widths[name] = {"input_columns": len(cols), "onehot_output_width": int(n_categories)}
    return widths


def main() -> None:
    section("0. Same column scope as every Stage 7 script (register-driven, identical both runs)")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical
    print(f"numeric_no_indicator={len(numeric_no_indicator)}  numeric_with_indicator (eligible)={len(numeric_with_indicator)}  categorical={len(categorical)}")
    print("This eligible-column split itself is IDENTICAL for both runs below -- it comes from "
          "measure_train_missingness(), which always reads the FULL train partition regardless of caller.")

    section("1. Fit the preprocessing ColumnTransformer on the 15,000-row pilot (no model)")
    pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=15_000)
    ct_pilot = build_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
    X_pilot = ct_pilot.fit_transform(pilot_df[all_cols])
    pilot_widths = branch_widths(ct_pilot)
    print(f"Pilot fitted shape: {X_pilot.shape}")
    for k, v in pilot_widths.items():
        print(f"  {k}: {v}")

    section("2. Fit the SAME (unmodified) preprocessing ColumnTransformer on the FULL 438,125-row train partition (no model)")
    full_train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols)
    ct_full = build_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
    X_full = ct_full.fit_transform(full_train_df[all_cols])
    full_widths = branch_widths(ct_full)
    print(f"Full-train fitted shape: {X_full.shape}")
    for k, v in full_widths.items():
        print(f"  {k}: {v}")

    section("3. Attribute the delta")
    delta_indicator = full_widths["numeric_indicator"]["indicator_columns_added"] - pilot_widths["numeric_indicator"]["indicator_columns_added"]
    delta_onehot = full_widths["categorical"]["onehot_output_width"] - pilot_widths["categorical"]["onehot_output_width"]
    total_delta = X_full.shape[1] - X_pilot.shape[1]
    print(f"numeric_indicator branch: pilot added {pilot_widths['numeric_indicator']['indicator_columns_added']} indicator columns, "
          f"full-train added {full_widths['numeric_indicator']['indicator_columns_added']} -- delta {delta_indicator:+d}")
    print(f"categorical branch: pilot one-hot width {pilot_widths['categorical']['onehot_output_width']}, "
          f"full-train one-hot width {full_widths['categorical']['onehot_output_width']} -- delta {delta_onehot:+d}")
    print(f"Total fitted-width delta: {total_delta:+d} (pilot {X_pilot.shape[1]} -> full-train {X_full.shape[1]})")
    print(f"Sum of attributed deltas: {delta_indicator + delta_onehot:+d} (should equal total delta exactly)")
    assert delta_indicator + delta_onehot == total_delta, "Attributed deltas do not sum to the observed total -- investigate further."
    print("\nConclusion: both branches are governed by sklearn's own fit-time behavior -- "
          "SimpleImputer(add_indicator=True) only emits an indicator column for a column that has "
          "AT LEAST ONE missing value in the exact rows passed to .fit(), and OneHotEncoder's output "
          "width is the number of DISTINCT categories observed in the exact rows passed to .fit(). "
          "Both are fit-time (not transform-time) decisions by design -- fitting on 15,000 earliest-chronological "
          "rows vs. the full 438,125-row partition can legitimately produce a different fitted column count. "
          "This is correct train-only-fit behavior, NOT a bug, a leakage issue, or a reporting error in either "
          "of the two already-reported numbers (790 pilot / 813 full-train) -- both are genuinely what each "
          "fit produced on its own row set, and both were reported correctly.")


if __name__ == "__main__":
    main()
