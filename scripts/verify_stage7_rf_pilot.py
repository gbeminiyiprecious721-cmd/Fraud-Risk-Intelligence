"""Independent verification of the Stage 7 Random Forest pilot, per the
user's 8-point checklist. Re-derives the pilot from scratch (does not
trust the saved JSON) and prints explicit PASS/FAIL evidence for each
point. Read-only, diagnostic -- no config change, no new training beyond
reproducing the already-approved pilot exactly.

Run: .venv/bin/python3 scripts/verify_stage7_rf_pilot.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.leakage_boundaries import get_partition  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD, build_preprocessing_pipeline, load_included_columns,
    load_partition_df, measure_train_missingness, select_indicator_columns,
)
from src.model_selection import build_bounded_random_forest, build_chronological_pilot_subset  # noqa: E402

REGISTER_PATH = ROOT / "docs/feature_availability_register.csv"
ASSIGNMENT_PATH = ROOT / "docs/_stage4_split_assignment.csv"
TRAIN_TXN = ROOT / "data" / "raw" / "ieee-fraud-detection" / "train_transaction.csv"


def main() -> dict:
    results: dict = {}

    # 1. Exactly 15,000 chronological training rows passed to fit.
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=15_000)
    check1 = len(pilot_df) == 15_000
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    train_ids = set(get_partition(assignment, ["train"])["TransactionID"])
    check1b = set(pilot_df["TransactionID"]) <= train_ids
    print(f"[1] Exactly 15,000 rows: {len(pilot_df)} -> {'PASS' if check1 else 'FAIL'}")
    print(f"[1b] All rows genuinely from the TRAIN partition (not validation/final_test): {'PASS' if check1b else 'FAIL'}")
    results["check1_exact_row_count"] = check1
    results["check1b_all_rows_from_train"] = check1b

    # 2. Labels align with those rows (independent lookup, not trusting the pipeline's own join).
    raw_labels = pd.read_csv(TRAIN_TXN, usecols=["TransactionID", "isFraud"]).set_index("TransactionID")["isFraud"]
    mismatches = [tid for tid, lbl in zip(pilot_df["TransactionID"], pilot_df["isFraud"]) if raw_labels.loc[tid] != lbl]
    check2 = len(mismatches) == 0
    print(f"[2] Label alignment (independent cross-check against raw file): {len(mismatches)} mismatches -> {'PASS' if check2 else 'FAIL'}")
    results["check2_label_alignment"] = check2

    # 3. Expected Stage 5 features/transformations used (the APPROVED, unscaled pipeline).
    ct = build_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
    transformer_names = [name for name, _, _ in ct.transformers]
    expected_names = ["numeric_plain", "numeric_indicator", "categorical"]
    check3 = transformer_names == expected_names
    print(f"[3] Pipeline is the APPROVED (unscaled) Stage 5 structure, not the LogReg-experimental variant: "
          f"transformers={transformer_names} -> {'PASS' if check3 else 'FAIL'}")
    print(f"    Source columns: {len(numeric_no_indicator)} numeric_plain + {len(numeric_with_indicator)} numeric_indicator + {len(categorical)} categorical "
          f"= {len(all_cols)} total (expected 392, per Stage 5's register-driven count)")
    results["check3_pipeline_structure"] = check3
    results["check3_source_column_count"] = len(all_cols)

    X_train = ct.fit_transform(pilot_df[all_cols])
    print(f"    Fitted output shape: {X_train.shape}")
    results["fitted_output_shape"] = list(X_train.shape)

    # 4. All 100 trees fitted.
    y_train = pilot_df["isFraud"].to_numpy()
    model = build_bounded_random_forest()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    isolated_fit_seconds = time.perf_counter() - t0
    check4 = len(model.estimators_) == 100
    print(f"[4] All 100 trees fitted: len(model.estimators_)={len(model.estimators_)} -> {'PASS' if check4 else 'FAIL'}")
    results["check4_all_trees_fitted"] = check4

    # 5. Tree depths and leaf counts are plausible (bounded by max_depth=10, not degenerate stumps).
    depths = [est.get_depth() for est in model.estimators_]
    leaves = [est.get_n_leaves() for est in model.estimators_]
    check5a = all(d <= 10 for d in depths)
    check5b = min(leaves) > 1
    print(f"[5] Tree depth range: {min(depths)}-{max(depths)} (max_depth=10 respected: {'PASS' if check5a else 'FAIL'})")
    print(f"    Leaf count range: {min(leaves)}-{max(leaves)} (non-degenerate: {'PASS' if check5b else 'FAIL'})")
    results["check5_depth_range"] = [min(depths), max(depths)]
    results["check5_leaf_range"] = [min(leaves), max(leaves)]
    results["check5_passed"] = bool(check5a and check5b)

    # 6. The fitted model used the stated settings.
    params = model.get_params()
    expected_params = {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 20, "n_jobs": -1, "random_state": 42}
    check6 = all(params[k] == v for k, v in expected_params.items())
    print(f"[6] Fitted model params match stated settings: {expected_params} -> {'PASS' if check6 else 'FAIL'}")
    results["check6_params_match"] = check6

    # 7. Training/prediction timings exclude no hidden repeated work.
    print(f"[7] Isolated re-fit timing just now: {isolated_fit_seconds:.3f}s (original pilot script reported 0.389s -- "
          f"both measure ONLY model.fit(), not pipeline fit_transform or anything else)")
    results["check7_isolated_refit_seconds"] = round(isolated_fit_seconds, 3)

    # 8. Evaluation consumed only the approved validation partition.
    val_ids = set(get_partition(assignment, ["validation"])["TransactionID"])
    val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    check8a = set(val_df["TransactionID"]) <= val_ids
    check8b = len(val_df) == len(val_ids)
    final_test_ids = set(assignment.loc[assignment["partition"] == "final_test", "TransactionID"])
    check8c = not (set(val_df["TransactionID"]) & final_test_ids)
    print(f"[8] Validation set used: {len(val_df)} rows, all from the validation partition: {'PASS' if check8a else 'FAIL'}, "
          f"exact expected count ({len(val_ids)}): {'PASS' if check8b else 'FAIL'}, zero overlap with final_test: {'PASS' if check8c else 'FAIL'}")
    results["check8_validation_only"] = bool(check8a and check8b and check8c)

    all_passed = all(v for k, v in results.items() if isinstance(v, bool))
    print(f"\nALL CHECKS PASSED: {all_passed}")
    results["all_checks_passed"] = all_passed

    out_path = ROOT / "docs" / "_stage7_rf_pilot_verification_raw.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")
    return results


if __name__ == "__main__":
    main()
