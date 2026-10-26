"""Stage 7 — Random Forest ablations (feature-engineering, missingness-indicator).

Both ablations use the SAME 15,000-row earliest-chronological `train`
subset as every prior Stage 7 pilot (NOT the full 438,125-row partition
-- a deliberate choice to stay bounded and avoid repeating the expensive
full-data run twice more; stated explicitly in the report, not hidden),
the SAME bounded Random Forest configuration
(n_estimators=100, max_depth=10, min_samples_leaf=20, n_jobs=-1,
random_state=42), and the SAME real `validation` partition for scoring.

Point-in-time correctness for the feature-engineering ablation (review
item 4): a chronological "earliest N rows" subset has a property worth
stating precisely -- no row OUTSIDE the subset can ever be "prior in
time" to a row INSIDE it (every excluded train row has a strictly later
TransactionDT). So computing `card1`'s historical features using the
15,000-row subset ALONE gives IDENTICAL values to computing them from
the full train+validation set, for every row IN the subset. Validation
rows do not have this property (their correct history must include ALL
of train, not just this subset), so validation's engineered features are
built via the already-approved `build_entity_history_train_and_validation`
(full train+validation), not from the pilot subset.

Usage:
    .venv/bin/python3 scripts/run_stage7_rf_ablations.py --experiment feature
    .venv/bin/python3 scripts/run_stage7_rf_ablations.py --experiment indicator
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
from src.feature_engineering import (  # noqa: E402
    collapsed_indicator_representatives,
    compute_interaction_and_amount_features,
    compute_temporal_phase_features,
    find_duplicate_indicator_groups,
    fit_frequency_encoder,
    apply_frequency_encoder,
    add_entity_historical_features,
    build_entity_history_train_and_validation,
)
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    build_preprocessing_pipeline,
    load_included_columns,
    load_partition_df,
    measure_train_missingness,
    select_indicator_columns,
)
from src.model_selection import (  # noqa: E402
    align_labels_to_partition_df,
    build_bounded_random_forest,
    build_chronological_pilot_subset,
    build_engineered_feature_block,
    build_preprocessing_pipeline_with_collapsed_indicators,
    evaluate_predictions,
    hstack_sparse,
    load_labels,
    sparse_magnitude_summary,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

PILOT_N_ROWS = 15_000


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def fit_rf_and_evaluate(X_train, y_train, X_val, y_val, label: str) -> dict:
    model = build_bounded_random_forest()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = model.predict_proba(X_val)[:, 1]
    predict_s = time.perf_counter() - t0
    metrics = evaluate_predictions(y_val, scores)
    mag = sparse_magnitude_summary(X_train) if hasattr(X_train, "nnz") else {"shape": list(X_train.shape)}
    print(f"[{label}] fit_seconds={fit_s:.3f}  predict_seconds={predict_s:.3f}  shape={mag.get('shape')}  "
          f"AP={metrics['average_precision']}  ROC-AUC={metrics['roc_auc']}")
    return {
        "fit_seconds": round(fit_s, 3),
        "predict_seconds": round(predict_s, 3),
        "shape": mag.get("shape"),
        "metrics": metrics,
    }


def run_feature_ablation() -> dict:
    section("Feature-engineering ablation (Random Forest): A = Stage 5 only, B = Stage 5 + Stage 6")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=PILOT_N_ROWS)
    y_train = pilot_df["isFraud"].to_numpy()

    section("0b. final_test refusal re-check")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    try:
        get_partition(assignment, ["train", "final_test"])
        final_test_refused = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        final_test_refused = True

    section("Arm A: Stage 5 only")
    ct_a = build_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
    X_train_a = ct_a.fit_transform(pilot_df[all_cols])
    val_raw_partition = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    val_labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")
    val_df = align_labels_to_partition_df(val_raw_partition, val_labels)
    y_val = val_df["isFraud"].to_numpy()
    X_val_a = ct_a.transform(val_df[all_cols])
    result_a = fit_rf_and_evaluate(X_train_a, y_train, X_val_a, y_val, "arm_a_stage5_only")

    section("Building Stage 6 engineered features -- point-in-time-correct for BOTH train-pilot and validation rows")
    raw_cols = ["TransactionID", "TransactionDT", "card1", "P_emaildomain", "R_emaildomain", "ProductCD", "card4", "TransactionAmt"]
    pilot_raw = pd.read_csv(TRAIN_TXN, usecols=raw_cols)
    pilot_raw = pilot_raw.loc[pilot_raw["TransactionID"].isin(set(pilot_df["TransactionID"]))]
    pilot_raw = compute_temporal_phase_features(pilot_raw)
    # TRAIN-pilot rows: computing entity history from the pilot subset ALONE is
    # correct here (see module docstring) -- no excluded train row can be prior
    # in time to any row inside this earliest-N-rows subset.
    pilot_hist = add_entity_historical_features(pilot_raw[["TransactionID", "TransactionDT", "card1"]], "card1")
    pilot_raw = pilot_raw.merge(pilot_hist[["TransactionID", "card1_prior_txn_count", "card1_time_since_prior_txn"]], on="TransactionID")

    # Frequency encoders fit on the PILOT subset only (the actual rows used for
    # this fit), not the full train partition -- consistent with "train-only"
    # meaning the rows this experiment's model is actually trained on.
    p_map = fit_frequency_encoder(pilot_raw, "P_emaildomain")
    r_map = fit_frequency_encoder(pilot_raw, "R_emaildomain")
    pilot_raw["p_emaildomain_freq"] = apply_frequency_encoder(pilot_raw["P_emaildomain"], p_map)
    pilot_raw["r_emaildomain_freq"] = apply_frequency_encoder(pilot_raw["R_emaildomain"], r_map)
    pilot_raw = compute_interaction_and_amount_features(pilot_raw)

    engineered_cols = [
        "TransactionID", "txn_hour_phase", "txn_dow_phase", "card1_prior_txn_count", "card1_time_since_prior_txn",
        "p_emaildomain_freq", "r_emaildomain_freq", "email_domain_match", "log1p_transaction_amt",
        "transaction_amt_is_round", "productcd_card4_interaction",
    ]
    train_engineered = pilot_df[["TransactionID"]].merge(pilot_raw[engineered_cols], on="TransactionID", how="left")

    # VALIDATION rows: correct history must include ALL of train (not just the
    # pilot subset) -- use the already-approved cross-partition builder.
    val_entity_hist = build_entity_history_train_and_validation(ASSIGNMENT_PATH, TRAIN_TXN, "card1")
    val_raw = pd.read_csv(TRAIN_TXN, usecols=raw_cols)
    val_raw = val_raw.loc[val_raw["TransactionID"].isin(set(val_df["TransactionID"]))]
    val_raw = compute_temporal_phase_features(val_raw)
    val_raw = val_raw.merge(val_entity_hist[["TransactionID", "card1_prior_txn_count", "card1_time_since_prior_txn"]], on="TransactionID")
    val_raw["p_emaildomain_freq"] = apply_frequency_encoder(val_raw["P_emaildomain"], p_map)  # train-only-fit map
    val_raw["r_emaildomain_freq"] = apply_frequency_encoder(val_raw["R_emaildomain"], r_map)
    val_raw = compute_interaction_and_amount_features(val_raw)
    val_engineered = val_df[["TransactionID"]].merge(val_raw[engineered_cols], on="TransactionID", how="left")

    feat_block = build_engineered_feature_block()
    X_train_feat = feat_block.fit_transform(train_engineered)
    X_val_feat = feat_block.transform(val_engineered)

    X_train_b = hstack_sparse([X_train_a, X_train_feat])
    X_val_b = hstack_sparse([X_val_a, X_val_feat])

    section("Arm B: Stage 5 + Stage 6 engineered features")
    result_b = fit_rf_and_evaluate(X_train_b, y_train, X_val_b, y_val, "arm_b_stage5_plus_stage6")

    ap_delta = result_b["metrics"]["average_precision"] - result_a["metrics"]["average_precision"]
    print(f"\nAP delta (B - A): {ap_delta:+.5f}")
    print("Per instruction: an AP change here is NOT interpreted as proof of operational feature availability --")
    print("it is a diagnostic signal only, and every engineered feature's availability caveats (docs/engineered_features_register.csv) stand unchanged.")

    return {
        "final_test_refused": final_test_refused,
        "arm_a": result_a,
        "arm_b": result_b,
        "ap_delta_b_minus_a": round(float(ap_delta), 5),
    }


def run_indicator_ablation() -> dict:
    section("Missingness-indicator ablation (Random Forest): original (full) vs collapsed")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=PILOT_N_ROWS)
    y_train = pilot_df["isFraud"].to_numpy()
    val_raw_partition = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    val_labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")
    val_df = align_labels_to_partition_df(val_raw_partition, val_labels)
    y_val = val_df["isFraud"].to_numpy()

    section("Arm 'original': the approved, unmodified Stage 5 pipeline (reference, unchanged)")
    ct_orig = build_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
    X_train_orig = ct_orig.fit_transform(pilot_df[all_cols])
    X_val_orig = ct_orig.transform(val_df[all_cols])
    result_orig = fit_rf_and_evaluate(X_train_orig, y_train, X_val_orig, y_val, "original_full_indicators")

    section("Indicator-duplicate groups -- recomputed fresh on this pilot subset")
    groups = find_duplicate_indicator_groups(pilot_df, numeric_with_indicator)
    representatives = collapsed_indicator_representatives(groups)
    print(f"{len(numeric_with_indicator)} indicator columns -> {len(representatives)} representatives on this 15,000-row subset")

    section("Arm 'collapsed': Stage 6 prototype, UNSCALED (tree-appropriate), approved pipeline NOT modified")
    ct_collapsed = build_preprocessing_pipeline_with_collapsed_indicators(numeric_no_indicator, numeric_with_indicator, categorical, representatives)
    X_train_collapsed = ct_collapsed.fit_transform(pilot_df[all_cols])
    X_val_collapsed = ct_collapsed.transform(val_df[all_cols])
    result_collapsed = fit_rf_and_evaluate(X_train_collapsed, y_train, X_val_collapsed, y_val, "collapsed_indicators")

    ap_delta = result_collapsed["metrics"]["average_precision"] - result_orig["metrics"]["average_precision"]
    print(f"\nAP delta (collapsed - original): {ap_delta:+.5f}")
    print(f"Feature width: original {result_orig['shape']} vs collapsed {result_collapsed['shape']}")
    print("The approved Stage 5 pipeline (src/preprocessing.py) is NOT modified by this ablation.")

    return {
        "n_representatives": len(representatives),
        "original": result_orig,
        "collapsed": result_collapsed,
        "ap_delta_collapsed_minus_original": round(float(ap_delta), 5),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["feature", "indicator"])
    args = parser.parse_args()

    result = run_feature_ablation() if args.experiment == "feature" else run_indicator_ablation()
    out_path = DOCS_DIR / f"_stage7_rf_{args.experiment}_ablation_raw.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")
