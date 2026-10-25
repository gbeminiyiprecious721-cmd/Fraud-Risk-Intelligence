"""Stage 6 driver — builds and measures each engineered feature family
against the real data. Read-only against data/raw/. All partition access
goes through src.leakage_boundaries.get_partition() (directly or via
src.feature_engineering's wrappers) -- final_test is never requested.

Run: .venv/bin/python3 scripts/run_stage6_feature_engineering.py

No model is trained or scored anywhere in this script, per explicit
instruction -- every number below is a STRUCTURAL property of a feature
(coverage, distribution, memory/time cost), never a predictive-performance
claim.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    add_entity_historical_features,
    apply_frequency_encoder,
    build_collapsed_indicator_pipeline,
    build_entity_history_train_and_validation,
    collapsed_indicator_representatives,
    compare_entity_key_candidates,
    compute_interaction_and_amount_features,
    compute_temporal_phase_features,
    find_duplicate_indicator_groups,
    fit_frequency_encoder_on_train,
)
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    load_included_columns,
    load_partition_df,
    measure_train_missingness,
    select_indicator_columns,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

ENTITY_COL = "card1"  # a documented, revisable choice -- see report §2


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {}
    assignment = pd.read_csv(ASSIGNMENT_PATH)

    section("1. Temporal phase features (train+validation, never final_test)")
    ids = set(get_partition(assignment, ["train", "validation"])["TransactionID"])
    dt_df = pd.read_csv(TRAIN_TXN, usecols=["TransactionID", "TransactionDT"])
    dt_df = dt_df.loc[dt_df["TransactionID"].isin(ids)]
    t0 = time.perf_counter()
    dt_df = compute_temporal_phase_features(dt_df)
    elapsed = time.perf_counter() - t0
    hour_dist = dt_df["txn_hour_phase"].value_counts(normalize=True).sort_index()
    dow_dist = dt_df["txn_dow_phase"].value_counts(normalize=True).sort_index()
    print(f"Computed on {len(dt_df)} rows in {elapsed:.3f}s")
    print("Hour-phase distribution (relative, NOT calendar hour):")
    print((hour_dist * 100).round(2))
    report["temporal_phase"] = {
        "n_rows": len(dt_df),
        "elapsed_seconds": round(elapsed, 3),
        "hour_phase_min_max_pct": [round(hour_dist.min() * 100, 2), round(hour_dist.max() * 100, 2)],
        "dow_phase_min_max_pct": [round(dow_dist.min() * 100, 2), round(dow_dist.max() * 100, 2)],
    }

    section(f"2. Entity historical features, entity_col='{ENTITY_COL}' (train+validation, never final_test)")
    t0 = time.perf_counter()
    entity_hist = build_entity_history_train_and_validation(ASSIGNMENT_PATH, TRAIN_TXN, ENTITY_COL)
    elapsed = time.perf_counter() - t0
    count_col = f"{ENTITY_COL}_prior_txn_count"
    recency_col = f"{ENTITY_COL}_time_since_prior_txn"
    pct_first_seen = (entity_hist[count_col] == 0).mean() * 100
    print(f"Computed on {len(entity_hist)} rows in {elapsed:.3f}s")
    print(f"% rows that are their entity's FIRST observed transaction (prior_count==0): {pct_first_seen:.2f}%")
    print(f"prior_count distribution: {entity_hist[count_col].describe()}")
    report["entity_history"] = {
        "entity_col": ENTITY_COL,
        "n_rows": len(entity_hist),
        "elapsed_seconds": round(elapsed, 3),
        "pct_first_seen": round(pct_first_seen, 2),
        "prior_count_mean": round(float(entity_hist[count_col].mean()), 3),
        "prior_count_max": int(entity_hist[count_col].max()),
        "recency_non_null_pct": round(entity_hist[recency_col].notna().mean() * 100, 2),
    }

    section("2b. Train-row consistency check on REAL data (not just the synthetic test)")
    train_ids = set(get_partition(assignment, ["train"])["TransactionID"])
    train_rows_in_combined = entity_hist.loc[entity_hist["TransactionID"].isin(train_ids)]
    train_only_df = pd.read_csv(TRAIN_TXN, usecols=["TransactionID", "TransactionDT", ENTITY_COL])
    train_only_df = train_only_df.loc[train_only_df["TransactionID"].isin(train_ids)]
    train_only_hist = add_entity_historical_features(train_only_df, ENTITY_COL)
    merged_check = train_rows_in_combined[["TransactionID", count_col]].merge(
        train_only_hist[["TransactionID", count_col]], on="TransactionID", suffixes=("_combined", "_train_only")
    )
    all_match = (merged_check[f"{count_col}_combined"] == merged_check[f"{count_col}_train_only"]).all()
    print(f"All {len(merged_check)} train rows' prior_count identical whether computed combined-with-validation or train-alone: {all_match}")
    report["train_consistency_check_real_data"] = {"n_rows_checked": len(merged_check), "all_identical": bool(all_match)}

    section("2c. Confirm final_test is refused without explicit override")
    try:
        get_partition(assignment, ["train", "validation", "final_test"])
        report["final_test_refused"] = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        report["final_test_refused"] = True

    section("2d. Entity-key structural comparison (review items 1/2) -- no target, no model")
    key_cols = ["TransactionID", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"]
    key_df = pd.read_csv(TRAIN_TXN, usecols=key_cols)
    key_df = key_df.loc[key_df["TransactionID"].isin(ids)]
    candidates = {
        "card1": ["card1"],
        "card1+card2": ["card1", "card2"],
        "card1..card6": ["card1", "card2", "card3", "card4", "card5", "card6"],
        "card1+card2+addr1": ["card1", "card2", "addr1"],
        "card1+card2+addr1+addr2": ["card1", "card2", "addr1", "addr2"],
        "card1..card6+addr1+addr2": ["card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"],
    }
    key_comparison = compare_entity_key_candidates(key_df, candidates)
    for name, stats in key_comparison.items():
        print(f"{name}: {stats}")
    report["entity_key_structural_comparison"] = key_comparison

    section("3. Frequency encoding, train-only fit (P_emaildomain, R_emaildomain)")
    freq_report = {}
    for col in ("P_emaildomain", "R_emaildomain"):
        mapping = fit_frequency_encoder_on_train(ASSIGNMENT_PATH, TRAIN_TXN, col)
        val_ids = set(get_partition(assignment, ["validation"])["TransactionID"])
        val_df = pd.read_csv(TRAIN_TXN, usecols=["TransactionID", col])
        val_series = val_df.loc[val_df["TransactionID"].isin(val_ids), col]
        unseen_pct = (~val_series.isin(mapping.keys())).mean() * 100
        encoded = apply_frequency_encoder(val_series, mapping)
        print(f"{col}: {len(mapping)} categories seen in train; {unseen_pct:.2f}% of validation rows have a category unseen in train (mapped to 0)")
        freq_report[col] = {
            "n_train_categories": len(mapping),
            "pct_validation_unseen": round(unseen_pct, 2),
            "encoded_has_no_nan": bool(not encoded.isna().any()),
        }
    report["frequency_encoding"] = freq_report

    section("4. Interaction / amount-derived features (train+validation)")
    cols_needed = ["TransactionID", "P_emaildomain", "R_emaildomain", "ProductCD", "card4", "TransactionAmt"]
    interaction_df = pd.read_csv(TRAIN_TXN, usecols=cols_needed)
    interaction_df = interaction_df.loc[interaction_df["TransactionID"].isin(ids)]
    t0 = time.perf_counter()
    interaction_df = compute_interaction_and_amount_features(interaction_df)
    elapsed = time.perf_counter() - t0
    match_rate = interaction_df["email_domain_match"].mean(skipna=True) * 100
    match_coverage = interaction_df["email_domain_match"].notna().mean() * 100
    round_rate = interaction_df["transaction_amt_is_round"].mean() * 100
    print(f"Computed on {len(interaction_df)} rows in {elapsed:.3f}s")
    print(f"email_domain_match: {match_coverage:.2f}% of rows have both domains present; of those, {match_rate:.2f}% match")
    print(f"transaction_amt_is_round: {round_rate:.2f}% of rows have a whole-dollar amount")
    report["interaction_features"] = {
        "n_rows": len(interaction_df),
        "elapsed_seconds": round(elapsed, 3),
        "email_match_coverage_pct": round(match_coverage, 2),
        "email_match_rate_pct": round(match_rate, 2),
        "round_amount_pct": round(round_rate, 2),
    }

    section("5. Duplicate missingness-indicator groups -- recomputed FRESH (not hardcoded from Stage 5)")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_with_indicator = indicator_split["with_indicator"]
    train_df_for_indicators = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", numeric_with_indicator)
    t0 = time.perf_counter()
    groups = find_duplicate_indicator_groups(train_df_for_indicators, numeric_with_indicator)
    reps = collapsed_indicator_representatives(groups)
    elapsed = time.perf_counter() - t0
    print(f"{len(numeric_with_indicator)} indicator columns -> {len(reps)} unique missingness patterns (recomputed fresh in {elapsed:.3f}s)")
    report["duplicate_indicator_groups"] = {
        "n_indicator_columns": len(numeric_with_indicator),
        "n_unique_patterns_recomputed": len(reps),
        "matches_stage5_finding_of_26": len(reps) == 26,
    }

    section("5b. Collapsed-indicator prototype pipeline -- shape check only, not fit for production use")
    ct = build_collapsed_indicator_pipeline(numeric_with_indicator, reps)
    t0 = time.perf_counter()
    output = ct.fit_transform(train_df_for_indicators[numeric_with_indicator])
    elapsed = time.perf_counter() - t0
    dense_shape = output.shape
    print(f"Collapsed prototype output shape: {dense_shape} (vs. Stage 5 baseline's {len(numeric_with_indicator)*2} for this block) in {elapsed:.3f}s")
    report["collapsed_indicator_prototype"] = {
        "output_shape": list(dense_shape),
        "baseline_stage5_block_width": len(numeric_with_indicator) * 2,
        "elapsed_seconds": round(elapsed, 3),
    }

    section("6. Recheck all 11 engineered features against the Stage 4 availability/leakage register (review item 8)")
    eng_register = pd.read_csv(DOCS_DIR / "engineered_features_register.csv")
    raw_register = pd.read_csv(REGISTER_PATH)
    raw_status = dict(zip(raw_register["column"], raw_register["status"]))
    register_check = {}
    for _, row in eng_register.iterrows():
        deps = [d.strip() for d in str(row["source_columns"]).split(",") if d.strip() in raw_status]
        statuses = {d: raw_status[d] for d in deps}
        excluded = {d: s for d, s in statuses.items() if s.startswith("exclude")}
        # TransactionID is used by the entity-history features only as a sort
        # tie-breaker (never exposed as a feature value) -- its own
        # exclude_structural status governs predictive-feature use, not this
        # internal bookkeeping use (same precedent as TransactionDT's use as
        # a sort/partition key throughout this project). Documented in
        # docs/engineered_features_register.csv's notes for these two rows.
        excluded = {d: s for d, s in excluded.items() if d != "TransactionID"}
        register_check[row["feature_name"]] = {"source_statuses": statuses, "unexplained_excluded_leak": excluded}
        flag = " <-- UNEXPLAINED EXCLUDED DEPENDENCY" if excluded else ""
        print(f"{row['feature_name']}: {statuses}{flag}")
    report["stage4_register_recheck"] = register_check
    any_unexplained_leak = any(v["unexplained_excluded_leak"] for v in register_check.values())
    print(f"\nAny unexplained excluded-column dependency across all 11 engineered features: {any_unexplained_leak}")
    report["any_unexplained_excluded_dependency"] = any_unexplained_leak

    out_path = DOCS_DIR / "_stage6_feature_engineering_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
