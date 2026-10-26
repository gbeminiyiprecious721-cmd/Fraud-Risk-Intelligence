import numpy as np
import pandas as pd
import pytest

from src.hyperparameter_tuning import (
    RF_SEARCH_SPACE,
    assess_timeseries_split_compatibility,
    build_expanding_window_folds,
    estimate_search_cost,
    find_tie_groups,
    fit_transform_fold,
    measure_missingness_pct,
    sample_rf_candidates,
    snap_boundary_forward,
    sort_chronologically,
    verify_fold,
)


def make_synthetic_df(n=100, seed=0, tie_at=None, tie_width=4):
    """A small synthetic chronological dataset. `tie_at` optionally forces
    `tie_width` consecutive rows (by sort order) to share one timestamp,
    for deliberately exercising the tie-straddle logic."""
    rng = np.random.default_rng(seed)
    dt = np.arange(n) * 100  # evenly spaced, distinct by default
    if tie_at is not None:
        dt[tie_at : tie_at + tie_width] = dt[tie_at]
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": dt,
        "isFraud": (rng.random(n) < 0.3).astype(int),
        "num_a": rng.normal(size=n),
        "cat_a": rng.choice(["x", "y"], size=n),
    })
    # Shuffle row order to prove sorting is actually applied, not assumed.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chronological sort + tie detection
# ---------------------------------------------------------------------------

def test_sort_chronologically_produces_nondecreasing_dt():
    df = make_synthetic_df(n=50, seed=1)
    sorted_df = sort_chronologically(df)
    assert (sorted_df["TransactionDT"].diff().dropna() >= 0).all()


def test_find_tie_groups_detects_forced_tie():
    df = make_synthetic_df(n=50, seed=2, tie_at=10, tie_width=5)
    sorted_df = sort_chronologically(df)
    report = find_tie_groups(sorted_df)
    assert report["n_tied_timestamp_groups"] >= 1
    assert report["max_rows_sharing_one_timestamp"] >= 5


def test_find_tie_groups_reports_zero_for_all_distinct_timestamps():
    df = make_synthetic_df(n=50, seed=3)
    sorted_df = sort_chronologically(df)
    report = find_tie_groups(sorted_df)
    assert report["n_tied_timestamp_groups"] == 0


# ---------------------------------------------------------------------------
# TimeSeriesSplit compatibility assessment
# ---------------------------------------------------------------------------

def test_assess_timeseries_split_compatibility_flags_a_known_straddle():
    """Force a tie exactly where a 2-split boundary would naturally fall,
    and confirm the assessment catches it rather than reporting false
    safety."""
    n = 40
    df = make_synthetic_df(n=n, seed=4)
    sorted_df = sort_chronologically(df)
    # TimeSeriesSplit(n_splits=2) on 40 rows splits at index ~13, ~26.
    # Force a tie spanning position 13 (one of those exact boundaries).
    sorted_df.loc[12:14, "TransactionDT"] = sorted_df.loc[13, "TransactionDT"]
    report = assess_timeseries_split_compatibility(sorted_df, n_splits=2)
    assert report["any_tie_straddled"] is True


def test_assess_timeseries_split_compatibility_reports_false_when_clean():
    df = make_synthetic_df(n=100, seed=5)  # all-distinct timestamps
    sorted_df = sort_chronologically(df)
    report = assess_timeseries_split_compatibility(sorted_df, n_splits=5)
    assert report["any_tie_straddled"] is False


# ---------------------------------------------------------------------------
# snap_boundary_forward
# ---------------------------------------------------------------------------

def test_snap_boundary_forward_advances_past_a_tie():
    df = make_synthetic_df(n=30, seed=6, tie_at=10, tie_width=4)
    sorted_df = sort_chronologically(df)
    # Naive boundary lands inside the tie (positions 10-13 share one dt).
    snapped = snap_boundary_forward(sorted_df, 12, "TransactionDT")
    assert snapped >= 14
    assert sorted_df.loc[snapped - 1, "TransactionDT"] != sorted_df.loc[snapped, "TransactionDT"]


def test_snap_boundary_forward_is_a_noop_when_already_clean():
    df = make_synthetic_df(n=30, seed=7)
    sorted_df = sort_chronologically(df)
    assert snap_boundary_forward(sorted_df, 15, "TransactionDT") == 15


def test_snap_boundary_forward_handles_edge_indices():
    df = make_synthetic_df(n=30, seed=8)
    sorted_df = sort_chronologically(df)
    assert snap_boundary_forward(sorted_df, 0, "TransactionDT") == 0
    assert snap_boundary_forward(sorted_df, len(sorted_df), "TransactionDT") == len(sorted_df)


# ---------------------------------------------------------------------------
# build_expanding_window_folds -- the core deterministic, tie-safe design
# ---------------------------------------------------------------------------

def test_build_expanding_window_folds_never_straddles_a_forced_tie():
    df = make_synthetic_df(n=200, seed=9, tie_at=33, tie_width=6)
    df_sorted, folds = build_expanding_window_folds(df, n_splits=5)
    for train_idx, val_idx in folds:
        train_dt = df_sorted.loc[train_idx, "TransactionDT"]
        val_dt = df_sorted.loc[val_idx, "TransactionDT"]
        assert train_dt.max() != val_dt.min(), "a tie group was split across train/validation"


def test_build_expanding_window_folds_train_strictly_precedes_validation():
    df = make_synthetic_df(n=300, seed=10)
    df_sorted, folds = build_expanding_window_folds(df, n_splits=4)
    for train_idx, val_idx in folds:
        assert df_sorted.loc[train_idx, "TransactionDT"].max() < df_sorted.loc[val_idx, "TransactionDT"].min()


def test_build_expanding_window_folds_is_expanding():
    """Each successive fold's train set should be at least as large as the
    previous one's (expanding window, not a fixed-size rolling one)."""
    df = make_synthetic_df(n=300, seed=11)
    _, folds = build_expanding_window_folds(df, n_splits=4)
    train_sizes = [len(train_idx) for train_idx, _ in folds]
    assert train_sizes == sorted(train_sizes)


def test_build_expanding_window_folds_covers_only_rows_from_the_input_df():
    df = make_synthetic_df(n=150, seed=12)
    df_sorted, folds = build_expanding_window_folds(df, n_splits=3)
    for train_idx, val_idx in folds:
        assert max(train_idx.max(), val_idx.max()) < len(df_sorted)
        assert set(train_idx) & set(val_idx) == set()


# ---------------------------------------------------------------------------
# verify_fold
# ---------------------------------------------------------------------------

def test_verify_fold_passes_for_a_well_formed_fold():
    df = make_synthetic_df(n=500, seed=13)
    df_sorted, folds = build_expanding_window_folds(df, n_splits=3)
    train_idx, val_idx = folds[-1]  # largest, most likely to have enough positives
    report = verify_fold(df_sorted, train_idx, val_idx, min_val_positive=1)
    assert report["chronology_ok"] is True
    assert report["no_tie_straddle"] is True


def test_verify_fold_flags_insufficient_positives():
    df = make_synthetic_df(n=30, seed=14)
    df.loc[:, "isFraud"] = 0  # force zero positives anywhere
    df_sorted, folds = build_expanding_window_folds(df, n_splits=2)
    train_idx, val_idx = folds[0]
    report = verify_fold(df_sorted, train_idx, val_idx, min_val_positive=1)
    assert report["sufficient_val_positives"] is False
    assert report["all_checks_passed"] is False


# ---------------------------------------------------------------------------
# measure_missingness_pct + fit_transform_fold -- per-fold leakage guard
# ---------------------------------------------------------------------------

def test_measure_missingness_pct_matches_manual_computation():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1.0, 2.0, 3.0, 4.0]})
    result = measure_missingness_pct(df, ["a", "b"])
    assert result["a"] == 50.0
    assert result["b"] == 0.0


def test_fit_transform_fold_fits_preprocessing_on_train_rows_only():
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.arange(n) * 10,
        "isFraud": (rng.random(n) < 0.3).astype(int),
        "num_plain": rng.normal(size=n),
        "num_missingish": np.where(rng.random(n) < 0.5, rng.normal(size=n), np.nan),
        "cat_a": rng.choice(["x", "y", "z"], size=n),
    })
    df_sorted, folds = build_expanding_window_folds(df, n_splits=2)
    train_idx, val_idx = folds[0]
    result = fit_transform_fold(df_sorted, train_idx, val_idx, ["num_plain", "num_missingish"], ["cat_a"])
    assert result["X_train"].shape[0] == len(train_idx)
    assert result["X_val"].shape[0] == len(val_idx)
    assert result["X_train"].shape[1] == result["X_val"].shape[1] == result["fitted_feature_count"]
    assert not np.isnan(result["X_train"].toarray() if hasattr(result["X_train"], "toarray") else result["X_train"]).any()


def test_fit_transform_fold_indicator_split_can_differ_between_folds():
    """Different folds' training rows can have different missingness
    patterns, which should produce different indicator/no-indicator splits
    -- direct evidence that the split is recomputed per fold, not reused
    from one global (potentially leaking) computation."""
    n = 300
    rng = np.random.default_rng(1)
    dt = np.arange(n) * 10
    # Column that is heavily missing only in the LATER half of the data.
    missing_col = np.where(np.arange(n) < n // 2, rng.normal(size=n), np.nan)
    df = pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": dt,
        "isFraud": (rng.random(n) < 0.3).astype(int),
        "num_plain": rng.normal(size=n), "num_time_varying_missing": missing_col,
        "cat_a": rng.choice(["x", "y"], size=n),
    })
    df_sorted, folds = build_expanding_window_folds(df, n_splits=2)
    early_train_idx, early_val_idx = folds[0]
    late_train_idx, late_val_idx = folds[-1]
    early = fit_transform_fold(df_sorted, early_train_idx, early_val_idx, ["num_plain", "num_time_varying_missing"], ["cat_a"])
    late = fit_transform_fold(df_sorted, late_train_idx, late_val_idx, ["num_plain", "num_time_varying_missing"], ["cat_a"])
    assert early["numeric_with_indicator"] != late["numeric_with_indicator"]


# ---------------------------------------------------------------------------
# RF search space sampling
# ---------------------------------------------------------------------------

def test_sample_rf_candidates_is_deterministic_with_fixed_seed():
    a = sample_rf_candidates(5, random_state=42)
    b = sample_rf_candidates(5, random_state=42)
    assert a == b


def test_sample_rf_candidates_draws_from_declared_space():
    candidates = sample_rf_candidates(10, random_state=0)
    assert len(candidates) == 10
    for c in candidates:
        assert c["n_estimators"] in RF_SEARCH_SPACE["n_estimators"]
        assert c["max_depth"] in RF_SEARCH_SPACE["max_depth"]
        assert c["min_samples_leaf"] in RF_SEARCH_SPACE["min_samples_leaf"]
        assert c["max_features"] in RF_SEARCH_SPACE["max_features"]


# ---------------------------------------------------------------------------
# Feasibility arithmetic
# ---------------------------------------------------------------------------

def test_estimate_search_cost_arithmetic():
    result = estimate_search_cost(measured_per_fold_seconds=2.0, n_candidates=10, n_folds=5, safety_margin=1.5)
    assert result["total_fits"] == 50
    assert result["naive_estimate_seconds"] == 100.0
    assert result["conservative_estimate_seconds"] == 150.0


def test_estimate_search_cost_never_measures_anything_itself():
    """Pure function: calling it twice with the same inputs gives the same
    output -- no hidden randomness or real timing inside it."""
    a = estimate_search_cost(1.23, 7, 3)
    b = estimate_search_cost(1.23, 7, 3)
    assert a == b
