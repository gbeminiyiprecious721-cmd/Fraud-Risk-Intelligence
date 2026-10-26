"""Stage 9 — chronological cross-validation design for hyperparameter tuning.

DESIGN-PHASE module: builds and verifies train-only, time-respecting CV
folds, and fits preprocessing/model per fold (never train-wide) -- but
contains no code path that reads `validation` or `final_test`, and this
stage has NOT been approved to execute a full tuning search. Everything
here operates on whatever train-partition rows the caller passes in.

Why not plain `sklearn.model_selection.TimeSeriesSplit`: it divides
pre-sorted rows into equal-size chunks by ROW COUNT, with no awareness of
tied timestamps. Measured on this project's real `train` partition
(13,332 duplicate-timestamp groups covering 27,056 of 438,125 rows), its
default row-count boundaries happen to avoid splitting a tied group for
n_splits in {3..8}, but DO split one at n_splits=10 -- a coincidence of
this dataset's specific tie distribution, not a guarantee `TimeSeriesSplit`
makes for any other n_splits. `assess_timeseries_split_compatibility()`
checks this empirically for whatever n_splits is actually chosen, rather
than assuming it's safe; `build_expanding_window_folds()` below is the
deterministic, tie-safe alternative used instead.

No embargo/gap is applied between a fold's train and validation chunks.
This is a stated, reasoned choice, not an oversight: no label-maturity
delay (e.g. "a fraud label isn't confirmed for N days after the
transaction") was established anywhere in this project's Stage 2/3 EDA.
Inventing a specific gap size without evidence would be an arbitrary
choice dressed up as caution -- worse than no gap, per explicit
instruction. If a real label-maturity assumption is confirmed later, an
embargo should be added then, with its size traceable to that evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.preprocessing import build_preprocessing_pipeline, select_indicator_columns

MIN_VAL_POSITIVE_FOR_AP = 20  # a floor below which AP on that fold is too noisy to trust


def sort_chronologically(
    df: pd.DataFrame, dt_col: str = "TransactionDT", id_col: str = "TransactionID"
) -> pd.DataFrame:
    """Stable sort by (dt_col, id_col), reset to a clean 0..n-1 positional
    index -- every function below operates on POSITIONAL indices into this
    sorted frame, never on the original (unsorted) index."""
    return df.sort_values([dt_col, id_col], kind="stable").reset_index(drop=True)


def find_tie_groups(df_sorted: pd.DataFrame, dt_col: str = "TransactionDT") -> dict:
    """Measured (not assumed) duplicate-timestamp structure of the ALREADY
    chronologically-sorted frame passed in."""
    counts = df_sorted[dt_col].value_counts()
    tied = counts[counts > 1]
    return {
        "n_unique_timestamps": int(counts.shape[0]),
        "n_tied_timestamp_groups": int(tied.shape[0]),
        "n_rows_in_a_tie": int(tied.sum()),
        "max_rows_sharing_one_timestamp": int(counts.max()) if len(counts) else 0,
    }


def assess_timeseries_split_compatibility(
    df_sorted: pd.DataFrame, n_splits: int, dt_col: str = "TransactionDT"
) -> dict:
    """Run sklearn's actual `TimeSeriesSplit(n_splits)` over `df_sorted` and
    check, per fold, whether its row-count boundary happens to fall inside a
    tied-timestamp group -- measured empirically for THIS n_splits, never
    assumed safe from a different n_splits' result."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_reports = []
    any_straddle = False
    for i, (train_idx, val_idx) in enumerate(tss.split(df_sorted)):
        boundary_dt = df_sorted.loc[train_idx[-1], dt_col]
        next_dt = df_sorted.loc[val_idx[0], dt_col]
        straddled = bool(boundary_dt == next_dt)
        any_straddle = any_straddle or straddled
        fold_reports.append({
            "fold": i, "train_n": len(train_idx), "val_n": len(val_idx),
            "boundary_dt": float(boundary_dt), "next_val_dt": float(next_dt), "tie_straddled": straddled,
        })
    return {"n_splits": n_splits, "any_tie_straddled": any_straddle, "folds": fold_reports}


def snap_boundary_forward(df_sorted: pd.DataFrame, idx: int, dt_col: str = "TransactionDT") -> int:
    """If position `idx` would split a tied-timestamp group (the row just
    before `idx` shares `idx`'s own timestamp), advance `idx` forward until
    it lands exactly on a genuine timestamp change. Advancing FORWARD (into
    train, out of validation) is the conservative direction: it can only
    make a fold's training data larger and its validation start later,
    never pull a future row backward into training.
    """
    n = len(df_sorted)
    if idx <= 0 or idx >= n:
        return idx
    while idx < n and df_sorted.loc[idx - 1, dt_col] == df_sorted.loc[idx, dt_col]:
        idx += 1
    return idx


def build_expanding_window_folds(
    df: pd.DataFrame, n_splits: int, dt_col: str = "TransactionDT", id_col: str = "TransactionID"
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """Deterministic, tie-safe expanding-window chronological folds.

    Returns `(df_sorted, folds)` where `df_sorted` is `df` chronologically
    sorted and reset-indexed, and `folds` is a list of `(train_idx, val_idx)`
    POSITIONAL-index arrays into `df_sorted`. Fold i's train set is
    `df_sorted[:train_end_i]` (expanding -- always starts at row 0, the
    earliest `train` row) and its validation set is the next chunk,
    `df_sorted[train_end_i:val_end_i]` -- both boundaries snapped forward
    past any tied-timestamp group via `snap_boundary_forward`, so no tie
    group is ever split across train/validation, and train_end_i is always
    reached before val_end_i (train strictly precedes validation in time
    by construction of a stable sort plus a forward-only snap).
    """
    df_sorted = sort_chronologically(df, dt_col, id_col)
    n = len(df_sorted)
    chunk = n // (n_splits + 1)
    folds = []
    for i in range(n_splits):
        naive_train_end = chunk * (i + 1)
        naive_val_end = chunk * (i + 2) if i < n_splits - 1 else n
        train_end = snap_boundary_forward(df_sorted, naive_train_end, dt_col)
        val_end = snap_boundary_forward(df_sorted, naive_val_end, dt_col)
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        folds.append((train_idx, val_idx))
    return df_sorted, folds


def verify_fold(
    df_sorted: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dt_col: str = "TransactionDT",
    label_col: str = "isFraud",
    min_val_positive: int = MIN_VAL_POSITIVE_FOR_AP,
) -> dict:
    """All of this stage's fold-level requirements, checked against the
    ACTUAL rows in this fold -- never assumed from how the fold was built."""
    train_dt = df_sorted.loc[train_idx, dt_col]
    val_dt = df_sorted.loc[val_idx, dt_col]
    val_labels = df_sorted.loc[val_idx, label_col]
    chronology_ok = bool(train_dt.max() < val_dt.min())
    no_tie_straddle = bool(train_dt.max() != val_dt.min())
    n_val_positive = int(val_labels.sum())
    sufficient_positives = n_val_positive >= min_val_positive
    report = {
        "train_n_rows": int(len(train_idx)),
        "val_n_rows": int(len(val_idx)),
        "train_dt_range": [float(train_dt.min()), float(train_dt.max())],
        "val_dt_range": [float(val_dt.min()), float(val_dt.max())],
        "val_n_positive": n_val_positive,
        "val_fraud_prevalence": round(float(val_labels.mean()), 5),
        "chronology_ok": chronology_ok,
        "no_tie_straddle": no_tie_straddle,
        "sufficient_val_positives": sufficient_positives,
    }
    report["all_checks_passed"] = chronology_ok and no_tie_straddle and sufficient_positives
    return report


# ---------------------------------------------------------------------------
# Per-fold, leakage-safe preprocessing -- fit on THIS fold's train rows only
# ---------------------------------------------------------------------------

def measure_missingness_pct(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """In-memory analogue of `src.preprocessing.measure_train_missingness`,
    operating on whatever rows are passed in (a fold's own training rows)
    instead of re-reading the full approved `train` partition from disk.
    Deliberately NOT added to `src/preprocessing.py` itself -- that file is
    an approved Stage 5 deliverable, left unmodified; this is Stage 9-only,
    new code for a new (per-fold) use case.
    """
    return (df[columns].isna().mean() * 100).round(3).sort_values(ascending=False)


def fit_transform_fold(
    df_sorted: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    numeric_columns: list[str],
    categorical_columns: list[str],
    label_col: str = "isFraud",
    indicator_threshold_pct: float = 5.0,
) -> dict:
    """Fit the APPROVED, UNMODIFIED `build_preprocessing_pipeline` structure
    on THIS FOLD'S train rows only -- including re-deciding which numeric
    columns get a missingness indicator (`select_indicator_columns`), since
    that decision is itself a statistic of whatever rows it's measured on.
    Using the Stage-5-approved split (computed once, over the ENTIRE `train`
    partition) here would let a fold "see" the missingness pattern of rows
    from LATER folds -- a real, if subtle, leakage path this function closes
    by recomputing it fresh, per fold, from only that fold's own training
    rows. The raw numeric/categorical COLUMN SET itself (not the
    indicator/no-indicator split within it) stays fixed -- that is a
    structural classification from the approved Stage 4/5 register, not a
    fitted statistic, and is correctly reused unchanged.
    """
    train_df = df_sorted.loc[train_idx]
    val_df = df_sorted.loc[val_idx]

    fold_missingness = measure_missingness_pct(train_df, numeric_columns)
    fold_indicator_split = select_indicator_columns(fold_missingness, threshold_pct=indicator_threshold_pct)

    ct = build_preprocessing_pipeline(
        fold_indicator_split["without_indicator"], fold_indicator_split["with_indicator"], categorical_columns
    )
    all_cols = fold_indicator_split["without_indicator"] + fold_indicator_split["with_indicator"] + categorical_columns
    X_train = ct.fit_transform(train_df[all_cols])
    X_val = ct.transform(val_df[all_cols])
    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": train_df[label_col].to_numpy(),
        "y_val": val_df[label_col].to_numpy(),
        "fitted_feature_count": X_train.shape[1],
        "numeric_with_indicator": fold_indicator_split["with_indicator"],
        "numeric_without_indicator": fold_indicator_split["without_indicator"],
    }


# ---------------------------------------------------------------------------
# Random Forest search space -- modest, explicit, justified relative to the
# Stage 7 reference config (n_estimators=100, max_depth=10,
# min_samples_leaf=20, random_state=42)
# ---------------------------------------------------------------------------

RF_SEARCH_SPACE = {
    # Stage 7 used 100 fixed. A small range around it, not an unrelated
    # library default -- lower values cut cost roughly linearly; 100 is
    # already in the typical "diminishing returns" zone for RF bagging.
    "n_estimators": [50, 100, 150],
    # Stage 7's max_depth=10 was a bounded, not-yet-empirically-justified
    # choice (stated explicitly in Stage 7's own docstring). This range
    # brackets it on both sides.
    "max_depth": [6, 8, 10, 12, 15],
    # Stage 7's min_samples_leaf=20 controls overfitting on a ~3.5%-positive
    # problem; too small risks fitting noise in the minority class, too
    # large risks washing out real fraud patterns.
    "min_samples_leaf": [5, 10, 20, 40],
    # Not varied in Stage 7 at all (sklearn default "sqrt" was implicit).
    # Explicitly included here since it directly trades split quality
    # against fit time -- a real, previously-unexamined knob.
    "max_features": ["sqrt", "log2", None],
}


def sample_rf_candidates(n_candidates: int, random_state: int, search_space: dict | None = None) -> list[dict]:
    """Deterministic (fixed `random_state`) random sample of `n_candidates`
    distinct hyperparameter combinations from `search_space` (default:
    `RF_SEARCH_SPACE`), via sklearn's own `ParameterSampler` -- no new
    dependency, and the same mechanism `RandomizedSearchCV` itself uses
    internally, so this is a preview of its actual behavior, not a
    reimplementation of it.
    """
    from sklearn.model_selection import ParameterSampler

    space = search_space or RF_SEARCH_SPACE
    sampler = ParameterSampler(space, n_iter=n_candidates, random_state=random_state)
    return list(sampler)


# ---------------------------------------------------------------------------
# Feasibility estimation -- structural, from MEASURED per-fold costs
# ---------------------------------------------------------------------------

def estimate_search_cost(
    measured_per_fold_seconds: float, n_candidates: int, n_folds: int, safety_margin: float = 1.5
) -> dict:
    """Pure arithmetic over an ALREADY-MEASURED per-(candidate x fold) cost
    -- never itself measures anything. `measured_per_fold_seconds` should
    come from a real, small, bounded probe (see
    `scripts/run_stage9_cv_design_and_feasibility.py`), explicitly labeled
    as a LOWER BOUND when the probe used fewer rows than the real search
    would -- this function does not know or check that; the caller must
    state it.
    """
    total_fits = n_candidates * n_folds
    naive_estimate_seconds = total_fits * measured_per_fold_seconds
    conservative_estimate_seconds = naive_estimate_seconds * safety_margin
    return {
        "n_candidates": n_candidates,
        "n_folds": n_folds,
        "total_fits": total_fits,
        "measured_per_fold_seconds": round(measured_per_fold_seconds, 3),
        "naive_estimate_seconds": round(naive_estimate_seconds, 1),
        "safety_margin": safety_margin,
        "conservative_estimate_seconds": round(conservative_estimate_seconds, 1),
    }
