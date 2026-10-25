"""Stage 6 feature engineering — IEEE-CIS Fraud Detection data.

Design principles (all enforced, not just stated):
- This module is ADDITIVE: it never imports from or modifies
  `src/preprocessing.py`. The approved Stage 5 baseline pipeline
  (`build_preprocessing_pipeline`) is untouched — every function here
  produces new, separate columns/artifacts a later stage can choose to
  combine with Stage 5's output.
- Every function that computes a per-row "historical" feature is causal by
  construction: a row's feature value depends only on rows with STRICTLY
  earlier `TransactionDT`, never on rows with an equal or later timestamp
  (including itself). This is verified by tests, not just claimed.
- Historical features may legitimately look across the train/validation
  boundary (validation is chronologically AFTER train, so a validation
  row's history correctly includes train rows) but must NEVER touch
  `final_test` — every entry point here routes through
  `src.leakage_boundaries.get_partition()` with `allow_final_test` left at
  its default `False`.
- No function in this module trains, fits, or scores a model, and none
  claims a feature "improves" anything — that comparison is explicitly
  Stage 7's job (per instruction). Functions here only construct features
  and report their own structural properties (coverage, distribution,
  memory cost).
- `include_with_caveat`/unresolved-availability columns used as INPUTS to a
  derived feature inherit that same caveat status — a derived feature is
  never "more available" than the raw columns it's computed from.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.leakage_boundaries import get_partition  # noqa: E402

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


# ---------------------------------------------------------------------------
# 1. Temporal "phase" features — explicitly NOT calendar time
# ---------------------------------------------------------------------------

def compute_temporal_phase_features(df: pd.DataFrame, dt_col: str = "TransactionDT") -> pd.DataFrame:
    """Add `txn_hour_phase` (0-23) and `txn_dow_phase` (0-6) from `TransactionDT`.

    These are RELATIVE PHASES within a 24h/7-day modulus of the raw offset,
    not calendar hour-of-day or day-of-week. `TransactionDT`'s absolute
    origin (epoch, timezone) is unconfirmed (Stage 2 Known Limitations) —
    so "phase 14" means "14 hours into whatever 24h cycle the offset
    implies," not "2pm." If the data happens to be captured starting at a
    cycle boundary this phase will coincide with real hour-of-day, but that
    coincidence is NOT assumed or asserted anywhere in this codebase.
    """
    out = df.copy()
    out["txn_hour_phase"] = (out[dt_col] // SECONDS_PER_HOUR) % 24
    out["txn_dow_phase"] = (out[dt_col] // SECONDS_PER_DAY) % 7
    return out


# ---------------------------------------------------------------------------
# 2. Causal, point-in-time entity historical features
# ---------------------------------------------------------------------------

def add_entity_historical_features(
    df: pd.DataFrame,
    entity_col: str,
    dt_col: str = "TransactionDT",
    id_col: str = "TransactionID",
    count_feature_name: str | None = None,
    recency_feature_name: str | None = None,
) -> pd.DataFrame:
    """Add a strictly-prior transaction count and time-since-prior-transaction
    per `entity_col`, vectorized and tie-safe.

    For each row, `prior_count` = number of OTHER rows sharing the same
    `entity_col` value with `dt_col` STRICTLY LESS than this row's own
    `dt_col` — rows with an EQUAL timestamp (including the row itself) are
    never counted, by construction: within each entity group, rows are
    sorted by `dt_col`, and `np.searchsorted(sorted_dts, sorted_dts,
    side="left")` returns, for every position, the count of array elements
    strictly less than the value at that position. Tied values all receive
    the same count (the count of the tie-block's predecessors), so no row
    in a tie ever credits another tied row as "prior," and no row ever
    credits itself.

    `time_since_prior` is `NaN` when `prior_count == 0` (no prior
    transaction exists for this entity in the data passed in), otherwise
    the gap to the most recent strictly-prior transaction's timestamp.

    Caller is responsible for what rows are IN `df` — this function imposes
    no partition rule itself (see `build_entity_history_train_and_validation`
    below for the actual leakage-safe partition wiring); it is a pure,
    testable transformation of whatever rows it is given.
    """
    count_feature_name = count_feature_name or f"{entity_col}_prior_txn_count"
    recency_feature_name = recency_feature_name or f"{entity_col}_time_since_prior_txn"

    out = df.sort_values([entity_col, dt_col, id_col]).copy()
    prior_counts = np.empty(len(out), dtype=np.int64)
    time_since = np.full(len(out), np.nan, dtype=np.float64)

    pos = 0
    for _, group in out.groupby(entity_col, sort=False):
        n = len(group)
        dts = group[dt_col].to_numpy()
        prior_count = np.searchsorted(dts, dts, side="left")
        prior_idx = prior_count - 1
        has_prior = prior_idx >= 0
        recency_vals = np.where(has_prior, dts[np.clip(prior_idx, 0, None)], np.nan)
        group_time_since = np.where(has_prior, dts - recency_vals, np.nan)

        prior_counts[pos : pos + n] = prior_count
        time_since[pos : pos + n] = group_time_since
        pos += n

    out[count_feature_name] = prior_counts
    out[recency_feature_name] = time_since
    return out


def build_entity_history_train_and_validation(
    assignment_path: str | Path,
    transaction_path: str | Path,
    entity_col: str,
    dt_col: str = "TransactionDT",
    id_col: str = "TransactionID",
) -> pd.DataFrame:
    """The leakage-safe entry point: historical features over `train` +
    `validation` ONLY, never `final_test`.

    Why train+validation together, not train-only: a historical
    count/recency feature is a per-row, time-indexed quantity, not a single
    global statistic fit once and reused (unlike Stage 5's median/category
    parameters). A `validation` row's "legitimate" history correctly
    includes `train` rows, because `train` is chronologically BEFORE
    `validation` — excluding it would make the feature artificially,
    incorrectly sparse for the start of `validation`, not more "leakage
    safe." What must never happen is a row's history reaching into
    `final_test` or into its own future — both are enforced here:
    `get_partition()` is called with `allow_final_test` left at its
    default `False` (so `final_test` rows are never even loaded), and
    `add_entity_historical_features()`'s strictly-less-than construction
    (above) enforces the "never the future, never itself" rule row-by-row,
    unconditionally, for any input.

    Scientific caveat worth stating precisely: a TRAIN row's own feature
    value is guaranteed identical whether computed alone or combined with
    validation ONLY because Stage 4's chronological split makes every
    train row's TransactionDT strictly earlier than every validation
    row's (verified by src.leakage_boundaries.verify_split) -- so no
    validation row can ever be "strictly prior" to a train row. This is a
    property of THIS project's approved split, not a general guarantee
    this function enforces for arbitrary interleaved timestamps; tested
    as such in tests/test_feature_engineering.py.

    OPERATIONAL ASSUMPTION, stated explicitly (Stage 6 review item 5):
    treating every earlier-timestamp `validation` row as available history
    for a later `validation` row's feature is correct ONLY under a
    fixed-batch / post-hoc scoring assumption -- i.e. the entire
    `validation` window has already occurred and been fully recorded
    before ANY row in it is scored, which is how Stage 10 actually intends
    to evaluate (`validation`/`final_test` are scored as a completed,
    static file, not transaction-by-transaction in real time). This
    function's offline recomputation CANNOT verify, and does not attempt
    to verify, a stronger claim: that a genuinely live, real-time,
    point-of-sale scoring system would have that same history available
    at the moment of serving, which additionally depends on feature-store
    ingestion latency (how quickly a just-happened transaction becomes
    readable by the aggregate that computes this feature for the NEXT
    transaction). That is a separate, unresolved production-readiness
    question for Stage 13+, not addressed here -- see
    docs/FEATURE_ENGINEERING_REPORT.md Stage-6-review-round §5 for the
    full discussion.
    """
    assignment = pd.read_csv(assignment_path)
    ids = set(get_partition(assignment, ["train", "validation"])[id_col])

    df = pd.read_csv(transaction_path, usecols=[id_col, dt_col, entity_col])
    df = df.loc[df[id_col].isin(ids)]
    return add_entity_historical_features(df, entity_col, dt_col=dt_col, id_col=id_col)


# ---------------------------------------------------------------------------
# 2b. Composite entity keys and STRUCTURAL (non-target, non-model) comparison
#
# Review item 1/2: card1 alone must not be described as a unique
# customer/card identifier without evidence -- these functions let that
# evidence be gathered, and let composite candidates be compared, WITHOUT
# ever reading isFraud or fitting any model.
# ---------------------------------------------------------------------------

def build_composite_entity_key(df: pd.DataFrame, columns: list[str], missing_sentinel: str = "missing") -> pd.Series:
    """Concatenate `columns` into one string key, column values joined by
    "|". A missing value in ANY component column becomes the literal
    sentinel string (not dropped, not propagated as NaN) -- rows sharing
    the same missingness pattern across `columns` are grouped together
    under this choice, an explicit, documented decision, not a silent one.
    """
    parts = [df[c].astype("string").fillna(missing_sentinel) for c in columns]
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def describe_entity_key_structure(df: pd.DataFrame, key: pd.Series) -> dict:
    """Pure STRUCTURAL description of a candidate entity key's grouping
    behavior -- cardinality, group-size distribution, singleton rate. Does
    NOT read any target column and does NOT fit, score, or compare against
    a model; this is evidence for a human (or a later, properly-scoped
    experiment) to judge key quality by, not a verdict in itself.
    """
    sizes = key.value_counts()
    n_rows = len(key)
    return {
        "n_rows": n_rows,
        "n_distinct_entities": int(sizes.shape[0]),
        "mean_group_size": round(float(sizes.mean()), 2),
        "median_group_size": float(sizes.median()),
        "max_group_size": int(sizes.max()),
        "pct_rows_in_singleton_entities": round(float((sizes[sizes == 1].sum() / n_rows) * 100), 3),
        "pct_rows_in_groups_le_5": round(float((sizes[sizes <= 5].sum() / n_rows) * 100), 3),
    }


def compare_entity_key_candidates(
    df: pd.DataFrame, candidates: dict[str, list[str]]
) -> dict[str, dict]:
    """{candidate_name: describe_entity_key_structure(...)} for several
    candidate key definitions, e.g. {"card1": ["card1"], "card1+card2+addr1":
    ["card1", "card2", "addr1"]}. Purely structural, purely comparative --
    see module docstring and docs/FEATURE_ENGINEERING_REPORT.md for why no
    single "best" key is declared from this alone.
    """
    results = {}
    for name, cols in candidates.items():
        key = cols[0] if len(cols) == 1 else build_composite_entity_key(df, cols)
        key_series = df[cols[0]].astype("string").fillna("missing") if len(cols) == 1 else key
        results[name] = describe_entity_key_structure(df, key_series)
    return results


# ---------------------------------------------------------------------------
# 3. Frequency/count encoding — train-only fit, unseen-category safe
# ---------------------------------------------------------------------------

def fit_frequency_encoder(train_df: pd.DataFrame, column: str) -> dict:
    """Category -> occurrence count, fit on `train` ONLY. Caller must pass
    only `train`-partition rows (see `fit_frequency_encoder_on_train`)."""
    return train_df[column].value_counts(dropna=False).to_dict()


def fit_frequency_encoder_on_train(
    assignment_path: str | Path, transaction_path: str | Path, column: str
) -> dict:
    """Leakage-safe entry point: fits the frequency map on the `train`
    partition exclusively, via `get_partition()`."""
    assignment = pd.read_csv(assignment_path)
    train_ids = set(get_partition(assignment, ["train"])["TransactionID"])
    df = pd.read_csv(transaction_path, usecols=["TransactionID", column])
    train_df = df.loc[df["TransactionID"].isin(train_ids)]
    return fit_frequency_encoder(train_df, column)


def apply_frequency_encoder(series: pd.Series, mapping: dict, unseen_value: int = 0) -> pd.Series:
    """Map a column through a fitted frequency mapping. A category absent
    from `mapping` (never seen in `train`) maps to `unseen_value` (default
    0) EXPLICITLY — never raises, never silently produces `NaN`."""
    return series.map(mapping).fillna(unseen_value).astype(np.int64)


# ---------------------------------------------------------------------------
# 4. Interpretable interaction / amount-derived features — candidates only
# ---------------------------------------------------------------------------

def compute_interaction_and_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add plausible, interpretable candidate features. NONE of these are
    claimed to improve predictive performance -- that evaluation is
    explicitly Stage 7's job. Each is a documented hypothesis, not a
    validated result.

    - `email_domain_match`: 1 if `P_emaildomain` == `R_emaildomain` (both
      present and equal), 0 if both present and different, `NaN` if either
      is missing (so "unknown" is never silently conflated with "match" or
      "mismatch").
    - `productcd_card4_interaction`: string concatenation of `ProductCD`
      and `card4` — a plausible interaction for tree-based models (which
      can't natively combine categoricals without being given the cross).
    - `log1p_transaction_amt`: `log1p(TransactionAmt)` — already assessed
      in Stage 3 EDA as NOT a strong standalone signal on either scale
      (Cohen's d near-zero, sign-flipping); reintroduced here only as an
      interaction-ready derived column, not as a claim that it now matters.
    - `transaction_amt_is_round`: 1 if `TransactionAmt` has zero cents
      (`== floor(TransactionAmt)`), a documented informal fraud heuristic
      (round-dollar amounts are sometimes associated with automated/testing
      transactions) — a hypothesis to test in Stage 7, not an established
      fact for this dataset.
    """
    out = df.copy()

    p = out["P_emaildomain"]
    r = out["R_emaildomain"]
    both_present = p.notna() & r.notna()
    match = pd.Series(np.nan, index=out.index, dtype="float64")
    match.loc[both_present] = (p.loc[both_present] == r.loc[both_present]).astype(float)
    out["email_domain_match"] = match

    out["productcd_card4_interaction"] = (
        out["ProductCD"].astype("string").fillna("missing")
        + "_"
        + out["card4"].astype("string").fillna("missing")
    )

    out["log1p_transaction_amt"] = np.log1p(out["TransactionAmt"])
    out["transaction_amt_is_round"] = (out["TransactionAmt"] == np.floor(out["TransactionAmt"])).astype(int)

    return out


# ---------------------------------------------------------------------------
# 5. Duplicate-missingness-indicator reduction — PROTOTYPE, not a replacement
# ---------------------------------------------------------------------------

def find_duplicate_indicator_groups(train_df: pd.DataFrame, indicator_columns: Iterable[str]) -> dict[str, list[str]]:
    """Group `indicator_columns` by EXACT-duplicate missingness mask on
    `train_df`. Returns {representative_column: [all columns sharing that
    exact pattern, including the representative]}. A column with a unique
    pattern maps to a single-element list (itself).

    This recomputes Stage 5's finding (95% of 271 indicator columns are
    exact duplicates of 13 patterns, 26 unique patterns overall) FRESH from
    the data passed in, rather than hardcoding Stage 5's numbers — so this
    function stays correct if the register or indicator-threshold ever
    change.
    """
    indicator_columns = list(indicator_columns)
    mask = train_df[indicator_columns].isna()
    seen: dict[bytes, list[str]] = {}
    for col in indicator_columns:
        key = mask[col].to_numpy().tobytes()
        seen.setdefault(key, []).append(col)
    return {names[0]: names for names in seen.values()}


def collapsed_indicator_representatives(duplicate_groups: dict[str, list[str]]) -> list[str]:
    """The reduced set of columns that should each get ONE MissingIndicator
    column (one per unique pattern) — 26 instead of 271 at the Stage 5
    5%-threshold set, in the real-data run below."""
    return list(duplicate_groups.keys())


def build_collapsed_indicator_pipeline(numeric_with_indicator: list[str], representatives: list[str]):
    """PROTOTYPE alternate pipeline fragment: every column in
    `numeric_with_indicator` still gets its own median-imputed VALUE (no
    information about any individual column's content is lost), but only
    `representatives` (one per unique missingness pattern) gets a
    `MissingIndicator` column, instead of one per column.

    Deliberately NOT wired into `src/preprocessing.py::build_preprocessing_pipeline`
    — this is a prototype for Stage 7 to evaluate against the approved
    Stage 5 baseline, per instruction, not a replacement of it.

    Stated precisely (Stage 6 review item 7): "no information about any
    individual column's content is lost" is a claim about the MISSINGNESS-
    MASK representation only (every one of the 271 original masks is
    exactly recoverable from its representative). It is NOT a claim that
    a model trained on this collapsed representation performs the same as
    one trained on Stage 5's baseline — that is untested and unknown, and
    is explicitly Stage 7's question, not answered here.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.pipeline import Pipeline

    value_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    indicator_pipe = Pipeline([("indicator", MissingIndicator(features="missing-only"))])

    return ColumnTransformer(
        transformers=[
            ("values", value_pipe, numeric_with_indicator),
            ("collapsed_indicators", indicator_pipe, representatives),
        ],
        sparse_threshold=1.0,
    )
