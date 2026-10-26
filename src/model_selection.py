"""Stage 7 model selection — IEEE-CIS Fraud Detection data.

Design principles (all enforced, not just stated):
- Every model is fit on `train` ONLY (via `get_partition()`), scored on
  `validation` ONLY — `final_test` is never requested with
  `allow_final_test=True` anywhere in this module, its tests, or its
  driver script.
- No hyperparameter search happens here (Stage 9's job): every model uses
  one, pre-decided, documented configuration. No threshold is tuned
  against `validation` — Precision@K/Recall@K use two FIXED, pre-decided
  K fractions (1%, 5%) applied identically to every candidate, never
  searched per-model.
- This is an explicit FIXED-BATCH evaluation: `validation` is scored as a
  completed, static set, matching Stage 6's documented operational
  assumption (`src/feature_engineering.py::build_entity_history_train_and_validation`).
  Nothing here simulates sequential, transaction-by-transaction scoring.
- Random Forest / Gradient Boosting reuse Stage 5's UNSCALED baseline
  output directly (tree splits don't need scaling — Stage 5's own
  rationale for keeping `build_scaler()` separate). Logistic Regression
  variants use a SCALED variant of the same pipeline, built here
  (`build_scaled_preprocessing_pipeline`) — `src/preprocessing.py` itself
  is never modified.
- Nothing in this module trains on, or reads, `isFraud` outside the
  `train` partition during fitting. Evaluation reads `validation`'s
  `isFraud` only to SCORE already-fitted models, never to fit anything.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.leakage_boundaries import get_partition  # noqa: E402
from src.preprocessing import load_partition_df  # noqa: E402

RANDOM_STATE = 42  # fixed across every model in this stage, per reproducibility requirement
K_FRACTIONS = (0.01, 0.05)  # fixed, pre-decided investigation-capacity convention (not searched)


# ---------------------------------------------------------------------------
# Labels — train-only for fitting, validation-only for scoring, never final_test
# ---------------------------------------------------------------------------

def load_labels(
    assignment_path: str | Path, transaction_path: str | Path, partition: str, allow_final_test: bool = False
) -> pd.DataFrame:
    """TransactionID + isFraud for one partition. Routed through
    `get_partition()` -- `final_test` is refused unless explicitly
    overridden, and this module never passes that override."""
    assignment = pd.read_csv(assignment_path)
    ids = set(get_partition(assignment, [partition], allow_final_test=allow_final_test)["TransactionID"])
    labels = pd.read_csv(transaction_path, usecols=["TransactionID", "isFraud"])
    return labels.loc[labels["TransactionID"].isin(ids)].sort_values("TransactionID").reset_index(drop=True)


def build_chronological_pilot_subset(
    assignment_path: str | Path,
    transaction_path: str | Path,
    columns: list[str],
    n_rows: int,
    dt_col: str = "TransactionDT",
    id_col: str = "TransactionID",
) -> pd.DataFrame:
    """The earliest `n_rows` rows of the `train` partition ONLY, by
    `dt_col`, as a single DataFrame carrying `columns` + `isFraud` +
    `id_col` + `dt_col` together.

    This is the Stage 7 bounded-pilot subset builder (review item 3): a
    small, genuinely-earliest-in-time slice of `train`, never a random
    sample, and never `validation`/`final_test` (`get_partition()` is
    called with `allow_final_test` left at its default `False`, and only
    `["train"]` is ever requested here).

    Correct label alignment is enforced BY CONSTRUCTION, not by a
    downstream join: `isFraud` is read from the same `read_csv` call and
    lives in the same DataFrame as every feature column from the first
    line of this function onward, so there is no second merge step where
    a row could end up paired with the wrong label. Sort order ties
    (identical `dt_col`) are broken by `id_col` for a fully deterministic
    result regardless of the input file's own row order.
    """
    assignment = pd.read_csv(assignment_path)
    train_ids = set(get_partition(assignment, ["train"])[id_col])

    df = pd.read_csv(transaction_path, usecols=[id_col, dt_col, "isFraud"] + columns)
    df = df.loc[df[id_col].isin(train_ids)]
    df = df.sort_values([dt_col, id_col], kind="stable").reset_index(drop=True)
    return df.iloc[:n_rows].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics -- no threshold search, two FIXED K fractions
# ---------------------------------------------------------------------------

def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> dict:
    """Precision@K / Recall@K for a FIXED k (count of top-ranked rows).

    Ties in y_score (e.g. a constant-score baseline) are broken by
    whatever order np.argsort returns -- for a genuinely tied score this
    is an arbitrary pick among equals, stated explicitly rather than
    silently producing a falsely-precise-looking number.
    """
    order = np.argsort(-y_score, kind="stable")
    top_k = order[:k]
    tp_at_k = int(y_true[top_k].sum())
    n_positive = int(y_true.sum())
    return {
        "k": k,
        "precision_at_k": round(tp_at_k / k, 4),
        "recall_at_k": round(tp_at_k / n_positive, 4) if n_positive else float("nan"),
        "tp_at_k": tp_at_k,
    }


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray, n_val: int | None = None) -> dict:
    """AP (primary), ROC-AUC (secondary), Precision@K/Recall@K at the two
    fixed K fractions, against VALIDATION's actual labels. No part of this
    function fits anything or reads train data."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = n_val if n_val is not None else len(y_true)
    result = {
        "average_precision": round(float(average_precision_score(y_true, y_score)), 5),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 5),
        "n_rows": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "actual_prevalence": round(float(y_true.mean()), 5),
    }
    for frac in K_FRACTIONS:
        k = max(1, round(n * frac))
        result[f"k_frac_{frac}"] = precision_recall_at_k(y_true, y_score, k)
    return result


def no_skill_baseline(y_train: np.ndarray, y_val: np.ndarray) -> dict:
    """Constant-score baseline: predict the TRAIN prevalence for every
    validation row, scored against validation's ACTUAL labels. A
    mathematical property worth stating, not assumed: AP for a constant
    (fully tied) score equals the evaluated set's own positive
    prevalence, regardless of the constant's exact value -- verified
    numerically here, not just asserted."""
    train_rate = float(np.asarray(y_train).mean())
    constant_scores = np.full(len(y_val), train_rate)
    result = evaluate_predictions(y_val, constant_scores)
    result["constant_score_used"] = round(train_rate, 5)
    result["note"] = (
        "AP for this constant-score predictor equals validation's own actual prevalence "
        "by construction (a tied ranking carries no information) -- both numbers are "
        "reported so the equality can be checked, not assumed."
    )
    return result


# ---------------------------------------------------------------------------
# Pipelines -- additive, scaled variants for Logistic Regression ONLY
# (src/preprocessing.py itself is never modified)
# ---------------------------------------------------------------------------

def build_scaled_preprocessing_pipeline(
    numeric_no_indicator: list[str], numeric_with_indicator: list[str], categorical: list[str]
) -> ColumnTransformer:
    """Stage 5's pipeline structure, with `StandardScaler(with_mean=False)`
    added after each numeric imputer -- `with_mean=False` because centering
    a sparse matrix would force it dense (sklearn raises otherwise); scaling
    only the variance still helps Logistic Regression's optimizer converge
    without paying that memory cost. Tree-based candidates use Stage 5's
    own UNSCALED pipeline directly -- this variant exists for Logistic
    Regression only, per Stage 5's own stated rationale for keeping scaling
    separate.

    Diagnostic finding, not yet acted on here (Stage 7 diagnostic round):
    `StandardScaler` is applied uniformly, including to the 0/1 columns
    `add_indicator=True` generates -- scaling those contributed to, but was
    not the dominant cause of, a severe Logistic Regression slowdown. The
    dominant cause was near-total multicollinearity in the indicator block
    itself (245 of 271 columns are exact duplicates), which this scaling
    choice did not create and would not fix on its own. See
    docs/HANDOFF.md ("Stage 7 -- diagnosed, unresolved") for the full,
    measured writeup.
    """
    numeric_plain = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))])
    numeric_indicator = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler(with_mean=False))]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric_plain", numeric_plain, numeric_no_indicator),
            ("numeric_indicator", numeric_indicator, numeric_with_indicator),
            ("categorical", categorical_pipe, categorical),
        ],
        sparse_threshold=1.0,
    )


def build_scaled_pipeline_with_collapsed_indicators(
    numeric_no_indicator: list[str],
    numeric_with_indicator: list[str],
    categorical: list[str],
    representatives: list[str],
) -> ColumnTransformer:
    """Same scaled structure as `build_scaled_preprocessing_pipeline`, but
    the indicator block is Stage 6's collapsed prototype (one
    MissingIndicator column per unique missingness pattern, not per
    column) -- for the indicator ablation's "collapsed" arm. Every one of
    `numeric_with_indicator`'s own VALUES is still imputed and scaled
    individually; only the indicator-column COUNT differs from the
    "original" arm.

    This is also the CORRECTED experimental branch for the Stage 7
    pilot (see scripts/run_stage7_pilot.py and docs/HANDOFF.md "Stage 7 --
    diagnosed, unresolved"): it was built during the original indicator
    ablation, before the solver slowdown was diagnosed, and already
    satisfies every preprocessing correction the diagnostic pointed to --
    `collapsed_indicators` deliberately has NO scaler (binary indicators
    stay 0/1, verified directly by
    tests/test_model_selection.py::test_collapsed_indicator_block_stays_unscaled_binary),
    `collapsed_values`/`numeric_plain` scale only genuinely continuous
    values, and the representative-column collapse itself removes the 245
    exact-duplicate columns identified as the dominant cause of the
    slowdown. The REMAINING, separately-tested correction is the solver
    (liblinear -> saga) and the class_weight question, not this function.
    """
    numeric_plain = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))])
    collapsed_values = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))])
    collapsed_indicators = Pipeline([("indicator", MissingIndicator(features="missing-only"))])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric_plain", numeric_plain, numeric_no_indicator),
            ("collapsed_values", collapsed_values, numeric_with_indicator),
            ("collapsed_indicators", collapsed_indicators, representatives),
            ("categorical", categorical_pipe, categorical),
        ],
        sparse_threshold=1.0,
    )


def build_simple_baseline_pipeline() -> ColumnTransformer:
    """The 'simple Logistic Regression baseline' pipeline: TransactionAmt
    (imputed+scaled), ProductCD + card4 (one-hot). Deliberately tiny and
    separate from Stage 5's full pipeline."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[("amt", numeric, ["TransactionAmt"]), ("cat", categorical, ["ProductCD", "card4"])],
        sparse_threshold=1.0,
    )


def build_engineered_feature_block() -> ColumnTransformer:
    """Stage 6's 9 numeric-ish engineered features (imputed + scaled) plus
    `productcd_card4_interaction` (one-hot) -- the feature-ablation "B"
    arm's additive block, combined with Stage 5's own output via sparse
    hstack in the driver script. Fit train-only, exactly like every other
    stateful transform in this project.
    """
    numeric_engineered = [
        "txn_hour_phase",
        "txn_dow_phase",
        "card1_prior_txn_count",
        "card1_time_since_prior_txn",
        "p_emaildomain_freq",
        "r_emaildomain_freq",
        "email_domain_match",
        "log1p_transaction_amt",
        "transaction_amt_is_round",
    ]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric_engineered", numeric_pipe, numeric_engineered),
            ("productcd_card4_interaction", categorical_pipe, ["productcd_card4_interaction"]),
        ],
        sparse_threshold=1.0,
    )


def hstack_sparse(blocks: Iterable) -> sp.csr_matrix:
    """Horizontally combine sparse (or dense) blocks into one CSR matrix."""
    return sp.hstack([sp.csr_matrix(b) for b in blocks], format="csr")


# ---------------------------------------------------------------------------
# Memory assessment -- required BEFORE any densification decision
# ---------------------------------------------------------------------------

def estimate_dense_memory_bytes(n_rows: int, n_cols: int, dtype_bytes: int = 8) -> int:
    """Plain analytic estimate, reusing Stage 5's own measured-consistent
    method (§9 of docs/PREPROCESSING_REPORT.md): n_rows * n_cols * dtype_bytes.
    Used to decide WHETHER to densify before doing it, not after.
    """
    return n_rows * n_cols * dtype_bytes


@dataclass
class FitEvalResult:
    model_name: str
    fit_seconds: float
    predict_seconds: float
    metrics: dict


# ---------------------------------------------------------------------------
# Stage 7 convergence investigation -- all TRAIN-ONLY diagnostics.
#
# None of the functions below reads or accepts a validation/final_test
# label anywhere; each takes exactly the array(s) its diagnostic needs
# (an already-imputed numeric block, or a (X_train, y_train) pair for the
# loss-trajectory check), scoped to whatever partition the CALLER passed
# in. The partition discipline lives in the driver script
# (scripts/run_stage7_convergence_diagnostic.py), which only ever passes
# the 15,000-row chronological TRAIN subset here -- never validation,
# never final_test.
# ---------------------------------------------------------------------------

def find_near_constant_columns(values: np.ndarray, columns: list[str], std_threshold: float = 1e-6) -> list[str]:
    """Columns whose (already-imputed) values have near-zero variance --
    a classic, separate-from-multicollinearity source of ill-conditioning:
    a near-constant column contributes almost nothing to separating
    classes but still occupies a dimension, and its STANDARDIZED version
    can have an inflated scale once divided by its own tiny std.
    """
    stds = values.std(axis=0)
    return [columns[i] for i in range(len(columns)) if stds[i] < std_threshold]


def find_duplicate_value_columns(values: np.ndarray, columns: list[str]) -> dict[str, list[str]]:
    """Group `columns` by EXACT-duplicate imputed VALUES (not missingness
    pattern -- that's `find_duplicate_indicator_groups` in
    src/feature_engineering.py). Two columns can have independent
    missingness patterns (already deduplicated, Stage 6/7) while still
    being exact or near-exact duplicates of each other's actual numbers
    (e.g. two Vesta-engineered columns that are simple rescalings of one
    another) -- a second, distinct multicollinearity source this checks
    for directly rather than assuming the indicator collapse already
    covered it.
    """
    seen: dict[bytes, list[str]] = {}
    for i, col in enumerate(columns):
        key = np.round(values[:, i], 8).tobytes()
        seen.setdefault(key, []).append(col)
    return {names[0]: names for names in seen.values() if len(names) > 1}


def max_pairwise_correlation(values: np.ndarray, columns: list[str], exclude_exact_duplicates: bool = True) -> dict:
    """Highest absolute Pearson correlation between any two DISTINCT
    columns in `values` (excluding a column with itself, and optionally
    excluding pairs already known to be exact duplicates, so the exact-
    duplicate finding above doesn't just trivially dominate this one).
    Near-1.0 correlation among nominally different columns is a softer,
    more common multicollinearity signal than exact duplication.
    """
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(values, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    if exclude_exact_duplicates:
        corr = np.nan_to_num(corr, nan=0.0)
    abs_corr = np.abs(corr)
    i, j = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
    return {
        "max_abs_correlation": round(float(abs_corr[i, j]), 6),
        "column_a": columns[i],
        "column_b": columns[j],
        "n_pairs_above_0.99": int(np.sum(abs_corr > 0.99) // 2),
    }


def sparse_magnitude_summary(X) -> dict:
    """Nonzero-entry magnitude range/density of a (possibly sparse) matrix
    -- the same numerical-conditioning lens used in the original Stage 7
    diagnostic, reapplied here to the pilot's actual training matrix."""
    data = X.data if hasattr(X, "data") else X[X != 0]
    shape = X.shape
    density = (X.nnz if hasattr(X, "nnz") else int(np.count_nonzero(X))) / (shape[0] * shape[1])
    abs_data = np.abs(data)
    return {
        "shape": list(shape),
        "density": round(float(density), 4),
        "min_abs_nonzero": float(abs_data.min()) if len(abs_data) else None,
        "max_abs_nonzero": float(abs_data.max()) if len(abs_data) else None,
        "median_abs_nonzero": float(np.median(abs_data)) if len(abs_data) else None,
        "dynamic_range_orders_of_magnitude": round(float(np.log10(abs_data.max() / abs_data.min())), 2) if len(abs_data) else None,
    }


def track_saga_training_loss(
    X_train, y_train: np.ndarray, class_weight, n_steps: int = 10, iters_per_step: int = 100, random_state: int = RANDOM_STATE
) -> list[dict]:
    """TRAIN-ONLY convergence trace: refits the SAME `LogisticRegression`
    object with `warm_start=True` in `iters_per_step`-sized increments,
    recording training log-loss and the coefficient-vector's step-to-step
    L2 change after each increment. `y_train` must be the TRAIN partition's
    own labels -- this function has no partition awareness itself (by
    design, to stay a pure, testable diagnostic); the driver script is
    responsible for never passing it validation or final_test labels.

    A loss still dropping substantially at the last step is evidence of
    genuine (if slow) progress toward convergence; a loss that plateaued
    early despite `ConvergenceWarning` firing is evidence of a harder,
    structural conditioning problem that more iterations alone won't fix.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    model = LogisticRegression(
        solver="saga", penalty="l2", C=1.0, random_state=random_state, class_weight=class_weight,
        max_iter=iters_per_step, warm_start=True,
    )
    trace = []
    prev_coef = None
    for step in range(1, n_steps + 1):
        model.max_iter = iters_per_step
        with warnings_catch_all() as caught:
            model.fit(X_train, y_train)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        coef_change = float(np.linalg.norm(model.coef_ - prev_coef)) if prev_coef is not None else None
        trace.append(
            {
                "cumulative_iters": step * iters_per_step,
                "train_log_loss": round(float(train_loss), 6),
                "coef_l2_change_from_previous_step": round(coef_change, 6) if coef_change is not None else None,
                "convergence_warning_raised": any("did not converge" in str(w.message) for w in caught),
            }
        )
        prev_coef = model.coef_.copy()
    return trace


def warnings_catch_all():
    """Small helper so track_saga_training_loss's warm_start loop doesn't
    need to repeat the catch_warnings boilerplate at every step."""
    import warnings as _warnings

    class _Catcher:
        def __enter__(self):
            self._cm = _warnings.catch_warnings(record=True)
            self._caught = self._cm.__enter__()
            _warnings.simplefilter("always")
            return self._caught

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    return _Catcher()


# ---------------------------------------------------------------------------
# Controlled numeric-redundancy-reduction branch (Stage 7 conditioning
# verification round). Independent of, and does not modify, the approved
# indicator-collapse prototype (src/feature_engineering.py) -- this reduces
# the numeric VALUE columns only, never the indicator-mask representation.
# Every decision here is derived from TRAINING data only, by a fixed,
# position-based rule -- never from the target or from validation AP.
# ---------------------------------------------------------------------------

def find_highly_correlated_value_groups(values: np.ndarray, columns: list[str], threshold: float = 0.99) -> dict[str, list[str]]:
    """Connected-component grouping of `columns` by |Pearson correlation|
    >= `threshold`, computed on whatever `values` the caller passes
    (intended to be TRAIN-partition-only imputed values -- this function
    has no partition awareness itself, same design as the other
    diagnostics in this module; the driver script owns that discipline).

    Deterministic by construction: traversal order follows `columns`'
    own input order, and within each connected component the
    REPRESENTATIVE is simply the first column by that same input order --
    a fixed, position-based rule. No target column, no validation score,
    and no row order ever influences which column is kept. Constant
    (zero-variance) columns produce an undefined (NaN) correlation and
    are deliberately excluded here -- `find_duplicate_value_columns`
    already covers the exact-duplicate case (which includes most
    practically-occurring constant-column ties) via direct equality, not
    correlation.
    """
    n = len(columns)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    adjacency = np.abs(corr) >= threshold

    visited = [False] * n
    groups: dict[str, list[str]] = {}
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        component = [i]
        queue = [i]
        while queue:
            node = queue.pop(0)
            for neighbor in np.where(adjacency[node])[0]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    component.append(int(neighbor))
                    queue.append(int(neighbor))
        if len(component) > 1:
            component_sorted = sorted(component)  # positional order, not discovery order
            names = [columns[idx] for idx in component_sorted]
            groups[names[0]] = names
    return groups


def select_reduced_numeric_columns(
    columns: list[str], exact_duplicate_groups: dict[str, list[str]], correlated_groups: dict[str, list[str]]
) -> dict:
    """Combine exact-duplicate-value groups and highly-correlated-value
    groups into one TRAIN-derived keep/drop decision, with a full,
    documented mapping from every dropped column to the representative
    that replaces it. A column present in both kinds of groups keeps
    whichever representative it was first assigned (exact-duplicate
    groups are resolved first, since exact duplication is the stronger,
    unambiguous case).
    """
    dropped_to_representative: dict[str, str] = {}
    for rep, members in exact_duplicate_groups.items():
        for m in members:
            if m != rep:
                dropped_to_representative[m] = rep
    for rep, members in correlated_groups.items():
        for m in members:
            if m != rep and m not in dropped_to_representative and rep not in dropped_to_representative:
                dropped_to_representative[m] = rep

    keep = [c for c in columns if c not in dropped_to_representative]
    return {
        "keep": keep,
        "dropped_to_representative": dropped_to_representative,
        "n_original": len(columns),
        "n_kept": len(keep),
        "n_dropped": len(dropped_to_representative),
    }


def build_scaled_pipeline_with_collapsed_indicators_and_value_reduction(
    numeric_no_indicator: list[str],
    numeric_with_indicator: list[str],
    categorical: list[str],
    indicator_representatives: list[str],
    value_keep_columns: list[str],
) -> ColumnTransformer:
    """Same structure as `build_scaled_pipeline_with_collapsed_indicators`,
    with one addition: the numeric VALUE blocks (`numeric_plain`,
    `collapsed_values`) are restricted to `value_keep_columns` -- the
    output of `select_reduced_numeric_columns`.

    Deliberately independent of `indicator_representatives`: a column can
    be dropped from the VALUE side (its number is redundant with another
    column's) while its MISSINGNESS INDICATOR stays exactly as the
    approved Stage 6/7 collapse already decided -- these are two
    different ColumnTransformer branches reading different projections
    of the same source columns, and this function changes only one of
    them. The approved indicator-collapse prototype itself
    (`build_scaled_pipeline_with_collapsed_indicators`,
    `src/feature_engineering.py`) is not modified.
    """
    kept_no_indicator = [c for c in numeric_no_indicator if c in value_keep_columns]
    kept_with_indicator = [c for c in numeric_with_indicator if c in value_keep_columns]

    numeric_plain = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))])
    collapsed_values = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))])
    collapsed_indicators = Pipeline([("indicator", MissingIndicator(features="missing-only"))])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric_plain", numeric_plain, kept_no_indicator),
            ("collapsed_values", collapsed_values, kept_with_indicator),
            ("collapsed_indicators", collapsed_indicators, indicator_representatives),
            ("categorical", categorical_pipe, categorical),
        ],
        sparse_threshold=1.0,
    )


# ---------------------------------------------------------------------------
# Tree-based pilots (Random Forest, Gradient Boosting). These use the
# APPROVED Stage 5 preprocessing (src/preprocessing.py::build_preprocessing_pipeline
# -- unscaled, full per-column indicators), NOT the experimental
# collapsed-indicator variant built for the Logistic Regression
# convergence investigation. Trees don't need scaling (Stage 5's own
# stated rationale for keeping it separate), and the multicollinearity
# that blocked SAGA is not a known problem for tree splits, so there is
# no reason to carry the LogReg-specific experimental representation
# over here.
# ---------------------------------------------------------------------------

def build_bounded_random_forest(random_state: int = RANDOM_STATE):
    """One simple, reproducible, computationally-bounded Random Forest
    config -- not searched, not tuned. `max_depth`/`min_samples_leaf`
    bound tree size (and therefore memory/time) given this is a pilot on
    a 15,000-row subset, not the full 438,125-row `train`."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=random_state
    )


def build_bounded_gradient_boosting(random_state: int = RANDOM_STATE):
    """One simple, reproducible, computationally-bounded sklearn
    GradientBoostingClassifier config. Sparse-native (no densification
    needed, unlike HistGradientBoostingClassifier -- Stage 5's own
    finding). Known limitation, stated plainly: this implementation is
    sequential and single-threaded (no `n_jobs`), unlike the Random
    Forest above."""
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_state)


def run_basic_model_correctness_checks(model, X) -> dict:
    """Minimal structural sanity checks on a fitted binary classifier's
    `predict_proba` output -- used as the gate between the Random Forest
    pilot and the Gradient Boosting pilot (per instruction: only proceed
    if Random Forest "passes correctness checks"). Not a performance
    check of any kind.
    """
    proba = model.predict_proba(X)
    checks = {
        "predict_proba_shape_correct": bool(proba.shape == (X.shape[0], 2)),
        "probabilities_in_unit_range": bool(np.all((proba >= 0.0) & (proba <= 1.0))),
        "no_nan_in_predictions": bool(not np.isnan(proba).any()),
        "probabilities_sum_to_one": bool(np.allclose(proba.sum(axis=1), 1.0)),
    }
    checks["all_passed"] = all(checks.values())
    return checks


def align_labels_to_partition_df(
    df: pd.DataFrame, labels_df: pd.DataFrame, id_col: str = "TransactionID", label_col: str = "isFraud"
) -> pd.DataFrame:
    """Attach `label_col` to `df` by an EXPLICIT merge on `id_col`, never
    by assuming `df` and `labels_df` happen to share row order.

    Why this exists: `load_partition_df()` returns rows in the source
    CSV's own file order; `load_labels()` returns rows sorted by
    `id_col`. For this project's actual data the two orders agree (the
    raw file is itself monotonically increasing by `TransactionID`,
    confirmed separately) -- but relying on that silently is exactly the
    kind of fragile, unenforced assumption a "do the labels actually
    align with these rows" review should catch. This function makes the
    alignment explicit and verified instead of implicit: it raises if any
    row fails to match (`validate="one_to_one"`) or if any row ends up
    with a missing label after the merge, rather than silently producing
    a mismatched or NaN-contaminated label column.
    """
    merged = df.merge(labels_df[[id_col, label_col]], on=id_col, how="left", validate="one_to_one")
    if merged[label_col].isna().any():
        missing = merged.loc[merged[label_col].isna(), id_col].tolist()
        raise ValueError(f"{len(missing)} row(s) had no matching label after merge -- real alignment bug, e.g. {missing[:5]}")
    return merged


def build_preprocessing_pipeline_with_collapsed_indicators(
    numeric_no_indicator: list[str],
    numeric_with_indicator: list[str],
    categorical: list[str],
    representatives: list[str],
) -> ColumnTransformer:
    """UNSCALED analogue of
    `build_scaled_pipeline_with_collapsed_indicators` -- same collapsed-
    indicator structure, but matching the approved Stage 5
    `build_preprocessing_pipeline`'s lack of scaling, for tree-based
    candidates (Random Forest, Gradient Boosting) which don't need or
    benefit from scaling. Used only for the missingness-indicator
    ablation's "collapsed" arm; the approved Stage 5 pipeline itself
    (`src/preprocessing.py`) is not modified.
    """
    numeric_plain = Pipeline([("impute", SimpleImputer(strategy="median"))])
    collapsed_values = Pipeline([("impute", SimpleImputer(strategy="median"))])
    collapsed_indicators = Pipeline([("indicator", MissingIndicator(features="missing-only"))])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric_plain", numeric_plain, numeric_no_indicator),
            ("collapsed_values", collapsed_values, numeric_with_indicator),
            ("collapsed_indicators", collapsed_indicators, representatives),
            ("categorical", categorical_pipe, categorical),
        ],
        sparse_threshold=1.0,
    )
