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
