"""Tests for src/model_selection.py.

Synthetic fixtures only (fast, no real-data I/O) -- the real-data run is
exercised in scripts/run_stage7_model_selection.py, same pattern as every
prior stage's test suite.

Run: .venv/bin/pytest tests/test_model_selection.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_selection import (  # noqa: E402
    RANDOM_STATE,
    build_engineered_feature_block,
    build_scaled_pipeline_with_collapsed_indicators,
    build_scaled_preprocessing_pipeline,
    build_simple_baseline_pipeline,
    estimate_dense_memory_bytes,
    evaluate_predictions,
    hstack_sparse,
    load_labels,
    no_skill_baseline,
    precision_recall_at_k,
)
from src.leakage_boundaries import FinalTestAccessError  # noqa: E402


# --- precision_recall_at_k: hand-computed correctness -----------------------

def test_precision_recall_at_k_hand_computed_example() -> None:
    # Scores rank rows 4,3,2,1,0 highest to lowest (reverse index order).
    y_true = np.array([0, 1, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # row 4 highest, row 0 lowest
    result = precision_recall_at_k(y_true, y_score, k=2)
    # top-2 by score: rows 4 (y=1), 3 (y=1) -> 2 true positives out of k=2
    assert result["tp_at_k"] == 2
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == pytest.approx(2 / 3, abs=1e-4)  # 3 total positives; result is rounded to 4dp


def test_precision_recall_at_k_handles_imperfect_ranking() -> None:
    y_true = np.array([1, 0, 0, 0, 0])  # the only positive is ranked LAST
    y_score = np.array([0.1, 0.9, 0.8, 0.7, 0.6])
    result = precision_recall_at_k(y_true, y_score, k=2)
    assert result["tp_at_k"] == 0
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0


# --- no_skill_baseline: AP of a constant score equals actual prevalence -----

def test_no_skill_baseline_ap_equals_validation_prevalence() -> None:
    rng = np.random.default_rng(0)
    y_val = rng.choice([0, 1], size=5000, p=[0.95, 0.05])
    y_train = rng.choice([0, 1], size=5000, p=[0.96, 0.04])  # deliberately different rate

    result = no_skill_baseline(y_train, y_val)
    actual_val_prevalence = y_val.mean()

    assert result["average_precision"] == pytest.approx(actual_val_prevalence, abs=1e-9)
    assert result["actual_prevalence"] == pytest.approx(actual_val_prevalence, abs=1e-4)
    # the constant predicted score itself came from TRAIN, not validation
    assert result["constant_score_used"] == pytest.approx(y_train.mean(), abs=1e-4)


def test_no_skill_baseline_ap_is_invariant_to_which_constant_value_is_used() -> None:
    """The mathematical property stated in the docstring, checked directly:
    ANY constant score gives the same AP (= actual prevalence), not just
    the train-rate constant this project happens to use."""
    rng = np.random.default_rng(1)
    y_val = rng.choice([0, 1], size=2000, p=[0.9, 0.1])

    result_a = evaluate_predictions(y_val, np.full(len(y_val), 0.1))
    result_b = evaluate_predictions(y_val, np.full(len(y_val), 0.9))
    assert result_a["average_precision"] == pytest.approx(result_b["average_precision"])
    assert result_a["average_precision"] == pytest.approx(y_val.mean(), abs=1e-9)


# --- evaluate_predictions: structure and ranges -----------------------------

def test_evaluate_predictions_structure_and_ranges() -> None:
    rng = np.random.default_rng(2)
    y_true = rng.choice([0, 1], size=1000, p=[0.9, 0.1])
    y_score = rng.random(1000)
    result = evaluate_predictions(y_true, y_score)

    assert 0.0 <= result["average_precision"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert result["n_rows"] == 1000
    assert result["n_positive"] == int(y_true.sum())
    assert "k_frac_0.01" in result and "k_frac_0.05" in result
    for frac_key in ("k_frac_0.01", "k_frac_0.05"):
        assert 0.0 <= result[frac_key]["precision_at_k"] <= 1.0
        assert 0.0 <= result[frac_key]["recall_at_k"] <= 1.0


def test_evaluate_predictions_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    y_true = rng.choice([0, 1], size=500, p=[0.9, 0.1])
    y_score = rng.random(500)
    assert evaluate_predictions(y_true, y_score) == evaluate_predictions(y_true, y_score)


# --- load_labels: partition isolation, same discipline as Stages 5/6 -------

def _write_label_fixture(tmp_path):
    assignment = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [10, 20, 30, 40, 50, 60],
            "isFraud": [0, 1, 0, 0, 1, 1],
            "partition": ["train", "train", "train", "validation", "validation", "final_test"],
        }
    )
    transaction = pd.DataFrame(
        {"TransactionID": [1, 2, 3, 4, 5, 6], "isFraud": [0, 1, 0, 0, 1, 1]}
    )
    assignment_path = tmp_path / "assignment.csv"
    transaction_path = tmp_path / "transaction.csv"
    assignment.to_csv(assignment_path, index=False)
    transaction.to_csv(transaction_path, index=False)
    return assignment_path, transaction_path


def test_load_labels_returns_only_the_requested_partition(tmp_path) -> None:
    assignment_path, transaction_path = _write_label_fixture(tmp_path)
    train_labels = load_labels(assignment_path, transaction_path, "train")
    val_labels = load_labels(assignment_path, transaction_path, "validation")
    assert sorted(train_labels["TransactionID"]) == [1, 2, 3]
    assert sorted(val_labels["TransactionID"]) == [4, 5]


def test_load_labels_refuses_final_test_by_default(tmp_path) -> None:
    assignment_path, transaction_path = _write_label_fixture(tmp_path)
    with pytest.raises(FinalTestAccessError):
        load_labels(assignment_path, transaction_path, "final_test")


# --- Pipeline builders: structural smoke tests ------------------------------

def test_scaled_preprocessing_pipeline_produces_no_nan() -> None:
    df = pd.DataFrame(
        {
            "num_plain": [1.0, 2.0, np.nan, 4.0],
            "num_ind": [np.nan, 2.0, 3.0, np.nan],
            "cat": ["a", "b", None, "a"],
        }
    )
    ct = build_scaled_preprocessing_pipeline(["num_plain"], ["num_ind"], ["cat"])
    output = ct.fit_transform(df)
    dense = output.toarray() if hasattr(output, "toarray") else output
    assert not np.isnan(dense).any()


def test_scaled_pipeline_with_collapsed_indicators_reduces_indicator_width() -> None:
    df = pd.DataFrame(
        {
            "num_plain": [1.0, 2.0, 3.0, 4.0],
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [10.0, np.nan, 30.0, np.nan],  # same missingness pattern as "a"
            "cat": ["x", "y", "x", "y"],
        }
    )
    ct = build_scaled_pipeline_with_collapsed_indicators(["num_plain"], ["a", "b"], ["cat"], ["a"])
    output = ct.fit_transform(df)
    dense = output.toarray() if hasattr(output, "toarray") else output
    assert not np.isnan(dense).any()
    # num_plain(1) + collapsed_values(2: a,b) + collapsed_indicators(1: only "a" is a representative) + cat one-hot(2)
    assert dense.shape[1] == 1 + 2 + 1 + 2


def test_simple_baseline_pipeline_structure() -> None:
    df = pd.DataFrame({"TransactionAmt": [10.0, np.nan, 30.0], "ProductCD": ["W", "C", "W"], "card4": ["visa", "visa", None]})
    ct = build_simple_baseline_pipeline()
    output = ct.fit_transform(df)
    dense = output.toarray() if hasattr(output, "toarray") else output
    assert not np.isnan(dense).any()
    assert dense.shape[0] == 3


def test_engineered_feature_block_structure() -> None:
    df = pd.DataFrame(
        {
            "txn_hour_phase": [1, 2, 3],
            "txn_dow_phase": [0, 1, 2],
            "card1_prior_txn_count": [0, 5, 10],
            "card1_time_since_prior_txn": [np.nan, 100.0, 200.0],
            "p_emaildomain_freq": [10, 20, 0],
            "r_emaildomain_freq": [5, 0, 15],
            "email_domain_match": [np.nan, 1.0, 0.0],
            "log1p_transaction_amt": [2.3, 3.4, 4.5],
            "transaction_amt_is_round": [1, 0, 1],
            "productcd_card4_interaction": ["W_visa", "C_visa", "W_missing"],
        }
    )
    ct = build_engineered_feature_block()
    output = ct.fit_transform(df)
    dense = output.toarray() if hasattr(output, "toarray") else output
    assert not np.isnan(dense).any()
    assert dense.shape[0] == 3


# --- hstack_sparse and estimate_dense_memory_bytes --------------------------

def test_hstack_sparse_combines_blocks_correctly() -> None:
    a = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = np.array([[5.0], [6.0]])
    combined = hstack_sparse([a, b])
    assert combined.shape == (2, 3)
    np.testing.assert_array_equal(combined.toarray(), [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])


def test_estimate_dense_memory_bytes_is_plain_multiplication() -> None:
    assert estimate_dense_memory_bytes(n_rows=1000, n_cols=10, dtype_bytes=8) == 80_000


# --- Reproducibility: fixed random_state gives identical model results -----

def test_fixed_random_state_gives_identical_model_fits() -> None:
    rng = np.random.default_rng(4)
    X = sp.csr_matrix(rng.random((200, 5)))
    y = rng.choice([0, 1], size=200, p=[0.8, 0.2])

    model_a = LogisticRegression(random_state=RANDOM_STATE, max_iter=200).fit(X, y)
    model_b = LogisticRegression(random_state=RANDOM_STATE, max_iter=200).fit(X, y)
    np.testing.assert_array_equal(model_a.coef_, model_b.coef_)
