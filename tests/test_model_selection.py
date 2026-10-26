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
    align_labels_to_partition_df,
    build_bounded_gradient_boosting,
    build_bounded_random_forest,
    build_chronological_pilot_subset,
    build_preprocessing_pipeline_with_collapsed_indicators,
    build_engineered_feature_block,
    build_scaled_pipeline_with_collapsed_indicators,
    build_scaled_pipeline_with_collapsed_indicators_and_value_reduction,
    build_scaled_preprocessing_pipeline,
    build_simple_baseline_pipeline,
    estimate_dense_memory_bytes,
    evaluate_predictions,
    find_duplicate_value_columns,
    find_highly_correlated_value_groups,
    find_near_constant_columns,
    hstack_sparse,
    load_labels,
    max_pairwise_correlation,
    no_skill_baseline,
    precision_recall_at_k,
    run_basic_model_correctness_checks,
    select_reduced_numeric_columns,
    sparse_magnitude_summary,
    track_saga_training_loss,
)
from src.feature_engineering import collapsed_indicator_representatives, find_duplicate_indicator_groups  # noqa: E402
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


# --- Stage 7 bounded-pilot correction: review item 2/3/4 tests --------------

def test_collapsed_indicator_block_stays_unscaled_binary() -> None:
    """Direct proof (not inference from reading code) that the
    'collapsed_indicators' block in build_scaled_pipeline_with_collapsed_indicators
    is genuinely unscaled: every nonzero entry contributed by that block
    must be exactly 1.0, never a scaled fraction."""
    df = pd.DataFrame(
        {
            "num_plain": [1.0, 2.0, 3.0, 4.0, 5.0],
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],  # missing 40% of the time
            "b": [10.0, np.nan, 30.0, np.nan, 50.0],  # same pattern as "a"
            "cat": ["x", "y", "x", "y", "x"],
        }
    )
    ct = build_scaled_pipeline_with_collapsed_indicators(["num_plain"], ["a", "b"], ["cat"], ["a"])
    output = ct.fit_transform(df)
    dense = output.toarray() if hasattr(output, "toarray") else output

    # Layout: num_plain(1, scaled) + collapsed_values(2: a,b, scaled) +
    # collapsed_indicators(1: representative "a") + cat one-hot(2)
    indicator_col = dense[:, 3]
    assert set(np.unique(indicator_col)) <= {0.0, 1.0}
    # rows 1 and 3 (0-indexed) are the missing ones -> indicator must be exactly 1
    assert indicator_col[1] == 1.0
    assert indicator_col[3] == 1.0
    assert indicator_col[0] == 0.0


def test_collapsed_pipeline_values_are_scaled_not_left_raw() -> None:
    """The continuous VALUE columns, unlike the indicators, ARE expected
    to be scaled -- confirms the two blocks are genuinely treated
    differently, not that scaling was accidentally dropped everywhere."""
    df = pd.DataFrame(
        {
            "num_plain": [1.0, 2.0, 3.0, 1000.0],  # large spread -> scaling should compress this
            "a": [1.0, 2.0, 3.0, np.nan],  # at least one missing value, matching real usage: a
            "cat": ["x", "y", "x", "y"],  # representative column always has SOME missingness
        }
    )
    ct = build_scaled_pipeline_with_collapsed_indicators(["num_plain"], ["a"], ["cat"], ["a"])
    output = ct.fit_transform(df)
    dense = output.toarray() if hasattr(output, "toarray") else output
    raw_std = df["num_plain"].std()
    # StandardScaler(with_mean=False) divides by std -- the scaled column's
    # own std should differ from the raw column's std (not pass through raw).
    assert not np.isclose(dense[:, 0].std(), raw_std)


# --- build_chronological_pilot_subset: review item 3 ------------------------

def _write_pilot_fixture(tmp_path, n=20):
    rng = np.random.default_rng(7)
    ids = list(range(1, n + 1))
    dts = rng.permutation(np.arange(100, 100 + n * 10, 10))  # shuffled on purpose
    fraud = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    amt = rng.random(n) * 100
    assignment = pd.DataFrame(
        {"TransactionID": ids, "TransactionDT": dts, "isFraud": fraud, "partition": ["train"] * (n - 3) + ["validation", "validation", "final_test"]}
    )
    transaction = pd.DataFrame({"TransactionID": ids, "TransactionDT": dts, "isFraud": fraud, "TransactionAmt": amt})
    assignment_path = tmp_path / "assignment.csv"
    transaction_path = tmp_path / "transaction.csv"
    assignment.to_csv(assignment_path, index=False)
    transaction.to_csv(transaction_path, index=False)
    return assignment_path, transaction_path, assignment, transaction


def test_chronological_pilot_subset_picks_genuinely_earliest_train_rows(tmp_path) -> None:
    assignment_path, transaction_path, assignment, transaction = _write_pilot_fixture(tmp_path, n=20)
    subset = build_chronological_pilot_subset(assignment_path, transaction_path, ["TransactionAmt"], n_rows=5)

    train_ids = set(assignment.loc[assignment["partition"] == "train", "TransactionID"])
    assert set(subset["TransactionID"]) <= train_ids  # never validation/final_test

    # Independently recompute the 5 earliest TRAIN rows by TransactionDT and compare.
    train_txn = transaction.loc[transaction["TransactionID"].isin(train_ids)].sort_values("TransactionDT")
    expected_earliest_5 = set(train_txn.head(5)["TransactionID"])
    assert set(subset["TransactionID"]) == expected_earliest_5
    assert subset["TransactionDT"].is_monotonic_increasing


def test_chronological_pilot_subset_label_alignment_is_correct(tmp_path) -> None:
    """Review item 4: correct label alignment. Cross-check each selected
    row's isFraud against an independent lookup from the raw transaction
    file -- would fail if any join/sort step silently misaligned labels."""
    assignment_path, transaction_path, assignment, transaction = _write_pilot_fixture(tmp_path, n=20)
    subset = build_chronological_pilot_subset(assignment_path, transaction_path, ["TransactionAmt"], n_rows=8)

    truth = transaction.set_index("TransactionID")["isFraud"]
    for _, row in subset.iterrows():
        assert row["isFraud"] == truth.loc[row["TransactionID"]]


def test_chronological_pilot_subset_never_touches_final_test(tmp_path) -> None:
    assignment_path, transaction_path, _, _ = _write_pilot_fixture(tmp_path, n=20)
    subset = build_chronological_pilot_subset(assignment_path, transaction_path, ["TransactionAmt"], n_rows=20)
    # n_rows=20 exceeds the train-only row count -- must still contain ONLY
    # train rows, never spill into validation/final_test even when asked
    # for "more rows than exist".
    assignment = pd.read_csv(assignment_path)
    final_test_ids = set(assignment.loc[assignment["partition"] == "final_test", "TransactionID"])
    validation_ids = set(assignment.loc[assignment["partition"] == "validation", "TransactionID"])
    assert not (set(subset["TransactionID"]) & final_test_ids)
    assert not (set(subset["TransactionID"]) & validation_ids)


def test_chronological_pilot_subset_is_deterministic_regardless_of_file_row_order(tmp_path) -> None:
    assignment_path, transaction_path, assignment, transaction = _write_pilot_fixture(tmp_path, n=20)
    subset_a = build_chronological_pilot_subset(assignment_path, transaction_path, ["TransactionAmt"], n_rows=6)

    shuffled_transaction_path = tmp_path / "transaction_shuffled.csv"
    transaction.sample(frac=1, random_state=99).to_csv(shuffled_transaction_path, index=False)
    subset_b = build_chronological_pilot_subset(assignment_path, shuffled_transaction_path, ["TransactionAmt"], n_rows=6)

    assert list(subset_a["TransactionID"]) == list(subset_b["TransactionID"])


# --- Feature-name / column-order stability on the collapsed pipeline -------

def test_collapsed_pipeline_feature_names_stable_across_fit_and_transform() -> None:
    df = pd.DataFrame(
        {"num_plain": [1.0, 2.0, 3.0], "a": [1.0, np.nan, 3.0], "cat": ["x", "y", "x"]}
    )
    ct = build_scaled_pipeline_with_collapsed_indicators(["num_plain"], ["a"], ["cat"], ["a"])
    ct.fit(df)
    names_before = list(ct.get_feature_names_out())
    ct.transform(pd.DataFrame({"num_plain": [9.0], "a": [9.0], "cat": ["y"]}))
    names_after = list(ct.get_feature_names_out())
    assert names_before == names_after


# --- Reproducibility of the saga + collapsed-indicator experimental config -

def test_saga_solver_fit_is_reproducible_with_fixed_random_state() -> None:
    """The specific solver/config combination this round's pilot uses
    (saga, fixed random_state) -- not just LogisticRegression in general --
    must be reproducible."""
    rng = np.random.default_rng(5)
    X = sp.csr_matrix(rng.random((300, 8)))
    y = rng.choice([0, 1], size=300, p=[0.85, 0.15])

    model_a = LogisticRegression(solver="saga", penalty="l2", random_state=RANDOM_STATE, max_iter=300).fit(X, y)
    model_b = LogisticRegression(solver="saga", penalty="l2", random_state=RANDOM_STATE, max_iter=300).fit(X, y)
    np.testing.assert_array_equal(model_a.coef_, model_b.coef_)


# --- Stage 7 convergence investigation: new diagnostic functions ----------

def test_find_near_constant_columns_detects_near_zero_variance() -> None:
    values = np.column_stack(
        [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # normal variance
            np.array([1.0, 1.0, 1.0, 1.0, 1.0000001]),  # near-constant
        ]
    )
    result = find_near_constant_columns(values, ["normal", "near_constant"], std_threshold=1e-4)
    assert result == ["near_constant"]


def test_find_duplicate_value_columns_detects_exact_duplicates_not_just_mask() -> None:
    """Distinct from find_duplicate_indicator_groups (feature_engineering.py),
    which only looks at missingness MASKS -- this checks the actual imputed
    VALUES, which can be duplicated even when two columns' missingness
    patterns were already deduplicated separately."""
    values = np.column_stack(
        [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),  # exact duplicate of column 0
            np.array([9.0, 8.0, 7.0, 6.0]),  # distinct
        ]
    )
    groups = find_duplicate_value_columns(values, ["a", "b", "c"])
    assert len(groups) == 1
    (rep, members), = groups.items()
    assert set(members) == {"a", "b"}
    assert "c" not in members


def test_find_duplicate_value_columns_returns_empty_when_none_duplicate() -> None:
    values = np.column_stack([np.array([1.0, 2.0, 3.0]), np.array([9.0, 1.0, 5.0])])
    assert find_duplicate_value_columns(values, ["a", "b"]) == {}


def test_max_pairwise_correlation_finds_the_known_correlated_pair() -> None:
    rng = np.random.default_rng(0)
    base = rng.random(500)
    values = np.column_stack(
        [
            base,
            base * 2.0 + 0.0001 * rng.random(500),  # near-perfectly correlated with column 0
            rng.random(500),  # independent
        ]
    )
    result = max_pairwise_correlation(values, ["a", "b", "c"])
    assert result["max_abs_correlation"] > 0.99
    assert {result["column_a"], result["column_b"]} == {"a", "b"}


def test_sparse_magnitude_summary_matches_hand_computed_values() -> None:
    X = sp.csr_matrix(np.array([[0.0, 2.0], [4.0, 0.0]]))
    result = sparse_magnitude_summary(X)
    assert result["shape"] == [2, 2]
    assert result["density"] == pytest.approx(0.5)
    assert result["min_abs_nonzero"] == pytest.approx(2.0)
    assert result["max_abs_nonzero"] == pytest.approx(4.0)


def test_track_saga_training_loss_reports_decreasing_or_stable_loss() -> None:
    """Training-only diagnostic -- no validation/final_test label is ever
    passed to this function; the driver script is responsible for that
    partition discipline, but this test confirms the function itself
    behaves correctly on a clearly-separable synthetic training set."""
    rng = np.random.default_rng(6)
    n = 400
    X = rng.random((n, 4))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)  # clearly separable, should converge fast

    trace = track_saga_training_loss(sp.csr_matrix(X), y, class_weight=None, n_steps=4, iters_per_step=50)
    assert len(trace) == 4
    assert all(t["cumulative_iters"] == (i + 1) * 50 for i, t in enumerate(trace))
    # loss should not INCREASE step over step on this easy, separable problem
    losses = [t["train_log_loss"] for t in trace]
    assert losses[-1] <= losses[0] + 1e-6


def test_track_saga_training_loss_is_reproducible() -> None:
    rng = np.random.default_rng(8)
    X = sp.csr_matrix(rng.random((200, 3)))
    y = rng.choice([0, 1], size=200, p=[0.8, 0.2])

    trace_a = track_saga_training_loss(X, y, class_weight="balanced", n_steps=3, iters_per_step=50)
    trace_b = track_saga_training_loss(X, y, class_weight="balanced", n_steps=3, iters_per_step=50)
    assert trace_a == trace_b


# --- Controlled numeric-redundancy reduction (conditioning verification) --

def test_find_highly_correlated_value_groups_finds_connected_cluster() -> None:
    rng = np.random.default_rng(1)
    base = rng.random(500)
    values = np.column_stack(
        [
            base,
            base * 3.0 + 1e-6 * rng.random(500),  # near-perfect correlation with col 0
            base * -2.0 + 1e-6 * rng.random(500),  # near-perfect NEGATIVE correlation with col 0 -> same cluster
            rng.random(500),  # independent
        ]
    )
    groups = find_highly_correlated_value_groups(values, ["a", "b", "c", "d"], threshold=0.99)
    assert len(groups) == 1
    (rep, members), = groups.items()
    assert set(members) == {"a", "b", "c"}
    assert "d" not in members
    assert rep == "a"  # deterministic: first column by input order, not by any score


def test_find_highly_correlated_value_groups_is_deterministic_regardless_of_row_order() -> None:
    rng = np.random.default_rng(2)
    base = rng.random(300)
    values = np.column_stack([base, base * 2.0, rng.random(300)])
    columns = ["a", "b", "c"]
    groups_a = find_highly_correlated_value_groups(values, columns, threshold=0.99)

    perm = rng.permutation(300)
    groups_b = find_highly_correlated_value_groups(values[perm], columns, threshold=0.99)
    assert groups_a == groups_b


def test_select_reduced_numeric_columns_documents_full_drop_mapping() -> None:
    columns = ["a", "b", "c", "d", "e"]
    exact = {"a": ["a", "b"]}  # b is an exact duplicate of a
    correlated = {"c": ["c", "d"]}  # d highly correlated with c
    result = select_reduced_numeric_columns(columns, exact, correlated)
    assert result["keep"] == ["a", "c", "e"]
    assert result["dropped_to_representative"] == {"b": "a", "d": "c"}
    assert result["n_original"] == 5
    assert result["n_kept"] == 3
    assert result["n_dropped"] == 2


def test_select_reduced_numeric_columns_never_drops_a_column_not_in_any_group() -> None:
    columns = ["a", "b", "c"]
    result = select_reduced_numeric_columns(columns, {}, {})
    assert result["keep"] == columns
    assert result["dropped_to_representative"] == {}


def test_value_reduction_pipeline_preserves_indicator_collapse_independently() -> None:
    """The indicator representative list must be usable EVEN IF the
    representative's own column was dropped from the value side -- proves
    the two branches (value reduction, indicator collapse) really are
    independent, not silently coupled."""
    df = pd.DataFrame(
        {
            "num_plain": [1.0, 2.0, 3.0, 4.0],
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [10.0, np.nan, 30.0, np.nan],  # same missingness pattern AND highly correlated values to "a"
            "cat": ["x", "y", "x", "y"],
        }
    )
    # "a" is the indicator representative, but suppose the value-reduction step
    # decided "b" (not "a") is the kept value representative -- "a"'s own value
    # is dropped, while its indicator flag (as representative) must still work.
    ct = build_scaled_pipeline_with_collapsed_indicators_and_value_reduction(
        numeric_no_indicator=["num_plain"],
        numeric_with_indicator=["a", "b"],
        categorical=["cat"],
        indicator_representatives=["a"],  # indicator collapse still points at "a"
        value_keep_columns=["num_plain", "b"],  # but "a"'s VALUE is dropped here
    )
    output = ct.fit_transform(df)
    dense = output.toarray() if hasattr(output, "toarray") else output
    assert not np.isnan(dense).any()
    # num_plain(1) + collapsed_values(1: only "b" kept) + collapsed_indicators(1: "a") + cat(2)
    assert dense.shape[1] == 1 + 1 + 1 + 2


def test_value_reduction_feature_names_stable_across_fit_and_transform() -> None:
    df = pd.DataFrame(
        {"num_plain": [1.0, 2.0, 3.0], "a": [1.0, np.nan, 3.0], "b": [10.0, np.nan, 30.0], "cat": ["x", "y", "x"]}
    )
    ct = build_scaled_pipeline_with_collapsed_indicators_and_value_reduction(
        ["num_plain"], ["a", "b"], ["cat"], indicator_representatives=["a"], value_keep_columns=["num_plain", "b"]
    )
    ct.fit(df)
    names_before = list(ct.get_feature_names_out())
    ct.transform(pd.DataFrame({"num_plain": [9.0], "a": [9.0], "b": [9.0], "cat": ["y"]}))
    assert list(ct.get_feature_names_out()) == names_before


def test_value_reduction_label_alignment_uses_same_subset_builder(tmp_path) -> None:
    """The value-reduction branch must be built from the SAME chronological
    pilot subset as the baseline branch -- confirmed by reusing
    build_chronological_pilot_subset and checking row-for-row TransactionID
    equality, not a freshly (and possibly differently) sampled subset."""
    assignment_path, transaction_path, _, _ = _write_pilot_fixture(tmp_path, n=20)
    subset_a = build_chronological_pilot_subset(assignment_path, transaction_path, ["TransactionAmt"], n_rows=6)
    subset_b = build_chronological_pilot_subset(assignment_path, transaction_path, ["TransactionAmt"], n_rows=6)
    assert list(subset_a["TransactionID"]) == list(subset_b["TransactionID"])
    assert list(subset_a["isFraud"]) == list(subset_b["isFraud"])


def test_value_reduction_never_touches_final_test(tmp_path) -> None:
    assignment_path, transaction_path, _, _ = _write_pilot_fixture(tmp_path, n=20)
    subset = build_chronological_pilot_subset(assignment_path, transaction_path, ["TransactionAmt"], n_rows=20)
    assignment = pd.read_csv(assignment_path)
    final_test_ids = set(assignment.loc[assignment["partition"] == "final_test", "TransactionID"])
    assert not (set(subset["TransactionID"]) & final_test_ids)


# --- Tree-based pilot configs (Random Forest, Gradient Boosting) ----------

def test_build_bounded_random_forest_has_expected_bounded_config() -> None:
    model = build_bounded_random_forest()
    assert model.n_estimators == 100
    assert model.max_depth == 10
    assert model.min_samples_leaf == 20
    assert model.random_state == RANDOM_STATE


def test_build_bounded_random_forest_is_reproducible() -> None:
    """`n_jobs=-1` means per-tree probabilities are summed in whatever
    order threads finish, so bit-for-bit equality isn't the right bar --
    sklearn's random_state still makes every TREE identical; only the
    floating-point SUMMATION order across threads differs, at the
    ~1e-16 (machine-epsilon) level. assert_allclose with a tight
    tolerance confirms that distinction rather than demanding exact
    bitwise equality a parallel reduction can't give."""
    rng = np.random.default_rng(10)
    X = sp.csr_matrix(rng.random((300, 6)))
    y = rng.choice([0, 1], size=300, p=[0.85, 0.15])
    model_a = build_bounded_random_forest().fit(X, y)
    model_b = build_bounded_random_forest().fit(X, y)
    np.testing.assert_allclose(model_a.predict_proba(X), model_b.predict_proba(X), rtol=1e-9, atol=1e-12)


def test_build_bounded_gradient_boosting_has_expected_bounded_config() -> None:
    model = build_bounded_gradient_boosting()
    assert model.n_estimators == 100
    assert model.max_depth == 3
    assert model.random_state == RANDOM_STATE


def test_build_bounded_gradient_boosting_is_reproducible() -> None:
    rng = np.random.default_rng(11)
    X = sp.csr_matrix(rng.random((300, 6)))
    y = rng.choice([0, 1], size=300, p=[0.85, 0.15])
    model_a = build_bounded_gradient_boosting().fit(X, y)
    model_b = build_bounded_gradient_boosting().fit(X, y)
    np.testing.assert_array_equal(model_a.predict_proba(X), model_b.predict_proba(X))


def test_run_basic_model_correctness_checks_passes_for_a_well_behaved_model() -> None:
    rng = np.random.default_rng(12)
    X = sp.csr_matrix(rng.random((200, 4)))
    y = rng.choice([0, 1], size=200, p=[0.8, 0.2])
    model = build_bounded_random_forest().fit(X, y)
    checks = run_basic_model_correctness_checks(model, X)
    assert checks["all_passed"] is True
    assert checks["predict_proba_shape_correct"] is True
    assert checks["probabilities_sum_to_one"] is True


def test_run_basic_model_correctness_checks_catches_a_broken_predict_proba() -> None:
    """A deliberately broken stand-in model (wrong-shaped, non-probability
    output) must make the checks fail, not silently pass -- proves the
    checks actually check something, not just return True unconditionally."""

    class BrokenModel:
        def predict_proba(self, X):
            return np.full((X.shape[0], 2), 5.0)  # not a valid probability

    checks = run_basic_model_correctness_checks(BrokenModel(), sp.csr_matrix(np.ones((10, 3))))
    assert checks["all_passed"] is False
    assert checks["probabilities_in_unit_range"] is False
    assert checks["probabilities_sum_to_one"] is False


# --- align_labels_to_partition_df: explicit, verified label alignment -----

def test_align_labels_to_partition_df_attaches_correct_labels_regardless_of_row_order() -> None:
    """The exact failure mode this function exists to prevent: df and
    labels_df in DIFFERENT row orders must still end up correctly paired,
    by TransactionID, not by position."""
    df = pd.DataFrame({"TransactionID": [3, 1, 2], "feature": [30.0, 10.0, 20.0]})  # out-of-order on purpose
    labels_df = pd.DataFrame({"TransactionID": [1, 2, 3], "isFraud": [0, 1, 0]})  # sorted
    result = align_labels_to_partition_df(df, labels_df)
    # row for TransactionID=3 must get label 0, ID=1 -> 0, ID=2 -> 1 -- regardless of input order
    lookup = dict(zip(result["TransactionID"], result["isFraud"]))
    assert lookup == {1: 0, 2: 1, 3: 0}


def test_align_labels_to_partition_df_raises_on_unmatched_row() -> None:
    df = pd.DataFrame({"TransactionID": [1, 2, 999], "feature": [1.0, 2.0, 3.0]})  # 999 has no label
    labels_df = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0, 1]})
    with pytest.raises(ValueError, match="no matching label"):
        align_labels_to_partition_df(df, labels_df)


def test_align_labels_to_partition_df_raises_on_duplicate_id_in_labels() -> None:
    """validate='one_to_one' must catch a labels_df with a duplicated ID
    -- a real data-integrity problem, not something to silently average
    or arbitrarily pick one of."""
    df = pd.DataFrame({"TransactionID": [1, 2], "feature": [1.0, 2.0]})
    labels_df = pd.DataFrame({"TransactionID": [1, 1, 2], "isFraud": [0, 1, 1]})  # ID=1 duplicated with conflicting labels
    with pytest.raises(Exception):  # pandas raises MergeError, a ValueError subclass
        align_labels_to_partition_df(df, labels_df)


# --- Unscaled collapsed-indicator pipeline (for tree-based candidates) ----

def test_unscaled_collapsed_indicator_pipeline_leaves_indicators_as_exact_binary() -> None:
    df = pd.DataFrame(
        {
            "num_plain": [1.0, 2.0, 3.0, 4.0],
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [10.0, np.nan, 30.0, np.nan],
            "cat": ["x", "y", "x", "y"],
        }
    )
    ct = build_preprocessing_pipeline_with_collapsed_indicators(["num_plain"], ["a", "b"], ["cat"], ["a"])
    output = ct.fit_transform(df)
    dense = output.toarray() if hasattr(output, "toarray") else output
    assert not np.isnan(dense).any()
    # num_plain(1, unscaled) + collapsed_values(2: a,b, unscaled) + collapsed_indicators(1: "a") + cat(2)
    assert dense.shape[1] == 1 + 2 + 1 + 2
    # unscaled: num_plain's raw values must appear exactly, not divided by any std
    np.testing.assert_array_equal(dense[:, 0], np.array([1.0, 2.0, 3.0, 4.0]))
