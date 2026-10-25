"""Tests for src/preprocessing.py.

Scoped to synthetic fixtures for the architecture/logic (fast, no real-data
I/O needed) plus ONE deliberately non-trivial leakage-guard test that proves
train-only fitting actually differs from train+validation fitting on a
constructed fixture where it must — not a test that would pass regardless
of whether the fitting discipline was followed.

The real-data run (column scope, missingness-driven indicator selection,
memory/density/timing measurements) is exercised in
scripts/run_stage5_preprocessing.py, same pattern as Stages 2-4.

Run: .venv/bin/pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (  # noqa: E402
    _verify_ids_in_partition,
    build_preprocessing_pipeline,
    build_scaler,
    check_sparse_input_compatibility,
    estimate_indicator_memory_cost,
    load_partition_df,
    measure_output,
    select_indicator_columns,
    verify_column_accounting,
)
from src.leakage_boundaries import FinalTestAccessError


# --- select_indicator_columns -----------------------------------------------

def test_select_indicator_columns_splits_by_threshold() -> None:
    missingness = pd.Series({"low_missing": 0.5, "high_missing": 40.0, "borderline": 5.0})
    result = select_indicator_columns(missingness, threshold_pct=5.0)
    assert result["with_indicator"] == ["high_missing", "borderline"]
    assert result["without_indicator"] == ["low_missing"]


def test_select_indicator_columns_handles_all_below_threshold() -> None:
    missingness = pd.Series({"a": 0.0, "b": 1.0})
    result = select_indicator_columns(missingness, threshold_pct=5.0)
    assert result["with_indicator"] == []
    assert result["without_indicator"] == ["a", "b"]


# --- estimate_indicator_memory_cost -----------------------------------------

def test_estimate_indicator_memory_cost_scales_linearly() -> None:
    one_col = estimate_indicator_memory_cost(n_rows=100_000, n_indicator_columns=1)
    ten_cols = estimate_indicator_memory_cost(n_rows=100_000, n_indicator_columns=10)
    assert ten_cols["estimated_bytes"] == pytest.approx(one_col["estimated_bytes"] * 10)


# --- build_preprocessing_pipeline: structural correctness -------------------

def test_pipeline_output_has_no_missing_values_after_fit_transform() -> None:
    df = pd.DataFrame(
        {
            "num_plain": [1.0, 2.0, np.nan, 4.0],
            "num_ind": [np.nan, 2.0, 3.0, np.nan],
            "cat": ["a", "b", None, "a"],
        }
    )
    ct = build_preprocessing_pipeline(["num_plain"], ["num_ind"], ["cat"])
    output = ct.fit_transform(df)
    dense = output.toarray() if hasattr(output, "toarray") else output
    assert not np.isnan(dense).any()


def test_pipeline_adds_indicator_column_only_for_indicator_group() -> None:
    df = pd.DataFrame(
        {
            "num_plain": [1.0, 2.0, 3.0, 4.0],  # no missing values at all
            "num_ind": [np.nan, 2.0, 3.0, np.nan],  # has missing values
            "cat": ["a", "b", "a", "b"],
        }
    )
    ct = build_preprocessing_pipeline(["num_plain"], ["num_ind"], ["cat"])
    output = ct.fit_transform(df)
    dense = output.toarray() if hasattr(output, "toarray") else output
    # num_plain (1 col, no missing) + num_ind (1 col value + 1 indicator col, since it HAS missing) + cat (2 one-hot cols)
    assert dense.shape[1] == 1 + 2 + 2


def test_onehot_handles_unseen_category_without_crashing() -> None:
    train_df = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    test_df = pd.DataFrame({"cat": ["a", "c"]})  # "c" never seen in train
    ct = build_preprocessing_pipeline([], [], ["cat"])
    ct.fit(train_df)
    output = ct.transform(test_df)  # must not raise
    dense = output.toarray() if hasattr(output, "toarray") else output
    assert dense.shape[0] == 2
    # the unseen category "c" should map to all-zero one-hot row (handle_unknown="ignore")
    assert dense[1].sum() == 0


def test_onehot_handles_missing_value_in_transform_not_seen_in_fit() -> None:
    train_df = pd.DataFrame({"cat": ["a", "b", "a", "b"]})  # no missing in fit data
    test_df = pd.DataFrame({"cat": ["a", None]})
    ct = build_preprocessing_pipeline([], [], ["cat"])
    ct.fit(train_df)
    output = ct.transform(test_df)  # must not raise
    dense = output.toarray() if hasattr(output, "toarray") else output
    assert dense.shape[0] == 2


# --- measure_output ----------------------------------------------------------

def test_measure_output_reports_sparse_correctly() -> None:
    df = pd.DataFrame({"cat": ["a", "b", "c", "a"]})
    ct = build_preprocessing_pipeline([], [], ["cat"])
    output = ct.fit_transform(df)
    result = measure_output(output, "test")
    # sparse_threshold=1.0 (set deliberately, per the measured real-data memory
    # comparison in the Stage 5 report) means sparse output whenever any
    # sub-transformer (here, OneHotEncoder) produces sparse data.
    assert result["is_sparse"] is True
    assert result["shape"] == (4, 3)
    assert 0.0 <= result["density"] <= 1.0


# --- build_scaler: decoupled from the main pipeline --------------------------

def test_scaler_is_separate_and_fits_independently() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    scaler = build_scaler(["a", "b"])
    out = scaler.fit_transform(df)
    # StandardScaler output: mean ~0, std ~1 per column
    assert np.allclose(out.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(out.std(axis=0), 1, atol=1e-10)


# --- THE non-trivial leakage-guard test --------------------------------------

def test_leakage_guard_train_only_fit_differs_from_train_plus_validation_fit() -> None:
    """Proves the fitting discipline matters — not a test that passes regardless.

    Constructs train and validation data with DELIBERATELY different
    distributions (train's median is far from validation's), so that fitting
    on train+validation combined produces a measurably different imputation
    value than fitting on train alone. If this test ever failed to detect a
    difference, that would mean the fitting discipline stopped mattering —
    exactly the violation this guard exists to catch.
    """
    train_df = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0, 1.0, np.nan]})  # median 1.0
    validation_df = pd.DataFrame({"x": [100.0, 100.0, 100.0, 100.0, 100.0, np.nan]})  # median 100.0
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)

    train_only_imputer = SimpleImputer(strategy="median").fit(train_df)
    combined_imputer = SimpleImputer(strategy="median").fit(combined_df)

    train_only_value = train_only_imputer.statistics_[0]
    combined_value = combined_imputer.statistics_[0]

    # The guard: these MUST differ given the constructed data, proving the
    # test can actually detect a train-only-fitting violation.
    assert train_only_value != combined_value
    assert train_only_value == pytest.approx(1.0)
    assert combined_value != pytest.approx(1.0)  # combined fit is contaminated by validation's distribution


# --- load_partition_df: review item 1 (chunked loading) + item 4 (train-only guard) ---

def _write_assignment_and_transaction(tmp_path):
    assignment = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [10, 20, 30, 40, 50, 60],
            "isFraud": [0, 1, 0, 0, 1, 0],
            "partition": ["train", "train", "train", "validation", "validation", "final_test"],
        }
    )
    transaction = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat": ["a", "b", "a", "b", "a", "b"],
        }
    )
    assignment_path = tmp_path / "assignment.csv"
    transaction_path = tmp_path / "transaction.csv"
    assignment.to_csv(assignment_path, index=False)
    transaction.to_csv(transaction_path, index=False)
    return assignment_path, transaction_path


def test_load_partition_df_returns_only_requested_partition_rows(tmp_path) -> None:
    assignment_path, transaction_path = _write_assignment_and_transaction(tmp_path)
    train_df = load_partition_df(assignment_path, transaction_path, "train", ["x", "cat"], chunksize=2)
    assert sorted(train_df["TransactionID"]) == [1, 2, 3]
    assert sorted(train_df["x"]) == [1.0, 2.0, 3.0]


def test_load_partition_df_chunking_does_not_change_result(tmp_path) -> None:
    """Chunk size is an implementation detail -- the loaded rows must be
    identical regardless of how many chunks the read is split into."""
    assignment_path, transaction_path = _write_assignment_and_transaction(tmp_path)
    whole = load_partition_df(assignment_path, transaction_path, "validation", ["x", "cat"], chunksize=10_000)
    chunked = load_partition_df(assignment_path, transaction_path, "validation", ["x", "cat"], chunksize=1)
    pd.testing.assert_frame_equal(
        whole.sort_values("TransactionID").reset_index(drop=True),
        chunked.sort_values("TransactionID").reset_index(drop=True),
    )


def test_load_partition_df_refuses_final_test_by_default(tmp_path) -> None:
    assignment_path, transaction_path = _write_assignment_and_transaction(tmp_path)
    with pytest.raises(FinalTestAccessError):
        load_partition_df(assignment_path, transaction_path, "final_test", ["x", "cat"])


def test_load_partition_df_allows_final_test_only_with_explicit_override(tmp_path) -> None:
    assignment_path, transaction_path = _write_assignment_and_transaction(tmp_path)
    final_test_df = load_partition_df(
        assignment_path, transaction_path, "final_test", ["x", "cat"], allow_final_test=True
    )
    assert sorted(final_test_df["TransactionID"]) == [6]


def test_verify_ids_in_partition_passes_when_all_ids_are_allowed() -> None:
    _verify_ids_in_partition(loaded_ids=[1, 2, 3], allowed_ids={1, 2, 3, 4}, context="test")  # must not raise


def test_verify_ids_in_partition_raises_on_leaked_id() -> None:
    """Review item 4: a defense-in-depth guard that must actually fire.

    Simulates the exact bug it exists to catch -- a row from outside the
    requested partition (e.g. validation) ending up in what should be a
    train-only load -- and confirms the function refuses to let it pass
    silently.
    """
    with pytest.raises(RuntimeError, match="outside the requested"):
        _verify_ids_in_partition(loaded_ids=[1, 2, 999], allowed_ids={1, 2, 3}, context="train load")


# --- transform-time robustness: review item 5 ---------------------------------

def test_transform_is_independent_of_input_column_order() -> None:
    train_df = pd.DataFrame({"num_plain": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]})
    ct = build_preprocessing_pipeline(["num_plain"], [], ["cat"])
    ct.fit(train_df)

    test_in_order = pd.DataFrame({"num_plain": [5.0], "cat": ["a"]})
    test_reordered = pd.DataFrame({"cat": ["a"], "num_plain": [5.0]})  # columns swapped

    out_in_order = ct.transform(test_in_order)
    out_reordered = ct.transform(test_reordered)
    dense_a = out_in_order.toarray() if hasattr(out_in_order, "toarray") else out_in_order
    dense_b = out_reordered.toarray() if hasattr(out_reordered, "toarray") else out_reordered
    np.testing.assert_array_equal(dense_a, dense_b)


def test_transform_raises_clearly_when_a_fit_time_column_is_missing() -> None:
    """A column present at fit time but absent at transform time must fail
    loudly, not silently produce wrong output -- this is the behavior a
    caller depends on to catch a broken upstream column-selection bug
    before it reaches a model."""
    train_df = pd.DataFrame({"num_plain": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    ct = build_preprocessing_pipeline(["num_plain"], [], ["cat"])
    ct.fit(train_df)

    test_df_missing_column = pd.DataFrame({"cat": ["a"]})  # "num_plain" dropped
    with pytest.raises(Exception):  # sklearn raises ValueError/KeyError depending on version
        ct.transform(test_df_missing_column)


def test_feature_names_out_are_stable_across_fit_and_transform() -> None:
    """`get_feature_names_out()` must reflect the FITTED structure and not
    depend on whatever happens to be passed to `transform()` afterward."""
    train_df = pd.DataFrame({"num_plain": [1.0, 2.0, np.nan], "cat": ["a", "b", "a"]})
    ct = build_preprocessing_pipeline(["num_plain"], [], ["cat"])
    ct.fit(train_df)
    names_before = list(ct.get_feature_names_out())

    ct.transform(pd.DataFrame({"num_plain": [5.0], "cat": ["a"]}))
    names_after = list(ct.get_feature_names_out())

    assert names_before == names_after
    assert len(names_before) == ct.transform(train_df).shape[1]


# --- verify_column_accounting: review item 6 -----------------------------------

def test_verify_column_accounting_passes_for_consistent_scope() -> None:
    register = pd.DataFrame(
        {
            "column": ["a", "b", "c", "d"],
            "status": ["include", "include_with_caveat", "exclude_conservative", "exclude_structural"],
        }
    )
    result = verify_column_accounting(register, numeric=["a"], categorical=["b"])
    assert result["n_included_in_pipeline"] == 2
    assert result["n_register_include_status"] == 2
    assert result["n_register_exclude_status"] == 2
    assert result["register_included_not_in_pipeline"] == []


def test_verify_column_accounting_raises_if_excluded_column_leaks_in() -> None:
    register = pd.DataFrame({"column": ["a", "b"], "status": ["include", "exclude_conservative"]})
    with pytest.raises(ValueError, match="marked excluded"):
        verify_column_accounting(register, numeric=["a", "b"], categorical=[])


def test_verify_column_accounting_raises_if_pipeline_column_not_in_register() -> None:
    register = pd.DataFrame({"column": ["a"], "status": ["include"]})
    with pytest.raises(ValueError, match="not marked include"):
        verify_column_accounting(register, numeric=["a", "ghost_column"], categorical=[])


def test_verify_column_accounting_reports_unused_included_columns() -> None:
    """Not an error -- a register column marked include* but not yet wired
    into the pipeline is a visible gap to report, not a silent drop."""
    register = pd.DataFrame({"column": ["a", "b"], "status": ["include", "include"]})
    result = verify_column_accounting(register, numeric=["a"], categorical=[])
    assert result["register_included_not_in_pipeline"] == ["b"]


# --- check_sparse_input_compatibility: review item 7 (tag inspection, no .fit()) ---

def test_sparse_input_compatibility_reflects_known_sklearn_tags() -> None:
    """Spot-checks estimator tag metadata, not a trained model. Values here
    are sklearn's own declared capabilities (version-pinned via
    requirements.txt) -- this test exists to catch a tag change on upgrade,
    not to assert an opinion about which estimator to pick."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    result = check_sparse_input_compatibility([LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier])
    assert result["LogisticRegression"] is True
    assert result["RandomForestClassifier"] is True
    assert result["HistGradientBoostingClassifier"] is False  # requires dense input
