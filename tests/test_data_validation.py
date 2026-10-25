"""Pytest-style tests for src/data_validation.py.

NOT YET EXECUTED: pytest is not installed in .venv as of Stage 2 (see
docs/DATA_VALIDATION_REPORT.md "Unresolved Issues"). This file is written
in standard pytest conventions so it's ready to run as soon as pytest is
approved and installed — `python-testing-patterns` skill's fixture/
parametrize patterns are used below. The underlying functions were
exercised directly (without pytest) in scripts/run_stage2_validation.py,
and that run's output is the actual Stage 2 evidence.

Run (once pytest is installed):
    .venv/bin/pytest tests/test_data_validation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest  # type: ignore[import-not-found]  # not installed yet — see module docstring

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_validation import (  # noqa: E402
    check_id_uniqueness,
    check_target_presence,
    classify_columns,
    diff_schema,
    flag_potential_leakage_columns,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ieee-fraud-detection"


@pytest.fixture(scope="session")
def train_transaction_path() -> Path:
    return DATA_DIR / "train_transaction.csv"


@pytest.fixture(scope="session")
def test_transaction_path() -> Path:
    return DATA_DIR / "test_transaction.csv"


def test_train_transaction_has_target(train_transaction_path: Path) -> None:
    assert check_target_presence(train_transaction_path, "isFraud") is True


def test_test_transaction_has_no_target(test_transaction_path: Path) -> None:
    # Kaggle withholds isFraud for the test split — this must stay False.
    assert check_target_presence(test_transaction_path, "isFraud") is False


def test_train_transaction_id_is_unique(train_transaction_path: Path) -> None:
    result = check_id_uniqueness(train_transaction_path, "TransactionID")
    assert result["duplicates"] == 0
    assert result["total"] == result["unique"]


@pytest.mark.parametrize(
    "schema, expected_numeric, expected_categorical",
    [
        ({"a": "int64", "b": "float64", "c": "object"}, ["a", "b"], ["c"]),
        ({"a": "str", "b": "string"}, [], ["a", "b"]),  # pandas >= 3.0 string dtype
        ({}, [], []),
    ],
)
def test_classify_columns(schema, expected_numeric, expected_categorical) -> None:
    result = classify_columns(schema)
    assert result["numeric"] == expected_numeric
    assert result["categorical"] == expected_categorical
    assert result["other"] == []


def test_diff_schema_detects_dtype_change() -> None:
    expected = {"a": "int64", "b": "object"}
    actual = {"a": "float64", "b": "object"}
    diff = diff_schema(expected, actual)
    assert diff["dtype_changes"] == {"a": ("int64", "float64")}
    assert diff["missing_columns"] == []
    assert diff["added_columns"] == []


def test_diff_schema_detects_missing_and_added_columns() -> None:
    expected = {"a": "int64", "b": "object"}
    actual = {"a": "int64", "c": "object"}
    diff = diff_schema(expected, actual)
    assert diff["missing_columns"] == ["b"]
    assert diff["added_columns"] == ["c"]


def test_flag_potential_leakage_columns_excludes_target_and_id() -> None:
    columns = ["TransactionID", "isFraud", "TransactionAmt", "V1", "C1", "D1"]
    result = flag_potential_leakage_columns(columns)
    assert set(result["exclude_from_features"]) == {"TransactionID", "isFraud"}
    assert set(result["unverifiable_from_schema_alone"]) == {"V1", "C1", "D1"}
