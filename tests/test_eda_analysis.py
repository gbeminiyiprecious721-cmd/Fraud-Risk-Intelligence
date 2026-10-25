"""Tests for the pure, deterministic functions in src/eda_analysis.py.

Scoped to the functions that don't require reading the real dataset
(standardized_mean_difference, cramers_v, the IQR helper) — the I/O-heavy
functions are exercised directly against the real data in
notebooks/01_exploratory_data_analysis.ipynb, same pattern as Stage 2.

Run: .venv/bin/pytest tests/test_eda_analysis.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eda_analysis import (  # noqa: E402
    _describe_with_iqr,
    cramers_v,
    standardized_mean_difference,
)


def test_standardized_mean_difference_zero_when_means_equal() -> None:
    d = standardized_mean_difference(mean_a=10, mean_b=10, std_a=2, std_b=2, n_a=100, n_b=100)
    assert d == 0.0


def test_standardized_mean_difference_known_value() -> None:
    # Equal variances and equal n: d reduces to (mean_a - mean_b) / std.
    d = standardized_mean_difference(mean_a=12, mean_b=10, std_a=4, std_b=4, n_a=50, n_b=50)
    assert d == pytest.approx(0.5, abs=1e-9)


def test_standardized_mean_difference_sign_reflects_direction() -> None:
    d_pos = standardized_mean_difference(mean_a=10, mean_b=5, std_a=5, std_b=5, n_a=30, n_b=30)
    d_neg = standardized_mean_difference(mean_a=5, mean_b=10, std_a=5, std_b=5, n_a=30, n_b=30)
    assert d_pos > 0
    assert d_neg < 0
    assert d_pos == pytest.approx(-d_neg, abs=1e-9)


def test_cramers_v_zero_for_independent_table() -> None:
    # Perfectly proportional rows -> no association -> V == 0.
    table = pd.DataFrame({"colA": [100, 200], "colB": [100, 200]}, index=["row1", "row2"])
    result = cramers_v(table)
    assert result["cramers_v"] == pytest.approx(0.0, abs=1e-6)
    assert result["chi2"] == pytest.approx(0.0, abs=1e-6)


def test_cramers_v_near_one_for_perfect_association() -> None:
    # Each row maps entirely to one column -> perfect association -> V near 1.
    table = pd.DataFrame({"colA": [1000, 0], "colB": [0, 1000]}, index=["row1", "row2"])
    result = cramers_v(table)
    assert result["cramers_v"] == pytest.approx(1.0, abs=1e-6)


def test_cramers_v_reports_expected_cell_count_assumption() -> None:
    sparse_table = pd.DataFrame({"colA": [3, 1], "colB": [2, 1]}, index=["row1", "row2"])
    result = cramers_v(sparse_table)
    assert "expected_cell_count_assumption_met" in result
    assert result["expected_cell_count_assumption_met"] is False  # tiny counts, rule of thumb fails


def test_describe_with_iqr_flags_known_outlier() -> None:
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])
    result = _describe_with_iqr(series)
    assert result["iqr_outlier_count"] == 1
    assert result["max"] == 1000
    assert result["n"] == 10
