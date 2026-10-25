"""Stage 3 EDA utilities — IEEE-CIS Fraud Detection data.

Design note on memory strategy (why there's no row-sampling here):
every analysis in this module only ever reads a handful of columns at a
time (`usecols=...`), and at ~590K rows, a 1-8 column slice is a few
megabytes regardless of row count — column selection alone already bounds
memory far below what row-sampling would save, while staying exact instead
of approximate. Chunking is used only where it's actually needed: the one
full-width (394-column) missingness scan. Row-sampling was considered and
deliberately not used — see docs/EDA_REPORT.md for the full reasoning.

Every function is read-only and pure. None of them fit, impute, encode, or
otherwise transform the data the way a model would — that's Stage 5/6.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_CHUNKSIZE = 100_000


def target_distribution(
    path: str | Path,
    target_col: str = "isFraud",
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    """Exact fraud/non-fraud counts and prevalence, via a single-column chunked scan.

    Why this matters for fraud detection: severe class imbalance means
    accuracy is meaningless (predicting "never fraud" scores ~96%+ and
    catches nothing), and the minority class's absolute count sets a hard
    ceiling on how much signal any model can learn about fraud specifically.
    """
    fraud = non_fraud = 0
    for chunk in pd.read_csv(path, usecols=[target_col], chunksize=chunksize):
        fraud += int(chunk[target_col].sum())
        non_fraud += int((chunk[target_col] == 0).sum())
    total = fraud + non_fraud
    return {
        "fraud": fraud,
        "non_fraud": non_fraud,
        "total": total,
        "fraud_prevalence_pct": round(100 * fraud / total, 3) if total else 0.0,
        "imbalance_ratio_non_fraud_per_fraud": round(non_fraud / fraud, 1) if fraud else float("inf"),
    }


def grouped_numeric_summary(
    path: str | Path,
    value_col: str,
    group_col: str = "isFraud",
) -> dict:
    """Descriptive statistics for a numeric column, split by group, plus IQR outliers.

    Reads only the two needed columns directly (no chunking needed — two
    float/int columns at 590K rows is a few MB). Returns exact mean/median/
    std/quartiles per group and an IQR-based outlier count (no scipy: the
    IQR rule — below Q1-1.5*IQR or above Q3+1.5*IQR — needs only quantiles).
    """
    df = pd.read_csv(path, usecols=[value_col, group_col])
    out: dict = {"overall": _describe_with_iqr(df[value_col])}
    for group_value, sub in df.groupby(group_col):
        out[str(group_value)] = _describe_with_iqr(sub[value_col])
    return out


def _describe_with_iqr(series: pd.Series) -> dict:
    q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        "n": int(series.shape[0]),
        "mean": round(float(series.mean()), 2),
        "median": round(float(median), 2),
        "std": round(float(series.std()), 2),
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(iqr), 2),
        "iqr_outlier_count": int(outliers.shape[0]),
        "iqr_outlier_pct": round(100 * outliers.shape[0] / series.shape[0], 2) if series.shape[0] else 0.0,
        "min": round(float(series.min()), 2),
        "max": round(float(series.max()), 2),
    }


def standardized_mean_difference(
    mean_a: float, mean_b: float, std_a: float, std_b: float, n_a: int, n_b: int
) -> float:
    """Cohen's d via pooled standard deviation — a scale-free effect size.

    d = (mean_a - mean_b) / pooled_std, pooled_std weighted by each group's
    (n-1) so the larger group contributes more to the pooled variance
    estimate. Benchmarks (|d|): 0.2 small, 0.5 medium, 0.8 large.

    Assumptions and limitations (asked for explicitly — see docs/EDA_REPORT.md
    for the full discussion): assumes the two distributions are reasonably
    symmetric and have comparable variances; it is sensitive to outliers and
    skew, both of which are expected in TransactionAmt. It says nothing about
    which group has the outliers driving the difference, and nothing about
    direction of causality — it is a magnitude-of-difference summary, not
    a test and not an explanation.
    """
    pooled_var = ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_std


def relative_time_bins(
    path: str | Path,
    dt_col: str = "TransactionDT",
    target_col: str = "isFraud",
    bin_seconds: int = 86_400,
) -> pd.DataFrame:
    """Transaction volume and fraud rate per relative time bin (default: 1 day).

    `bin = TransactionDT // bin_seconds` is an index relative to the start of
    this file's own observed window — e.g. bin 0 is "the first bin_seconds
    of whatever period this file covers," NOT a calendar day. No real
    calendar date is invented or implied anywhere in this function.
    """
    df = pd.read_csv(path, usecols=[dt_col, target_col])
    df["_bin"] = df[dt_col] // bin_seconds
    grouped = df.groupby("_bin")[target_col].agg(["count", "sum"]).rename(
        columns={"count": "transaction_count", "sum": "fraud_count"}
    )
    grouped["fraud_rate_pct"] = (100 * grouped["fraud_count"] / grouped["transaction_count"]).round(3)
    grouped.index.name = f"relative_bin_of_{bin_seconds}s"
    return grouped.reset_index()


def missingness_by_group(
    path: str | Path,
    group_col: str = "isFraud",
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Per-column missing-% for each group value, via one chunked full-width pass.

    This is the one analysis in this module that genuinely needs chunking —
    it touches all ~394 columns, unlike everything else here which reads a
    handful of columns directly. The chunked accumulate-then-divide pattern
    (same as Stage 2's compute_missingness) gives exact percentages, split
    by group, without ever holding the full 683MB file in memory at once.
    """
    null_counts: dict[object, pd.Series] = {}
    row_counts: dict[object, int] = {}
    for chunk in pd.read_csv(path, chunksize=chunksize):
        for group_value, sub in chunk.groupby(group_col):
            nulls = sub.isna().sum()
            null_counts[group_value] = nulls if group_value not in null_counts else null_counts[group_value].add(nulls, fill_value=0)
            row_counts[group_value] = row_counts.get(group_value, 0) + len(sub)
    result = pd.DataFrame(
        {str(group_value): (null_counts[group_value] / row_counts[group_value] * 100).round(2) for group_value in null_counts}
    )
    result[f"gap_({'-'.join(str(g) for g in sorted(null_counts, reverse=True))})"] = result.iloc[:, 0] - result.iloc[:, 1]
    return result.sort_values(result.columns[-1], ascending=False, key=abs)


def categorical_fraud_rates(
    path: str | Path,
    categorical_cols: Iterable[str],
    target_col: str = "isFraud",
    min_count: int = 50,
) -> dict[str, pd.DataFrame]:
    """Fraud rate per category, for each named categorical column, in one combined read.

    `min_count` filters out categories with too few observations to give a
    stable rate estimate (a category with 3 rows and 1 fraud case is not a
    40%-fraud-rate finding, it's noise) — filtered rows are reported, not
    silently dropped from the summary.
    """
    categorical_cols = list(categorical_cols)
    df = pd.read_csv(path, usecols=categorical_cols + [target_col])
    out = {}
    for col in categorical_cols:
        grouped = df.groupby(col, dropna=False)[target_col].agg(["count", "sum"]).rename(
            columns={"count": "n", "sum": "fraud_count"}
        )
        grouped["fraud_rate_pct"] = (100 * grouped["fraud_count"] / grouped["n"]).round(3)
        grouped["stable_estimate"] = grouped["n"] >= min_count
        out[col] = grouped.sort_values("n", ascending=False)
    return out


def cramers_v(contingency: pd.DataFrame) -> dict:
    """Cramér's V — association strength between two categorical variables, via numpy only.

    Computed from first principles (no scipy): chi-squared statistic from
    observed vs. expected counts under independence, then normalized by
    sample size and table shape so the result falls in [0, 1].

    Assumptions and limitations (asked for explicitly):
    - The usual chi-squared rule of thumb is that expected cell counts
      should be >= 5 for the approximation to be trustworthy; this is
      checked and reported, not assumed.
    - Cramér's V has a known upward bias with many categories relative to
      sample size (it tends to overstate association for sparse tables) —
      a well-known limitation of the uncorrected version used here, which
      is why `min_expected_cell_count` is reported alongside the value.
    - It measures association strength only — not direction, not
      causation, and not which category drives the association (pair it
      with the per-category fraud rates from `categorical_fraud_rates`
      for that).
    """
    observed = contingency.to_numpy(dtype=float)
    n = observed.sum()
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / n
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    r, k = observed.shape
    cramers = math.sqrt((chi2 / n) / (min(r - 1, k - 1))) if min(r - 1, k - 1) > 0 else 0.0
    return {
        "chi2": round(chi2, 2),
        "cramers_v": round(cramers, 4),
        "n": int(n),
        "dof": (r - 1) * (k - 1),
        "min_expected_cell_count": round(float(expected.min()), 2),
        "expected_cell_count_assumption_met": bool(expected.min() >= 5),
    }


def identity_coverage_fraud_rates(
    transaction_path: str | Path,
    identity_path: str | Path,
    target_col: str = "isFraud",
    id_col: str = "TransactionID",
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    """Fraud rate for transactions with vs. without a matching identity row.

    Builds the identity-ID set once (cheap — a few hundred KB of ints), then
    does a single chunked pass over the transaction file tagging membership.
    This does NOT assume missing identity implies fraud — it measures the
    actual rate in both groups and reports both, letting the data answer the
    question rather than presupposing it.
    """
    identity_ids: set = set()
    for chunk in pd.read_csv(identity_path, usecols=[id_col], chunksize=chunksize):
        identity_ids.update(chunk[id_col].tolist())

    counts = {"has_identity": {"fraud": 0, "total": 0}, "no_identity": {"fraud": 0, "total": 0}}
    for chunk in pd.read_csv(transaction_path, usecols=[id_col, target_col], chunksize=chunksize):
        has_identity = chunk[id_col].isin(identity_ids)
        for key, mask in (("has_identity", has_identity), ("no_identity", ~has_identity)):
            sub = chunk.loc[mask, target_col]
            counts[key]["fraud"] += int(sub.sum())
            counts[key]["total"] += int(sub.shape[0])

    for key in counts:
        fraud, total = counts[key]["fraud"], counts[key]["total"]
        counts[key]["fraud_rate_pct"] = round(100 * fraud / total, 3) if total else 0.0
    return counts
