"""Stage 2 dataset validation utilities — IEEE-CIS Fraud Detection data.

Every function here is read-only (never writes to the source CSVs), pure
(same input -> same output, no hidden state), and memory-conscious (reads
in chunks or selected columns rather than loading a full 394-column,
~650MB file at once). None of them fit, learn from, or transform the data
in a way a model would — that boundary belongs to later stages
(Preprocessing, Feature Engineering), not Dataset Discovery & Validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_CHUNKSIZE = 100_000


@dataclass
class FileIntegrityResult:
    path: str
    exists: bool
    size_bytes: int
    row_count: int | None
    readable: bool
    error: str | None = None


def check_file_integrity(
    path: str | Path,
    count_rows: bool = True,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> FileIntegrityResult:
    """Confirm a CSV exists, is readable, and (optionally) count its rows.

    Row counting reads only the first column per chunk, so this stays cheap
    even on the largest files — a 652MB file's row count costs roughly the
    I/O of its first column, not the whole file.
    """
    p = Path(path)
    if not p.exists():
        return FileIntegrityResult(str(p), False, 0, None, False, "file does not exist")

    size = p.stat().st_size
    row_count = None
    readable = True
    error = None
    try:
        if count_rows:
            row_count = 0
            for chunk in pd.read_csv(p, chunksize=chunksize, usecols=[0]):
                row_count += len(chunk)
    except Exception as exc:  # noqa: BLE001 - any parse failure is itself the finding
        readable = False
        error = str(exc)
    return FileIntegrityResult(str(p), True, size, row_count, readable, error)


def get_schema_sample(path: str | Path, nrows: int = 5000) -> dict[str, str]:
    """Infer column names and dtypes from a sample, without loading the full file.

    Known limitation: a sample can miss a rare mixed-type value that only
    appears later in the file (e.g. a string value in an otherwise-numeric
    column, which would make pandas infer `object` for the *full* file but
    not for the sample). Treat this as a fast first check, not a guarantee —
    `compute_missingness` below reads every row and would surface such a
    column indirectly via unexpected null/dtype behavior.
    """
    df = pd.read_csv(path, nrows=nrows)
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def check_target_presence(path: str | Path, target_col: str = "isFraud", nrows: int = 5) -> bool:
    """Confirm whether the target column exists in this file.

    The single most important Stage 2 fact for this project: training data
    must carry the label; the Kaggle test files must not (their labels are
    held back for leaderboard scoring) — confirming this programmatically
    prevents a later stage from silently assuming a column that isn't there.
    """
    df = pd.read_csv(path, nrows=nrows)
    return target_col in df.columns


def check_id_uniqueness(
    path: str | Path,
    id_col: str = "TransactionID",
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    """Check whether an identifier column is unique, reading only that column.

    A duplicated ID would silently corrupt any join or row-level evaluation
    later (double-counting a transaction, or joining one identity row to
    more than one transaction). Reading only the ID column keeps this cheap
    regardless of how many feature columns the file has.
    """
    seen: set = set()
    duplicates = 0
    total = 0
    for chunk in pd.read_csv(path, usecols=[id_col], chunksize=chunksize):
        for value in chunk[id_col]:
            total += 1
            if value in seen:
                duplicates += 1
            else:
                seen.add(value)
    return {"total": total, "unique": len(seen), "duplicates": duplicates}


def check_join_coverage(
    transaction_path: str | Path,
    identity_path: str | Path,
    id_col: str = "TransactionID",
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    """Measure what fraction of transaction rows have a matching identity row.

    `train_identity` covers only a subset of `train_transaction` BY DESIGN —
    this turns that into a measured number instead of an assumption, and
    separately checks for orphan identity rows (an identity row with no
    matching transaction), which *would* indicate a real data problem.
    """
    txn_ids: set = set()
    for chunk in pd.read_csv(transaction_path, usecols=[id_col], chunksize=chunksize):
        txn_ids.update(chunk[id_col].tolist())

    identity_ids: set = set()
    for chunk in pd.read_csv(identity_path, usecols=[id_col], chunksize=chunksize):
        identity_ids.update(chunk[id_col].tolist())

    matched = txn_ids & identity_ids
    orphan_identity_rows = identity_ids - txn_ids

    return {
        "n_transactions": len(txn_ids),
        "n_identity": len(identity_ids),
        "n_matched": len(matched),
        "coverage_pct": round(100 * len(matched) / len(txn_ids), 2) if txn_ids else 0.0,
        "orphan_identity_rows": len(orphan_identity_rows),
    }


def compute_missingness(
    path: str | Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    columns: Iterable[str] | None = None,
) -> pd.Series:
    """Compute per-column missing-value percentage without loading the whole file at once.

    The chunked accumulate-then-divide pattern gives the identical number
    a full `df.isna().mean()` would on a machine with enough RAM to load
    everything — it's the version that also works when there isn't enough
    RAM, at no accuracy cost.
    """
    null_counts: pd.Series | None = None
    total_rows = 0
    read_kwargs: dict = {"chunksize": chunksize}
    if columns is not None:
        read_kwargs["usecols"] = list(columns)
    for chunk in pd.read_csv(path, **read_kwargs):
        total_rows += len(chunk)
        chunk_nulls = chunk.isna().sum()
        null_counts = chunk_nulls if null_counts is None else null_counts.add(chunk_nulls, fill_value=0)
    if null_counts is None or total_rows == 0:
        return pd.Series(dtype=float)
    return (null_counts / total_rows * 100).round(2).sort_values(ascending=False)


def check_transactiondt_ordering(
    path: str | Path,
    dt_col: str = "TransactionDT",
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    """Check whether TransactionDT is sorted in file-row order, and report its range.

    Feeds directly into the Stage 4 leakage-boundary decision: if the file
    is already time-sorted, a naive shuffled split would scramble that order
    and leak future rows into training; the split must be done by sorting on
    TransactionDT explicitly, not by trusting row order blindly either way.
    """
    prev_chunk_max = None
    is_sorted = True
    overall_min = None
    overall_max = None
    for chunk in pd.read_csv(path, usecols=[dt_col], chunksize=chunksize):
        values = chunk[dt_col]
        chunk_min, chunk_max = values.min(), values.max()
        overall_min = chunk_min if overall_min is None else min(overall_min, chunk_min)
        overall_max = chunk_max if overall_max is None else max(overall_max, chunk_max)
        if not values.is_monotonic_increasing:
            is_sorted = False
        if prev_chunk_max is not None and chunk_min < prev_chunk_max:
            is_sorted = False
        prev_chunk_max = chunk_max
    return {
        "min": int(overall_min),
        "max": int(overall_max),
        "range_seconds": int(overall_max - overall_min),
        "sorted_ascending_in_file": is_sorted,
    }


def compare_dt_ranges(
    train_path: str | Path,
    test_path: str | Path,
    dt_col: str = "TransactionDT",
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    """Compare TransactionDT ranges between two files (e.g. train vs. test).

    If the second file's minimum DT is >= the first's maximum, that
    structurally confirms a chronological holdout (the second file is
    strictly "the future" relative to the first) — this is what actually
    validates the Stage 0 hypothesis that this dataset needs a time-based
    split, rather than taking it on faith.
    """
    first = check_transactiondt_ordering(train_path, dt_col, chunksize)
    second = check_transactiondt_ordering(test_path, dt_col, chunksize)
    gap = second["min"] - first["max"]
    return {
        "first_range": (first["min"], first["max"]),
        "second_range": (second["min"], second["max"]),
        "second_starts_after_first_ends": gap >= 0,
        "gap_seconds": gap,
    }


def classify_columns(schema: dict[str, str]) -> dict[str, list[str]]:
    """Classify columns into numeric/categorical/other from inferred dtypes.

    This is a schema-level classification only — it says what *kind* of
    preprocessing a column will likely need (Stage 5), not what it *means*
    or how it relates to the target. That's Stage 3 (EDA), deliberately not
    done here.

    Note: pandas >= 3.0 infers plain string columns as `StringDtype`
    (`str(dtype) == "str"`), not the legacy `object` dtype — both are
    treated as categorical here. Verified against this project's actual
    pandas 3.1.0 environment; an environment on an older pandas would still
    classify correctly via the `object` branch.
    """
    numeric, categorical, other = [], [], []
    for col, dtype in schema.items():
        if dtype.startswith(("int", "float")):
            numeric.append(col)
        elif dtype in ("object", "category", "bool", "str", "string"):
            categorical.append(col)
        else:
            other.append(col)
    return {"numeric": numeric, "categorical": categorical, "other": other}


def diff_schema(expected: dict[str, str], actual: dict[str, str]) -> dict:
    """Compare an actual schema against a previously recorded one and report drift.

    This is the check that catches "the data changed under us" — a renamed
    or reordered column, or a dtype that flipped from numeric to `object`
    (often caused by an unexpected string value appearing) — on a future
    rerun, before it breaks something downstream silently.
    """
    expected_cols, actual_cols = set(expected), set(actual)
    missing = sorted(expected_cols - actual_cols)
    added = sorted(actual_cols - expected_cols)
    dtype_changes = {
        col: (expected[col], actual[col])
        for col in expected_cols & actual_cols
        if expected[col] != actual[col]
    }
    return {"missing_columns": missing, "added_columns": added, "dtype_changes": dtype_changes}


def flag_potential_leakage_columns(
    columns: Iterable[str],
    target_col: str = "isFraud",
    id_col: str = "TransactionID",
) -> dict:
    """Flag columns that should never be used as model features, structurally.

    This only catches the mechanical cases (the target itself, the raw row
    identifier) — trivial to check, embarrassing to miss. It explicitly
    cannot detect subtler leakage in the anonymized V/C/D columns from the
    column name alone; that requires either domain documentation (not
    published for this dataset) or empirical checks in later stages, and is
    reported here as unresolved, not silently assumed safe.
    """
    columns = list(columns)
    exclude_always = [c for c in columns if c in (target_col, id_col)]
    anonymized_unverifiable = [
        c for c in columns if c.startswith(("V", "C", "D")) and c not in exclude_always
    ]
    return {
        "exclude_from_features": exclude_always,
        "unverifiable_from_schema_alone": anonymized_unverifiable,
    }
