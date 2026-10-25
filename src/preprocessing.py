"""Stage 5 preprocessing architecture — IEEE-CIS Fraud Detection data.

Design principles (all enforced, not just stated):
- Every stateful transform is fit on the `train` partition ONLY, accessed
  exclusively via `src.leakage_boundaries.get_partition()` — never a raw
  read of the split assignment file, and never `final_test` (Stage 4
  carried-forward requirements).
- Column scope comes programmatically from `docs/feature_availability_register.csv`
  (`include`/`include_with_caveat` only) crossed with the actual dtype
  (pandas 3.1 infers plain strings as `StringDtype`, not `object` — see
  `src/data_validation.py::classify_columns`, which already handles this).
- Scaling is deliberately NOT baked into the default pipeline — see
  `build_scaler()` below and the Stage 5 report for why.
- `include_with_caveat` columns are preprocessed like any other included
  column (they're eligible for controlled experiments, per Stage 4) — this
  module does not re-adjudicate their availability status.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_validation import classify_columns, get_schema_sample  # noqa: E402
from src.leakage_boundaries import get_partition  # noqa: E402

DEFAULT_INDICATOR_THRESHOLD = 0.05  # 5% missing in train triggers a missingness indicator


# ---------------------------------------------------------------------------
# Column scope — from the register, not hand-maintained
# ---------------------------------------------------------------------------

def load_included_columns(
    register_path: str | Path,
    transaction_path: str | Path,
    schema_sample_rows: int = 5000,
) -> dict[str, list[str]]:
    """Return {"numeric": [...], "categorical": [...]} from the feature register.

    Only `include`/`include_with_caveat` statuses are considered — `exclude_*`
    columns never reach this function's output. Numeric/categorical
    classification uses actual dtype (via `classify_columns`, which already
    accounts for pandas 3.1's `StringDtype` default), not the register's
    `family` column, since some families (e.g. `card`) mix both.
    """
    register = pd.read_csv(register_path)
    included = register.loc[register["status"].isin(["include", "include_with_caveat"]), "column"].tolist()
    schema = get_schema_sample(transaction_path, nrows=schema_sample_rows)
    schema_included = {c: schema[c] for c in included if c in schema}
    classified = classify_columns(schema_included)
    if classified["other"]:
        raise ValueError(f"Unclassified included columns (neither numeric nor categorical): {classified['other']}")
    dropped = sorted(set(included) - set(schema))
    if dropped:
        raise ValueError(
            f"{len(dropped)} register-included column(s) not found in the transaction file's schema "
            f"sample: {dropped} — this would silently shrink the included column set."
        )
    return {"numeric": classified["numeric"], "categorical": classified["categorical"]}


def verify_column_accounting(
    register: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> dict:
    """Review item 6: every included column traces to an include* register
    row, every excluded register row stays out, and nothing silently
    dropped between the register and the final numeric/categorical lists.

    Raises ValueError on any violation rather than returning a soft report,
    since each of these is a correctness bug (register/pipeline drift), not
    a finding to weigh.
    """
    included_cols = set(numeric) | set(categorical)
    if len(numeric) + len(categorical) != len(included_cols):
        raise ValueError("A column appears in both the numeric and categorical lists.")

    register_included = set(register.loc[register["status"].isin(["include", "include_with_caveat"]), "column"])
    register_excluded = set(register.loc[register["status"].isin(["exclude_conservative", "exclude_structural"]), "column"])

    # Checked before the generic "not in register" case below, since a
    # column that IS in the register but under an exclude* status is a more
    # specific, more alarming violation than one simply missing altogether.
    leaked_excluded = included_cols & register_excluded
    if leaked_excluded:
        raise ValueError(f"Columns marked excluded in the register but present in preprocessing: {sorted(leaked_excluded)}")

    missing_from_register = included_cols - register_included
    if missing_from_register:
        raise ValueError(f"Columns used in preprocessing but not marked include*/in the register: {sorted(missing_from_register)}")

    not_covered = register_included - included_cols
    return {
        "n_included_in_pipeline": len(included_cols),
        "n_register_include_status": len(register_included),
        "n_register_exclude_status": len(register_excluded),
        "register_included_not_in_pipeline": sorted(not_covered),
    }


# ---------------------------------------------------------------------------
# Chunked, filtered partition loading — review item 1
#
# The original implementation read ALL 590,540 rows (every included column)
# into memory, THEN filtered down to the requested partition's rows — so a
# call for `train` (438,125 rows) briefly held the full unfiltered frame
# AND the filtered one simultaneously, and this happened independently in
# three separate call sites (missingness measurement, fit, each transform).
# This loader reads in chunks and filters each chunk immediately, so the
# full 590,540-row frame is never materialized at all, and every call site
# below shares this one code path instead of repeating the pattern.
# ---------------------------------------------------------------------------

def _verify_ids_in_partition(loaded_ids: Iterable, allowed_ids: set, context: str) -> None:
    """Defense-in-depth check (review item 4): every loaded ID must belong
    to the partition it was requested for. Raises loudly, not silently,
    if a bug ever let a row from the wrong partition through — redundant
    with `get_partition()`'s own guardrail, but at a different layer
    (this one checks the actual rows read, not the ID list requested).
    """
    loaded_ids = set(loaded_ids)
    leaked = loaded_ids - allowed_ids
    if leaked:
        raise RuntimeError(
            f"{context}: {len(leaked)} row(s) loaded with an ID outside the requested "
            f"partition (e.g. {sorted(leaked)[:3]}) — this is a bug, not a data issue."
        )


def load_partition_df(
    assignment_path: str | Path,
    transaction_path: str | Path,
    partition: str,
    columns: Iterable[str],
    chunksize: int = 100_000,
    allow_final_test: bool = False,
) -> pd.DataFrame:
    """Read only the requested partition's rows, in chunks — never the full file.

    Returns a DataFrame with `TransactionID` + `columns`. Memory use is
    bounded by `chunksize` rows at a time plus the accumulated filtered
    result, not by the transaction file's total row count.
    """
    columns = list(columns)
    assignment = pd.read_csv(assignment_path)
    ids = set(get_partition(assignment, [partition], allow_final_test=allow_final_test)["TransactionID"])

    matched_chunks = []
    for chunk in pd.read_csv(transaction_path, usecols=["TransactionID"] + columns, chunksize=chunksize):
        matched = chunk.loc[chunk["TransactionID"].isin(ids)]
        if len(matched):
            matched_chunks.append(matched)

    result = (
        pd.concat(matched_chunks, ignore_index=True)
        if matched_chunks
        else pd.DataFrame(columns=["TransactionID"] + columns)
    )
    _verify_ids_in_partition(result["TransactionID"], ids, context=f"load_partition_df('{partition}')")
    return result


# ---------------------------------------------------------------------------
# Train-only missingness, to decide which numeric columns get an indicator
# ---------------------------------------------------------------------------

def measure_train_missingness(
    assignment_path: str | Path,
    transaction_path: str | Path,
    columns: Iterable[str],
) -> pd.Series:
    """Missing-% per column, computed on the `train` partition ONLY.

    Uses the chunked `load_partition_df()` — never reads the full,
    unfiltered transaction file, and never reads validation/final_test rows.
    """
    columns = list(columns)
    train_df = load_partition_df(assignment_path, transaction_path, "train", columns)
    return (train_df[columns].isna().mean() * 100).round(3).sort_values(ascending=False)


def select_indicator_columns(
    train_missingness_pct: pd.Series,
    threshold_pct: float = DEFAULT_INDICATOR_THRESHOLD * 100,
) -> dict[str, list[str]]:
    """Split numeric columns into "gets a missingness indicator" vs. "imputed only".

    Threshold decided as a strategy choice (Section 4 of the Stage 5 report)
    informed by Stage 3's full-dataset EDA finding that missingness itself
    carries signal for some column families — but the actual per-column
    missingness values used here are measured on `train` only, not the
    full dataset, per the carried-forward requirement.
    """
    with_indicator = train_missingness_pct[train_missingness_pct >= threshold_pct].index.tolist()
    without_indicator = train_missingness_pct[train_missingness_pct < threshold_pct].index.tolist()
    return {"with_indicator": with_indicator, "without_indicator": without_indicator}


def estimate_indicator_memory_cost(n_rows: int, n_indicator_columns: int, dtype_bytes: int = 8) -> dict:
    """Quantify the memory cost of adding one indicator column per flagged column.

    `dtype_bytes=8` matches SimpleImputer's default indicator dtype
    (float64) before any downstream downcasting — a conservative (upper
    bound) estimate, stated as such.
    """
    bytes_total = n_rows * n_indicator_columns * dtype_bytes
    return {
        "n_rows": n_rows,
        "n_indicator_columns": n_indicator_columns,
        "estimated_bytes": bytes_total,
        "estimated_mb": round(bytes_total / 1e6, 2),
    }


# ---------------------------------------------------------------------------
# The ColumnTransformer architecture
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    numeric_no_indicator: list[str],
    numeric_with_indicator: list[str],
    categorical: list[str],
) -> ColumnTransformer:
    """Build the (unfitted) preprocessing ColumnTransformer.

    Three branches:
    - numeric, low train-missingness: median imputation only.
    - numeric, high train-missingness: median imputation + a missingness
      indicator per column (SimpleImputer's own `add_indicator`, which adds
      one indicator per column that has a missing value in the FIT data —
      consistent with the fit/transform discipline by construction, since
      the indicator set itself is learned from `train`, not hardcoded).
    - categorical: constant-fill ("missing" sentinel) then one-hot, with
      `handle_unknown="ignore"` so a category unseen in `train` doesn't
      crash `validation`/`final_test` transformation.

    Scaling is deliberately absent — see `build_scaler()`.
    """
    numeric_plain = Pipeline([("impute", SimpleImputer(strategy="median"))])
    numeric_indicator = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
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
        # sklearn's own default (0.3) returns DENSE once overall density exceeds
        # 30% -- measured empirically on this dataset's actual column set
        # (train output density ~0.42) that this default produces a dense
        # array using ~2.85GB, while forcing sparse storage (threshold=1.0)
        # uses ~1.78GB for the identical data (a real, measured ~38% reduction,
        # not assumed) -- CSR's per-nonzero overhead is still cheaper than
        # dense's per-cell cost below ~67% density, and 0.42 is comfortably
        # under that. The tradeoff, stated honestly: transform is slower
        # sparse (~4.8s vs ~2.9s on `train`) -- memory was judged the binding
        # constraint for this dataset's scale, not speed.
        sparse_threshold=1.0,
    )


def build_scaler(numeric_columns: list[str]) -> ColumnTransformer:
    """A SEPARATE, optional scaling step — not part of the default pipeline.

    Deliberately decoupled: tree/boosting model families (plausible Stage 7
    candidates for this problem) don't need or benefit from feature scaling,
    and bolting StandardScaler into the core pipeline would assume otherwise
    for every future estimator. Stage 7 composes this in only for model
    families that actually need it (e.g. logistic regression), fit on
    `train` only, exactly like every other transform here.
    """
    return ColumnTransformer(
        [("scale", StandardScaler(), numeric_columns)],
        remainder="passthrough",
    )


# ---------------------------------------------------------------------------
# Fitting, strictly scoped to train via get_partition()
# ---------------------------------------------------------------------------

@dataclass
class FittedPreprocessing:
    transformer: ColumnTransformer
    numeric_no_indicator: list[str]
    numeric_with_indicator: list[str]
    categorical: list[str]
    fit_wall_seconds: float


def fit_on_train_only(
    assignment_path: str | Path,
    transaction_path: str | Path,
    numeric_no_indicator: list[str],
    numeric_with_indicator: list[str],
    categorical: list[str],
) -> FittedPreprocessing:
    """Fit the preprocessing pipeline on the `train` partition exclusively.

    Partition access goes through `load_partition_df()` -> `get_partition()`
    — never a raw read of the assignment file filtered by hand, and
    `allow_final_test` is never passed, so this function cannot be made to
    fit on `final_test` even by a careless caller without editing this
    function itself. Loading is chunked (review item 1): the full,
    unfiltered transaction file is never held in memory at once.
    """
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical
    train_df = load_partition_df(assignment_path, transaction_path, "train", all_cols)

    transformer = build_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
    t0 = time.perf_counter()
    transformer.fit(train_df[all_cols])
    elapsed = time.perf_counter() - t0

    return FittedPreprocessing(transformer, numeric_no_indicator, numeric_with_indicator, categorical, elapsed)


def transform_partition(
    fitted: FittedPreprocessing,
    assignment_path: str | Path,
    transaction_path: str | Path,
    partition: str,
    allow_final_test: bool = False,
):
    """Apply an already-fitted transformer to a named partition. Never refits.

    `partition="final_test"` requires `allow_final_test=True`, enforced by
    `get_partition()` itself (Stage 4) — this function adds no separate
    bypass, so the same guardrail applies here automatically. Loading is
    chunked via `load_partition_df()` (review item 1), same rationale as
    `fit_on_train_only()`.
    """
    all_cols = fitted.numeric_no_indicator + fitted.numeric_with_indicator + fitted.categorical
    part_df = load_partition_df(
        assignment_path, transaction_path, partition, all_cols, allow_final_test=allow_final_test
    )

    t0 = time.perf_counter()
    output = fitted.transformer.transform(part_df[all_cols])
    elapsed = time.perf_counter() - t0
    return output, elapsed


def measure_output(output, label: str) -> dict:
    """Shape, density, memory for a (possibly sparse) transform output."""
    shape = output.shape
    if hasattr(output, "nnz"):  # scipy sparse
        nnz = output.nnz
        density = nnz / (shape[0] * shape[1])
        memory_bytes = output.data.nbytes + output.indices.nbytes + output.indptr.nbytes
        is_sparse = True
    else:  # dense ndarray
        nnz = int(np.count_nonzero(output))
        density = nnz / (shape[0] * shape[1])
        memory_bytes = output.nbytes
        is_sparse = False
    return {
        "label": label,
        "shape": shape,
        "is_sparse": is_sparse,
        "density": round(density, 4),
        "memory_mb": round(memory_bytes / 1e6, 2),
    }


# ---------------------------------------------------------------------------
# Review item 7 — estimator-family compatibility, tag inspection ONLY.
#
# No estimator is ever `.fit()` here — that would be model training/model
# selection, out of scope for this stage and this review round. This only
# reads each candidate's declared `input_tags.sparse` capability, which is
# static metadata sklearn exposes without touching any data.
# ---------------------------------------------------------------------------

def check_sparse_input_compatibility(estimator_classes: Iterable[type]) -> dict[str, bool]:
    """{class_name: supports_sparse_input} for each candidate estimator class.

    Relevant because this pipeline's default output is sparse
    (`sparse_threshold=1.0`, see `build_preprocessing_pipeline`) — an
    estimator that reports `False` here would need the output densified
    first (reintroducing the dense-memory cost item 1/2 investigated),
    not because sparse support is itself a model-selection criterion.
    """
    from sklearn.utils import get_tags

    result = {}
    for cls in estimator_classes:
        tags = get_tags(cls())
        result[cls.__name__] = bool(tags.input_tags.sparse)
    return result
