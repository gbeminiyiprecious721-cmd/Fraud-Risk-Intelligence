"""Stage 16 -- repeatable data-contract tests for the approved, labelled
`train` partition.

Two tiers, split for speed (per explicit instruction):
  - FAST (no marker, run by default): register-level and CSV-header-level
    checks only -- no full-file read, completes in well under a second.
  - COMPREHENSIVE (marked `data_contract_full`): a full scan of the
    complete `train` partition via the project's own partition-safe
    `load_partition_df` -- reused unchanged, never reimplemented. Never
    touches `validation` or `final_test`.

Both tiers use ONLY the already-approved Stage 2/4 schema and source-data
evidence (`docs/feature_availability_register.csv`,
`docs/_stage4_split_assignment.csv`) -- no invented transaction-amount
bounds, fraud-rate limits, or demographic attributes are asserted
anywhere in this file. Legitimate feature missingness (anything other
than the two structural columns, `TransactionID` and `isFraud`) is
explicitly NOT treated as a violation -- see
`test_legitimate_feature_missingness_is_not_a_contract_violation`.
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

STRUCTURAL_COLUMNS = ["TransactionID", "isFraud"]  # register status == exclude_structural
EXPECTED_PARTITION_ROW_COUNTS = {"train": 438_125, "validation": 75_560, "final_test": 76_855}
EXPECTED_FEATURE_COLUMN_COUNT = 392  # register include + include_with_caveat; matches
# Stage 13's FeatureSchema.raw_columns count, independently cross-checked here.

# A small, already-documented (Stage 2/3/5) sample of columns known to
# carry real, legitimate missingness -- used only to PROVE this contract
# does not (and must not) reject nulls in feature columns.
KNOWN_LEGITIMATELY_MISSING_COLUMNS = ["D1", "D4", "D10", "dist1"]


def _register() -> pd.DataFrame:
    return pd.read_csv(REGISTER_PATH)


def _required_feature_columns() -> list[str]:
    reg = _register()
    return reg[reg["status"].isin(["include", "include_with_caveat"])]["column"].tolist()


# ---------------------------------------------------------------------------
# FAST tier -- register/header/assignment-file level only, no full raw-data read
# ---------------------------------------------------------------------------

def test_register_has_exactly_the_documented_structural_columns():
    reg = _register()
    structural = reg[reg["status"] == "exclude_structural"]["column"].tolist()
    assert sorted(structural) == sorted(STRUCTURAL_COLUMNS)


def test_register_has_no_duplicate_column_names():
    reg = _register()
    assert reg["column"].is_unique, "duplicate column name(s) in the feature-availability register"


def test_register_feature_column_count_matches_known_build():
    assert len(_required_feature_columns()) == EXPECTED_FEATURE_COLUMN_COUNT


def test_csv_header_contains_every_required_column_exactly_once():
    header = pd.read_csv(TRAIN_TXN, nrows=0).columns.tolist()
    assert len(header) == len(set(header)), "duplicate column name(s) in the real train_transaction.csv header"
    required = set(_required_feature_columns()) | set(STRUCTURAL_COLUMNS)
    missing = required - set(header)
    assert not missing, f"required column(s) missing from the real CSV header: {sorted(missing)}"


def test_assignment_file_has_the_documented_partition_row_counts():
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    counts = assignment["partition"].value_counts().to_dict()
    assert counts == EXPECTED_PARTITION_ROW_COUNTS


def test_assignment_file_has_no_duplicate_or_missing_transaction_ids():
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    assert assignment["TransactionID"].is_unique
    assert assignment["TransactionID"].notna().all()


def test_assignment_file_isfraud_is_strictly_binary_where_labelled():
    # final_test rows are NOT read here -- this only inspects the
    # assignment file's own label column for train/validation, which
    # carries a real label; final_test's label is never consulted by
    # this project outside Stage 10's one authorised evaluation.
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    labelled = assignment[assignment["partition"] != "final_test"]
    assert set(labelled["isFraud"].unique()) <= {0, 1}
    assert labelled["isFraud"].notna().all()


# ---------------------------------------------------------------------------
# COMPREHENSIVE tier -- full `train`-partition scan, train-only, never final_test
# ---------------------------------------------------------------------------

@pytest.mark.data_contract_full
def test_full_train_partition_isfraud_is_strictly_binary_with_no_nulls():
    from src.preprocessing import load_partition_df

    df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", ["isFraud"])
    assert df["isFraud"].notna().all()
    assert set(df["isFraud"].unique()) == {0, 1}
    assert df["isFraud"].dtype.kind in "iu", f"isFraud dtype is {df['isFraud'].dtype}, expected an integer type"


@pytest.mark.data_contract_full
def test_full_train_partition_transactionid_is_unique_not_null_and_matches_documented_count():
    from src.preprocessing import load_partition_df

    df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", [])
    assert len(df) == EXPECTED_PARTITION_ROW_COUNTS["train"]
    assert df["TransactionID"].is_unique
    assert df["TransactionID"].notna().all()


@pytest.mark.data_contract_full
def test_full_train_partition_transactiondt_has_no_nulls_and_is_numeric():
    from src.preprocessing import load_partition_df

    df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", ["TransactionDT"])
    assert df["TransactionDT"].notna().all()
    assert pd.api.types.is_numeric_dtype(df["TransactionDT"])


@pytest.mark.data_contract_full
def test_full_train_partition_has_every_required_feature_column_present():
    from src.preprocessing import load_partition_df

    required = _required_feature_columns()
    df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", required)
    missing = set(required) - set(df.columns)
    assert not missing, f"required feature column(s) absent from the loaded train partition: {sorted(missing)}"
    assert len(df) == EXPECTED_PARTITION_ROW_COUNTS["train"]


@pytest.mark.data_contract_full
def test_legitimate_feature_missingness_is_not_a_contract_violation():
    """Proves this contract does NOT reject nulls in feature columns --
    the opposite failure mode from the structural-column checks above.
    If a future edit of this file accidentally asserted "no nulls
    anywhere," this test would fail, since these columns are already
    documented (Stage 2/3/5) to carry real, legitimate missingness."""
    from src.preprocessing import load_partition_df

    df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", KNOWN_LEGITIMATELY_MISSING_COLUMNS)
    for col in KNOWN_LEGITIMATELY_MISSING_COLUMNS:
        assert df[col].isna().any(), (
            f"expected column '{col}' to have real, legitimate missing values in the full train "
            f"partition (per Stage 2/3/5's own findings) -- found none, which would mean either the "
            f"data changed or this column was selected incorrectly"
        )
