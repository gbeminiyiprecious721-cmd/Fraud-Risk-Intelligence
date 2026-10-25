"""Tests for src/leakage_boundaries.py — split integrity, determinism, and the register builder.

Scoped to functions testable with synthetic fixtures (no real-data I/O needed
for these); the real-data empirical checks (check_remaildomain_productcd_relationship,
check_d_columns_sanity) and the real chronological_split are exercised directly
against the dataset in scripts/run_stage4_analysis.py, same pattern as Stages 2-3.

Run: .venv/bin/pytest tests/test_leakage_boundaries.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import (  # noqa: E402
    FamilyDecision,
    FinalTestAccessError,
    assert_no_final_test_contamination,
    build_feature_availability_register,
    check_cutoff_tie_handling,
    chronological_split,
    classify_family,
    get_partition,
    verify_split,
)


@pytest.fixture
def synthetic_transaction_csv(tmp_path: Path) -> Path:
    """100 rows, TransactionDT 0..9900 step 100, isFraud alternating-ish, all IDs unique."""
    n = 100
    df = pd.DataFrame(
        {
            "TransactionID": range(1000, 1000 + n),
            "TransactionDT": [i * 100 for i in range(n)],
            "isFraud": [1 if i % 7 == 0 else 0 for i in range(n)],  # ~14% fraud, every partition gets some
        }
    )
    path = tmp_path / "synthetic_transaction.csv"
    df.to_csv(path, index=False)
    return path


def test_chronological_split_is_deterministic(synthetic_transaction_csv: Path) -> None:
    assignment_1, cutoffs_1 = chronological_split(synthetic_transaction_csv)
    assignment_2, cutoffs_2 = chronological_split(synthetic_transaction_csv)
    pd.testing.assert_frame_equal(assignment_1, assignment_2)
    assert cutoffs_1 == cutoffs_2


def test_chronological_split_respects_ordering(synthetic_transaction_csv: Path) -> None:
    assignment, _ = chronological_split(synthetic_transaction_csv)
    train_max = assignment.loc[assignment["partition"] == "train", "TransactionDT"].max()
    val_min = assignment.loc[assignment["partition"] == "validation", "TransactionDT"].min()
    val_max = assignment.loc[assignment["partition"] == "validation", "TransactionDT"].max()
    test_min = assignment.loc[assignment["partition"] == "final_test", "TransactionDT"].min()
    assert train_max < val_min
    assert val_max < test_min


def test_chronological_split_proportions_approximate_requested_fractions(synthetic_transaction_csv: Path) -> None:
    assignment, _ = chronological_split(synthetic_transaction_csv, train_frac=0.70, validation_frac=0.15)
    counts = assignment["partition"].value_counts(normalize=True)
    # Time-proportional, not row-count-proportional, so allow tolerance — but should be in the right ballpark.
    assert 0.55 < counts["train"] < 0.85
    assert counts["final_test"] > 0


def test_verify_split_detects_clean_split(synthetic_transaction_csv: Path) -> None:
    assignment, _ = chronological_split(synthetic_transaction_csv)
    result = verify_split(assignment)
    assert result["chronological_ordering_ok"] is True
    assert result["id_disjoint"] is True
    assert result["id_overlap_count"] == 0
    assert result["total_accounted_for"] is True


def test_verify_split_detects_id_overlap() -> None:
    # Manually construct a BROKEN assignment (same TransactionID in two partitions) and confirm detection.
    broken = pd.DataFrame(
        {
            "TransactionID": [1, 1, 2, 3],
            "TransactionDT": [10, 20, 30, 40],
            "isFraud": [0, 1, 0, 1],
            "partition": ["train", "validation", "validation", "final_test"],  # ID 1 in both train and validation
        }
    )
    result = verify_split(broken)
    assert result["id_disjoint"] is False
    assert result["id_overlap_count"] == 1


def test_verify_split_detects_ordering_violation() -> None:
    # A validation row with an earlier TransactionDT than a train row — should fail ordering.
    broken = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [100, 200, 50, 300, 400, 500],  # row 3 (val) is earlier than rows in train
            "isFraud": [0, 1, 0, 1, 0, 1],
            "partition": ["train", "train", "validation", "validation", "final_test", "final_test"],
        }
    )
    result = verify_split(broken)
    assert result["chronological_ordering_ok"] is False


@pytest.mark.parametrize(
    "column, expected_family",
    [
        ("TransactionID", "identifier"),
        ("isFraud", "target"),
        ("TransactionDT", "transaction_time"),
        ("TransactionAmt", "transaction_amount"),
        ("ProductCD", "product"),
        ("card4", "card"),
        ("addr1", "address"),
        ("dist2", "distance"),
        ("P_emaildomain", "purchaser_email"),
        ("R_emaildomain", "receiver_email"),
        ("C7", "C_count"),
        ("D11", "D_timedelta"),
        ("M3", "M_match"),
        ("V250", "V_engineered"),
        ("DeviceType", "identity_device"),
        ("id_14", "identity_id"),
        ("id-14", "identity_id"),
    ],
)
def test_classify_family(column: str, expected_family: str) -> None:
    assert classify_family(column) == expected_family


def test_build_feature_availability_register_covers_every_column() -> None:
    transaction_schema = {"TransactionID": "int64", "isFraud": "int64", "ProductCD": "str", "V1": "float64"}
    identity_schema = {"TransactionID": "int64", "id_01": "float64", "DeviceType": "str"}
    decisions = {
        "identifier": FamilyDecision("exclude_structural", "row identifier", "documented_primary"),
        "target": FamilyDecision("exclude_structural", "the label itself", "documented_primary"),
        "product": FamilyDecision("include", "transaction-time attribute", "domain_logic"),
        "V_engineered": FamilyDecision("include_with_caveat", "documented family, per-column unverified", "documented_secondary"),
        "identity_id": FamilyDecision("exclude_conservative", "mechanism unresolved", "absence_of_evidence"),
        "identity_device": FamilyDecision("exclude_conservative", "mechanism unresolved", "absence_of_evidence"),
    }
    register = build_feature_availability_register(transaction_schema, identity_schema, decisions)
    assert len(register) == len(transaction_schema) + len(identity_schema)
    assert set(register["column"]) >= {"ProductCD", "V1", "id_01", "DeviceType"}
    assert register.loc[register["column"] == "V1", "status"].iloc[0] == "include_with_caveat"


def test_build_feature_availability_register_raises_on_missing_family_decision() -> None:
    transaction_schema = {"Mystery123": "float64"}  # classify_family -> "unclassified", not in decisions
    with pytest.raises(KeyError):
        build_feature_availability_register(transaction_schema, {}, {})


# --- Final-holdout enforcement mechanism (item 3 of the review round) -----

@pytest.fixture
def sample_assignment() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [10, 20, 30, 40, 50, 60],
            "isFraud": [0, 1, 0, 1, 0, 1],
            "partition": ["train", "train", "validation", "validation", "final_test", "final_test"],
        }
    )


def test_get_partition_refuses_final_test_by_default(sample_assignment: pd.DataFrame) -> None:
    with pytest.raises(FinalTestAccessError):
        get_partition(sample_assignment, ["train", "final_test"])


def test_get_partition_allows_final_test_when_explicitly_requested(sample_assignment: pd.DataFrame) -> None:
    result = get_partition(sample_assignment, ["final_test"], allow_final_test=True)
    assert set(result["TransactionID"]) == {5, 6}


def test_get_partition_non_final_test_requests_unaffected(sample_assignment: pd.DataFrame) -> None:
    result = get_partition(sample_assignment, ["train", "validation"])
    assert set(result["TransactionID"]) == {1, 2, 3, 4}


def test_assert_no_final_test_contamination_passes_when_clean(sample_assignment: pd.DataFrame) -> None:
    assert_no_final_test_contamination(used_ids=[1, 2, 3, 4], assignment=sample_assignment)  # should not raise


def test_assert_no_final_test_contamination_raises_when_dirty(sample_assignment: pd.DataFrame) -> None:
    with pytest.raises(FinalTestAccessError):
        assert_no_final_test_contamination(used_ids=[1, 2, 5], assignment=sample_assignment)  # 5 is final_test


# --- Cutoff tie-handling (item 4 of the review round) ----------------------

def test_check_cutoff_tie_handling_reports_no_split_ties_on_clean_data(synthetic_transaction_csv: Path) -> None:
    assignment, cutoffs = chronological_split(synthetic_transaction_csv)
    result = check_cutoff_tie_handling(assignment, cutoffs)
    assert result["train_end"]["all_on_same_side"] is True
    assert result["validation_end"]["all_on_same_side"] is True


def test_check_cutoff_tie_handling_detects_a_real_tie(tmp_path: Path) -> None:
    # Construct data where multiple rows share the exact TransactionDT that will become train_end.
    df = pd.DataFrame(
        {
            "TransactionID": range(10),
            "TransactionDT": [0, 0, 10, 10, 20, 20, 30, 30, 40, 40],
            "isFraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    path = tmp_path / "tie_fixture.csv"
    df.to_csv(path, index=False)
    assignment, cutoffs = chronological_split(path, train_frac=0.5, validation_frac=0.25)
    result = check_cutoff_tie_handling(assignment, cutoffs)
    # By construction (dt <= cutoff -> earlier partition), every tie must land on one side.
    assert result["train_end"]["all_on_same_side"] is True
    assert result["validation_end"]["all_on_same_side"] is True
