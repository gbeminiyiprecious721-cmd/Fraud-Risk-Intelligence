"""Tests for src/feature_engineering.py.

Centerpiece of this file: synthetic fixtures deliberately constructed to
EXPOSE each failure mode item 10 of the Stage 6 approval named explicitly —
future-information leakage, equal-timestamp mishandling, and train/
validation boundary violations — not just fixtures that happen to pass.

Run: .venv/bin/pytest tests/test_feature_engineering.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import (  # noqa: E402
    add_entity_historical_features,
    apply_frequency_encoder,
    build_collapsed_indicator_pipeline,
    build_composite_entity_key,
    build_entity_history_train_and_validation,
    collapsed_indicator_representatives,
    compare_entity_key_candidates,
    compute_interaction_and_amount_features,
    compute_temporal_phase_features,
    describe_entity_key_structure,
    find_duplicate_indicator_groups,
    fit_frequency_encoder,
)
from src.leakage_boundaries import FinalTestAccessError  # noqa: E402


# --- compute_temporal_phase_features: relative phase, not calendar time -----

def test_temporal_phase_features_are_pure_modular_arithmetic() -> None:
    df = pd.DataFrame({"TransactionDT": [0, 3600, 86400, 86400 * 8 + 3600 * 5]})
    out = compute_temporal_phase_features(df)
    # 0s -> hour phase 0, dow phase 0
    # 3600s (1h) -> hour phase 1, dow phase 0
    # 86400s (1 day) -> hour phase 0, dow phase 1
    # 8 days + 5h -> hour phase 5, dow phase (8 % 7) = 1
    assert out["txn_hour_phase"].tolist() == [0, 1, 0, 5]
    assert out["txn_dow_phase"].tolist() == [0, 0, 1, 1]


# --- add_entity_historical_features: the core causal-correctness tests -----

def test_prior_count_is_zero_for_a_rows_own_earliest_transaction_even_with_future_rows_present() -> None:
    """Exposes future-information leakage: the earliest row for an entity
    must show prior_count=0 regardless of how many LATER transactions by
    the same entity exist in the data passed in."""
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 20, 30],
            "entity": ["A", "A", "A"],
        }
    )
    out = add_entity_historical_features(df, "entity")
    row_dt10 = out.loc[out["TransactionDT"] == 10].iloc[0]
    assert row_dt10["entity_prior_txn_count"] == 0
    assert pd.isna(row_dt10["entity_time_since_prior_txn"])

    row_dt30 = out.loc[out["TransactionDT"] == 30].iloc[0]
    assert row_dt30["entity_prior_txn_count"] == 2
    assert row_dt30["entity_time_since_prior_txn"] == 10  # 30 - 20


def test_equal_timestamps_never_count_each_other_as_prior() -> None:
    """Exposes equal-timestamp mishandling: two (or three) rows for the
    same entity sharing the EXACT same TransactionDT must not credit each
    other as 'prior' -- they are not actually prior in time."""
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [50, 50, 50, 100],
            "entity": ["B", "B", "B", "B"],
        }
    )
    out = add_entity_historical_features(df, "entity")
    tied = out.loc[out["TransactionDT"] == 50]
    assert (tied["entity_prior_txn_count"] == 0).all()
    assert tied["entity_time_since_prior_txn"].isna().all()

    later = out.loc[out["TransactionDT"] == 100].iloc[0]
    assert later["entity_prior_txn_count"] == 3  # all 3 tied rows ARE strictly prior to this one
    assert later["entity_time_since_prior_txn"] == 50  # 100 - 50 (the tied group's shared timestamp)


def test_a_single_row_entity_never_contributes_to_its_own_feature() -> None:
    df = pd.DataFrame({"TransactionID": [1], "TransactionDT": [999], "entity": ["C"]})
    out = add_entity_historical_features(df, "entity")
    assert out.iloc[0]["entity_prior_txn_count"] == 0
    assert pd.isna(out.iloc[0]["entity_time_since_prior_txn"])


def test_historical_features_are_independent_per_entity() -> None:
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 20, 15],
            "entity": ["A", "A", "Z"],  # Z's single row must not see A's history
        }
    )
    out = add_entity_historical_features(df, "entity")
    z_row = out.loc[out["entity"] == "Z"].iloc[0]
    assert z_row["entity_prior_txn_count"] == 0


# --- build_entity_history_train_and_validation: partition wiring -----------

def _write_entity_fixture(tmp_path, rows):
    """rows: list of (TransactionID, TransactionDT, entity, partition)."""
    assignment = pd.DataFrame(
        [{"TransactionID": r[0], "TransactionDT": r[1], "isFraud": 0, "partition": r[3]} for r in rows]
    )
    transaction = pd.DataFrame([{"TransactionID": r[0], "TransactionDT": r[1], "entity": r[2]} for r in rows])
    assignment_path = tmp_path / "assignment.csv"
    transaction_path = tmp_path / "transaction.csv"
    assignment.to_csv(assignment_path, index=False)
    transaction.to_csv(transaction_path, index=False)
    return assignment_path, transaction_path


def test_entity_history_never_includes_final_test_rows(tmp_path) -> None:
    rows = [
        (1, 10, "A", "train"),
        (2, 20, "A", "validation"),
        (3, 30, "A", "final_test"),  # must never appear in the output
    ]
    assignment_path, transaction_path = _write_entity_fixture(tmp_path, rows)
    out = build_entity_history_train_and_validation(assignment_path, transaction_path, "entity")
    assert set(out["TransactionID"]) == {1, 2}
    assert 3 not in set(out["TransactionID"])


def test_train_row_history_is_unaffected_by_a_later_validation_row(tmp_path) -> None:
    """Exposes a train/validation boundary violation: a TRAIN row's
    historical feature must be identical whether or not a chronologically
    LATER validation row for the same entity is present in the computation."""
    rows = [
        (1, 5, "A", "train"),
        (2, 15, "A", "validation"),
    ]
    assignment_path, transaction_path = _write_entity_fixture(tmp_path, rows)
    out = build_entity_history_train_and_validation(assignment_path, transaction_path, "entity")

    train_row = out.loc[out["TransactionID"] == 1].iloc[0]
    assert train_row["entity_prior_txn_count"] == 0  # unaffected by the later validation row

    validation_row = out.loc[out["TransactionID"] == 2].iloc[0]
    assert validation_row["entity_prior_txn_count"] == 1  # correctly sees the earlier TRAIN row
    assert validation_row["entity_time_since_prior_txn"] == 10


def test_train_validation_consistency_check_train_rows_identical_with_or_without_validation_present(tmp_path) -> None:
    """Deliverable: train/validation transformation consistency check.
    A train row's feature value, computed via the combined train+validation
    entry point, must equal the value computed from train rows ALONE --
    proving validation's presence never perturbs train's own features.

    Timestamps here respect the real, approved Stage 4 invariant: every
    `train` row's TransactionDT is strictly less than every `validation`
    row's (the chronological split puts ALL of train before ALL of
    validation). That invariant is exactly why this consistency property
    holds; it is not a general guarantee for arbitrary interleaved
    timestamps (see this file's future-leakage tests above for what IS
    guaranteed unconditionally: a row's feature never depends on a
    strictly-later timestamp, regardless of partition)."""
    rows = [
        (1, 5, "A", "train"),
        (2, 10, "A", "train"),
        (3, 15, "A", "validation"),
    ]
    assignment_path, transaction_path = _write_entity_fixture(tmp_path, rows)
    combined = build_entity_history_train_and_validation(assignment_path, transaction_path, "entity")

    train_only_df = pd.read_csv(transaction_path).loc[lambda d: d["TransactionID"].isin([1, 2])]
    train_only = add_entity_historical_features(train_only_df, "entity")

    for txn_id in (1, 2):
        combined_row = combined.loc[combined["TransactionID"] == txn_id].iloc[0]
        solo_row = train_only.loc[train_only["TransactionID"] == txn_id].iloc[0]
        assert combined_row["entity_prior_txn_count"] == solo_row["entity_prior_txn_count"]
        assert combined_row["entity_time_since_prior_txn"] == pytest.approx(
            solo_row["entity_time_since_prior_txn"], nan_ok=True
        )


# --- Frequency encoding: train-only fit, unseen-category safety ------------

def test_frequency_encoder_fits_on_train_counts_only() -> None:
    train_df = pd.DataFrame({"domain": ["gmail.com", "gmail.com", "yahoo.com"]})
    mapping = fit_frequency_encoder(train_df, "domain")
    assert mapping["gmail.com"] == 2
    assert mapping["yahoo.com"] == 1


def test_unseen_category_at_apply_time_maps_to_explicit_unseen_value_not_crash_or_nan() -> None:
    train_df = pd.DataFrame({"domain": ["gmail.com", "gmail.com", "yahoo.com"]})
    mapping = fit_frequency_encoder(train_df, "domain")

    validation_series = pd.Series(["gmail.com", "protonmail.com"])  # "protonmail.com" never in train
    encoded = apply_frequency_encoder(validation_series, mapping)  # must not raise
    assert encoded.tolist() == [2, 0]
    assert encoded.isna().sum() == 0


# --- Interaction / amount-derived features ----------------------------------

def test_email_domain_match_flag_distinguishes_match_mismatch_and_unknown() -> None:
    df = pd.DataFrame(
        {
            "P_emaildomain": ["gmail.com", "gmail.com", "gmail.com", np.nan],
            "R_emaildomain": ["gmail.com", "yahoo.com", np.nan, np.nan],
            "ProductCD": ["W", "W", "W", "W"],
            "card4": ["visa", "visa", "visa", "visa"],
            "TransactionAmt": [10.0, 10.0, 10.0, 10.0],
        }
    )
    out = compute_interaction_and_amount_features(df)
    assert out["email_domain_match"].tolist()[0] == 1.0
    assert out["email_domain_match"].tolist()[1] == 0.0
    assert pd.isna(out["email_domain_match"].tolist()[2])  # one side missing -> unknown, not mismatch
    assert pd.isna(out["email_domain_match"].tolist()[3])  # both missing -> unknown


def test_round_amount_flag_and_log1p() -> None:
    df = pd.DataFrame(
        {
            "P_emaildomain": [np.nan, np.nan],
            "R_emaildomain": [np.nan, np.nan],
            "ProductCD": ["W", "W"],
            "card4": ["visa", "visa"],
            "TransactionAmt": [50.0, 49.99],
        }
    )
    out = compute_interaction_and_amount_features(df)
    assert out["transaction_amt_is_round"].tolist() == [1, 0]
    assert out["log1p_transaction_amt"].iloc[0] == pytest.approx(np.log1p(50.0))


# --- Duplicate-indicator-group detection + collapsed prototype pipeline ----

def test_find_duplicate_indicator_groups_detects_exact_duplicates_and_singletons() -> None:
    train_df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [1.0, np.nan, 3.0, np.nan],  # identical missingness mask to "a"
            "c": [np.nan, np.nan, 3.0, 4.0],  # different mask
        }
    )
    groups = find_duplicate_indicator_groups(train_df, ["a", "b", "c"])
    reps = collapsed_indicator_representatives(groups)
    assert len(reps) == 2  # one group for {a,b}, one singleton for {c}
    # whichever column became representative for the a/b group, it must list both members
    ab_group = [v for v in groups.values() if set(v) == {"a", "b"}]
    assert len(ab_group) == 1


def test_collapsed_indicator_pipeline_preserves_all_values_with_fewer_indicator_columns() -> None:
    train_df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [10.0, np.nan, 30.0, np.nan],  # same missingness pattern as "a", different values
            "c": [np.nan, np.nan, 3.0, 4.0],
        }
    )
    groups = find_duplicate_indicator_groups(train_df, ["a", "b", "c"])
    reps = collapsed_indicator_representatives(groups)
    ct = build_collapsed_indicator_pipeline(["a", "b", "c"], reps)
    output = ct.fit_transform(train_df)
    dense = output.toarray() if hasattr(output, "toarray") else output

    assert not np.isnan(dense).any()
    # 3 value columns + len(reps) indicator columns (2: one for the a/b pattern, one for c)
    assert dense.shape[1] == 3 + len(reps)
    assert len(reps) == 2


# --- Stage 6 review round, item 4: mutation-invariance, not just existence --

def test_earlier_rows_feature_value_is_invariant_to_mutating_a_later_rows_timestamp() -> None:
    """Stronger than 'the earliest row sees prior_count=0 regardless of
    later rows being present' (already tested above): this proves an
    EARLIER row's historical feature value does not change AT ALL when a
    LATER row for the same entity is mutated -- moved further into the
    future, or given a different timestamp entirely. If the algorithm ever
    accidentally looked ahead, perturbing a later row would perturb an
    earlier row's result; it must not."""
    base = pd.DataFrame(
        {"TransactionID": [1, 2, 3], "TransactionDT": [10, 20, 30], "entity": ["A", "A", "A"]}
    )
    mutated = base.copy()
    mutated.loc[mutated["TransactionID"] == 3, "TransactionDT"] = 500  # push the later row much further out

    out_base = add_entity_historical_features(base, "entity")
    out_mutated = add_entity_historical_features(mutated, "entity")

    for txn_id in (1, 2):
        row_base = out_base.loc[out_base["TransactionID"] == txn_id].iloc[0]
        row_mutated = out_mutated.loc[out_mutated["TransactionID"] == txn_id].iloc[0]
        assert row_base["entity_prior_txn_count"] == row_mutated["entity_prior_txn_count"]
        assert row_base["entity_time_since_prior_txn"] == pytest.approx(
            row_mutated["entity_time_since_prior_txn"], nan_ok=True
        )


def test_earlier_rows_feature_value_is_invariant_to_mutating_a_tied_later_row() -> None:
    """Same property as above, specifically for the equal-timestamp case:
    mutating ONE member of a later tied-timestamp group (changing its
    TransactionDT away from the tie) must not change an EARLIER row's
    historical feature, and must not change the OTHER tied row's feature
    either (it was never "prior" to begin with)."""
    base = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [10, 50, 50, 50],  # rows 2,3,4 tied at t=50
            "entity": ["A", "A", "A", "A"],
        }
    )
    mutated = base.copy()
    mutated.loc[mutated["TransactionID"] == 4, "TransactionDT"] = 999  # break the tie for just one row

    out_base = add_entity_historical_features(base, "entity")
    out_mutated = add_entity_historical_features(mutated, "entity")

    for txn_id in (1, 2, 3):  # none of these should be affected by row 4's mutation
        row_base = out_base.loc[out_base["TransactionID"] == txn_id].iloc[0]
        row_mutated = out_mutated.loc[out_mutated["TransactionID"] == txn_id].iloc[0]
        assert row_base["entity_prior_txn_count"] == row_mutated["entity_prior_txn_count"]
        assert row_base["entity_time_since_prior_txn"] == pytest.approx(
            row_mutated["entity_time_since_prior_txn"], nan_ok=True
        )


def test_real_data_validation_rows_are_mutation_invariant_to_a_later_validation_rows_timestamp(tmp_path) -> None:
    """Real-data-flavored version using the leakage-safe entry point: two
    validation rows for the same entity; mutating the LATER one's timestamp
    (pushing it further out) must not change the EARLIER validation row's
    historical feature."""
    rows = [
        (1, 5, "A", "train"),
        (2, 15, "A", "validation"),
        (3, 25, "A", "validation"),
    ]
    assignment_path, transaction_path = _write_entity_fixture(tmp_path, rows)
    base = build_entity_history_train_and_validation(assignment_path, transaction_path, "entity")

    # Mutate row 3's timestamp in the transaction file and recompute.
    txn_df = pd.read_csv(transaction_path)
    txn_df.loc[txn_df["TransactionID"] == 3, "TransactionDT"] = 9999
    txn_df.to_csv(transaction_path, index=False)
    mutated = build_entity_history_train_and_validation(assignment_path, transaction_path, "entity")

    for txn_id in (1, 2):
        base_row = base.loc[base["TransactionID"] == txn_id].iloc[0]
        mutated_row = mutated.loc[mutated["TransactionID"] == txn_id].iloc[0]
        assert base_row["entity_prior_txn_count"] == mutated_row["entity_prior_txn_count"]


# --- Stage 6 review round, item 6: frequency encoding determinism ----------

def test_frequency_encoding_is_deterministic_across_repeated_calls() -> None:
    train_df = pd.DataFrame({"domain": ["a", "a", "b", "c", "c", "c"]})
    mapping1 = fit_frequency_encoder(train_df, "domain")
    mapping2 = fit_frequency_encoder(train_df, "domain")
    assert mapping1 == mapping2

    series = pd.Series(["a", "b", "c", "unseen"])
    encoded1 = apply_frequency_encoder(series, mapping1)
    encoded2 = apply_frequency_encoder(series, mapping2)
    pd.testing.assert_series_equal(encoded1, encoded2)
    # Re-applying to a shuffled copy of the same series gives the same per-value result
    shuffled = series.sample(frac=1, random_state=0)
    encoded_shuffled = apply_frequency_encoder(shuffled, mapping1)
    assert dict(zip(shuffled, encoded_shuffled)) == dict(zip(series, encoded1))


# --- Stage 6 review round, items 1/2: entity-key structural comparison -----

def test_card1_alone_shows_high_repeat_concentration_not_a_clean_identity_key() -> None:
    """Review item 1: card1 must not be assumed a unique customer/card key.
    This constructs a case where a SINGLE card1 value is shared by rows
    that are otherwise clearly different entities (different addr1) --
    structurally demonstrating card1 alone cannot distinguish them,
    without reading any target column or training any model."""
    df = pd.DataFrame(
        {
            "card1": [111, 111, 111, 222],
            "addr1": [10.0, 20.0, 30.0, 40.0],  # same card1, three different addr1 values
        }
    )
    card1_only = describe_entity_key_structure(df, df["card1"].astype("string"))
    composite = describe_entity_key_structure(df, build_composite_entity_key(df, ["card1", "addr1"]))

    assert card1_only["n_distinct_entities"] == 2  # card1 alone can't separate the three addr1 variants
    assert composite["n_distinct_entities"] == 4  # the composite key does


def test_compare_entity_key_candidates_reads_no_target_column() -> None:
    df = pd.DataFrame(
        {
            "card1": [1, 1, 2, 2, 2],
            "card2": [10.0, 10.0, 20.0, 21.0, 20.0],
            "addr1": [100.0, 100.0, 200.0, 200.0, 200.0],
            "isFraud": [0, 1, 0, 1, 0],  # present in the frame, must never be touched
        }
    )
    results = compare_entity_key_candidates(df, {"card1": ["card1"], "card1+card2+addr1": ["card1", "card2", "addr1"]})
    assert set(results.keys()) == {"card1", "card1+card2+addr1"}
    assert results["card1"]["n_distinct_entities"] == 2
    assert results["card1+card2+addr1"]["n_distinct_entities"] == 3

    # Behavioral proof the comparison does not use the target: permuting
    # isFraud's values must not change the structural result at all.
    df_permuted = df.copy()
    df_permuted["isFraud"] = df_permuted["isFraud"].sample(frac=1, random_state=1).to_numpy()
    results_permuted = compare_entity_key_candidates(
        df_permuted, {"card1": ["card1"], "card1+card2+addr1": ["card1", "card2", "addr1"]}
    )
    assert results == results_permuted
