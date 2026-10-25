"""Stage 4 driver — runs empirical leakage checks, builds the feature-availability
register, builds and verifies the chronological split, against the real data.

Run: .venv/bin/python3 scripts/run_stage4_analysis.py

Read-only against data/raw/. Writes only to docs/ (register CSV, split
assignment CSV, raw JSON summary) — never touches the original CSVs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_validation import get_schema_sample  # noqa: E402
from src.leakage_boundaries import (  # noqa: E402
    FamilyDecision,
    FinalTestAccessError,
    assert_no_final_test_contamination,
    build_feature_availability_register,
    check_cutoff_tie_handling,
    check_d_columns_sanity,
    check_remaildomain_productcd_relationship,
    chronological_split,
    classify_family,
    get_partition,
    verify_split,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
TRAIN_ID = DATA_DIR / "train_identity.csv"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

D_COLUMNS = [f"D{i}" for i in range(1, 16)]


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {}

    section("1. R_emaildomain vs. ProductCD (does presence track a legitimate product attribute?)")
    by_product = check_remaildomain_productcd_relationship(TRAIN_TXN)
    print(by_product)
    report["remaildomain_by_productcd"] = by_product.to_dict(orient="index")

    section("2. D-column sanity (negativity + correlation with TransactionDT)")
    d_sanity = check_d_columns_sanity(TRAIN_TXN, D_COLUMNS)
    print(d_sanity)
    report["d_column_sanity"] = d_sanity.to_dict(orient="index")

    section("3. Chronological split (train/validation/final_test, time-proportional, no gap)")
    assignment, cutoffs = chronological_split(TRAIN_TXN, train_frac=0.70, validation_frac=0.15, gap_seconds=0)
    print(f"Cutoffs: train_end_dt={cutoffs.train_end_dt}  validation_end_dt={cutoffs.validation_end_dt}  gap={cutoffs.gap_seconds}")
    report["split_cutoffs"] = cutoffs.__dict__

    section("4. Split verification")
    verification = verify_split(assignment)
    for key, value in verification.items():
        print(f"{key}: {value}")
    report["split_verification"] = verification

    section("4b. Cutoff tie-handling (review item 4)")
    tie_results = check_cutoff_tie_handling(assignment, cutoffs)
    for key, value in tie_results.items():
        print(f"{key}: {value}")
    report["cutoff_tie_handling"] = tie_results

    section("4c. Final-holdout enforcement mechanism, demonstrated against real data (review item 3)")
    try:
        get_partition(assignment, ["train", "final_test"])
        enforcement_demo_blocked = False
    except FinalTestAccessError as exc:
        enforcement_demo_blocked = True
        print(f"get_partition() correctly refused: {exc}")
    allowed = get_partition(assignment, ["final_test"], allow_final_test=True)
    print(f"get_partition(allow_final_test=True) correctly returned {len(allowed)} rows.")
    try:
        assert_no_final_test_contamination(used_ids=assignment.loc[assignment["partition"] == "train", "TransactionID"], assignment=assignment)
        contamination_check_passed_on_clean_ids = True
    except FinalTestAccessError:
        contamination_check_passed_on_clean_ids = False
    report["final_test_enforcement_demo"] = {
        "unguarded_request_was_blocked": enforcement_demo_blocked,
        "explicit_override_returned_rows": len(allowed),
        "contamination_check_passed_on_clean_train_ids": contamination_check_passed_on_clean_ids,
    }
    print(f"Enforcement demo: {report['final_test_enforcement_demo']}")

    section("5. Feature-availability register")
    txn_schema = get_schema_sample(TRAIN_TXN, nrows=5000)
    id_schema = get_schema_sample(TRAIN_ID, nrows=5000)

    decisions = {
        "identifier": FamilyDecision(
            "exclude_structural",
            "Row identifier; no predictive meaning.",
            "documented_primary",
        ),
        "target": FamilyDecision(
            "exclude_structural",
            "The label itself.",
            "documented_primary",
        ),
        "transaction_time": FamilyDecision(
            "include",
            "Current time is always known at the moment a prediction is made; used for the split, and derivable relative-time features are legitimate.",
            "domain_logic",
        ),
        "transaction_amount": FamilyDecision(
            "include",
            "Known at authorization time by definition (the amount being charged). Stage 3 found it a weak standalone signal, not unavailable.",
            "domain_logic",
        ),
        "product": FamilyDecision(
            "include",
            "Checkout-time attribute (product category).",
            "domain_logic",
        ),
        "card": FamilyDecision(
            "include",
            "Card/issuer attributes presented at authorization time.",
            "domain_logic",
        ),
        "address": FamilyDecision(
            "include",
            "Billing/shipping address attributes, known at checkout.",
            "domain_logic",
        ),
        "distance": FamilyDecision(
            "include",
            "Distance-type attribute (e.g. billing-to-shipping), computable at checkout.",
            "domain_logic",
        ),
        "purchaser_email": FamilyDecision(
            "include",
            "Purchaser email domain, known at checkout.",
            "domain_logic",
        ),
        "receiver_email": FamilyDecision(
            "include_with_caveat",
            "REFINED, review round 2: presence is ~0% for ProductCD=W and ~95-100% for all other product codes, which explains Stage 3's missingness-by-fraud-status gap as a structural, product-type-driven pattern rather than one that directly tracks fraud outcome -- this rules out ONE specific failure mode (presence correlating with fraud status because of some post-outcome process). It does NOT independently establish prediction-time availability: the ProductCD relationship is silent on WHEN in the checkout/fulfillment flow R_emaildomain's value is actually captured relative to the moment a risk score would need to be computed, and that timing has not been verified. Status stays include_with_caveat on BOTH grounds now -- the structural-missingness leakage worry is substantially addressed, the availability-timing question is not -- eligible for controlled Stage 7+ experiments, not confirmed safe for production use.",
            "empirical",
        ),
        "C_count": FamilyDecision(
            "include_with_caveat",
            "Documented (secondary sources, not the primary Kaggle page directly) as counting features computed against historical entity data (e.g. number of addresses associated with a card) — causally knowable by description. Not individually verified per column.",
            "documented_secondary",
        ),
        "D_timedelta": FamilyDecision(
            "include_with_caveat",
            "Documented (secondary sources) as time-delta features. Sanity-checked this stage (Section 2): all 15 columns show ~0% negative values and low correlation with TransactionDT. REFINED, review round 2: these two checks are supporting evidence only -- they rule out two specific, narrow failure modes (a value inconsistent with any past-event delta; a value that's really just a disguised absolute-time position) -- they do NOT prove the underlying computation is causal (uses only pre-row information) or safe at prediction time. A column computed from a symmetric window (partly backward- and partly forward-looking) could pass both checks and still be non-causal; this check cannot distinguish that case. A few columns (D4/D6/D11/D12/D14/D15) show a tiny non-zero negative share (<=0.005%), noted as an open minor anomaly, not explained away.",
            "documented_secondary",
        ),
        "M_match": FamilyDecision(
            "include",
            "Documented (secondary sources) as match flags (e.g. name/address match) — knowable at transaction time by domain logic.",
            "documented_secondary",
        ),
        "V_engineered": FamilyDecision(
            "include_with_caveat",
            "Documented (secondary sources, moderate confidence) as ranking/counting/entity-relation features -- by description, plausibly causally computable, not individually confirmed. The 14 columns from Stage 3's missingness-gap finding were spot-checked (2 of 14: V180, V197) and show the same ProductCD-driven structural-missingness pattern as R_emaildomain -- same refinement applies here as to receiver_email: this explains the missingness pattern, it does NOT independently prove prediction-time availability for these 2 (let alone the other 12, inferred not re-verified, or the ~325 remaining V-columns, not checked at all). Included per the explicit instruction not to blanket-exclude anonymized features by name alone -- eligible for controlled Stage 7+ experiments only, not cleared for production use.",
            "documented_secondary",
        ),
        "identity_device": FamilyDecision(
            "exclude_conservative",
            "Identity-table membership itself is the open question (see identity_id below) — device sub-fields inherit the same unresolved timing/mechanism uncertainty.",
            "absence_of_evidence",
        ),
        "identity_id": FamilyDecision(
            "exclude_conservative",
            "Stage 3 found a ~3.75x fraud-rate gap between has-identity and no-identity transactions. Documented fact: not every transaction has a matching identity record (Kaggle data description). NOT documented: why/when identity records are created (automatic-but-sometimes-fails vs. risk-triggered secondary verification) — searched explicitly this stage, found no source addressing it. Per the conservative-exclusion instruction, excluded pending positive evidence of availability-at-prediction-time, not included by default.",
            "absence_of_evidence",
        ),
    }

    register = build_feature_availability_register(txn_schema, id_schema, decisions)
    print(f"Register built: {len(register)} columns.")
    print(register["status"].value_counts())
    report["register_status_counts"] = register["status"].value_counts().to_dict()

    section("5b. Register completeness verification (review item 5)")
    expected_total = len(txn_schema) + len(id_schema)
    valid_statuses = {"include", "include_with_caveat", "exclude_conservative", "exclude_structural"}
    duplicate_rows = register.duplicated(subset=["column", "source_file"])
    unclassified = register.loc[register["family"] == "unclassified"]
    completeness = {
        "expected_total_columns": expected_total,
        "actual_register_rows": len(register),
        "row_count_matches_expected": len(register) == expected_total == 435,
        "all_statuses_valid": bool(register["status"].isin(valid_statuses).all()),
        "duplicate_column_rows": int(duplicate_rows.sum()),
        "unclassified_family_count": len(unclassified),
        "every_row_has_nonempty_rationale": bool((register["rationale"].str.len() > 0).all()),
        "status_counts_sum_to_total": int(register["status"].value_counts().sum()) == len(register),
    }
    for key, value in completeness.items():
        print(f"{key}: {value}")
    report["register_completeness"] = completeness

    register_path = DOCS_DIR / "feature_availability_register.csv"
    register.to_csv(register_path, index=False)
    print(f"Register written to {register_path}")

    assignment_path = DOCS_DIR / "_stage4_split_assignment.csv"
    assignment.to_csv(assignment_path, index=False)
    print(f"Split assignment written to {assignment_path}")

    raw_path = DOCS_DIR / "_stage4_analysis_raw.json"
    with open(raw_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {raw_path}")

    return report


if __name__ == "__main__":
    main()
