"""Stage 2 validation driver — runs src/data_validation.py against the real
IEEE-CIS files and prints a structured report.

Run with the project's existing .venv, e.g.:
    .venv/bin/python3 scripts/run_stage2_validation.py

Read-only: never writes to data/raw/. Memory-conscious: never loads more
than one file's relevant columns into memory at a time, and never loads all
five files' full contents simultaneously.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_validation import (  # noqa: E402
    check_file_integrity,
    check_id_uniqueness,
    check_join_coverage,
    check_target_presence,
    check_transactiondt_ordering,
    classify_columns,
    compare_dt_ranges,
    compute_missingness,
    flag_potential_leakage_columns,
    get_schema_sample,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ieee-fraud-detection"
FILES = {
    "train_transaction": DATA_DIR / "train_transaction.csv",
    "train_identity": DATA_DIR / "train_identity.csv",
    "test_transaction": DATA_DIR / "test_transaction.csv",
    "test_identity": DATA_DIR / "test_identity.csv",
    "sample_submission": DATA_DIR / "sample_submission.csv",
}


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {}
    t0 = time.time()

    section("1. File integrity + row counts")
    integrity = {}
    for name, path in FILES.items():
        result = check_file_integrity(path)
        integrity[name] = result.__dict__
        print(f"{name}: exists={result.exists} readable={result.readable} "
              f"rows={result.row_count} size_mb={result.size_bytes / 1e6:.1f} error={result.error}")
    report["file_integrity"] = integrity

    section("2. Schema samples + target presence")
    schemas = {}
    target_presence = {}
    for name, path in FILES.items():
        schema = get_schema_sample(path, nrows=5000)
        schemas[name] = schema
        has_target = check_target_presence(path, "isFraud")
        target_presence[name] = has_target
        print(f"{name}: {len(schema)} columns (sampled), isFraud present={has_target}")
    report["schemas_sampled"] = {k: len(v) for k, v in schemas.items()}
    report["target_presence"] = target_presence

    section("3. Column classification (train_transaction, train_identity)")
    classification = {}
    for name in ("train_transaction", "train_identity"):
        cls = classify_columns(schemas[name])
        classification[name] = {k: len(v) for k, v in cls.items()}
        print(f"{name}: numeric={len(cls['numeric'])} categorical={len(cls['categorical'])} other={len(cls['other'])}")
    report["column_classification"] = classification

    section("4. TransactionID uniqueness")
    id_uniqueness = {}
    for name in ("train_transaction", "test_transaction", "train_identity", "test_identity"):
        result = check_id_uniqueness(FILES[name], "TransactionID")
        id_uniqueness[name] = result
        print(f"{name}: {result}")
    report["id_uniqueness"] = id_uniqueness

    section("5. Transaction <-> Identity join coverage")
    train_join = check_join_coverage(FILES["train_transaction"], FILES["train_identity"])
    test_join = check_join_coverage(FILES["test_transaction"], FILES["test_identity"])
    print(f"train: {train_join}")
    print(f"test:  {test_join}")
    report["join_coverage"] = {"train": train_join, "test": test_join}

    section("6. TransactionDT ordering + train/test range comparison")
    dt_compare = compare_dt_ranges(FILES["train_transaction"], FILES["test_transaction"])
    print(f"train/test DT comparison: {dt_compare}")
    report["dt_range_comparison"] = dt_compare

    section("7. Missingness — train_identity (small file, full scan)")
    identity_missingness = compute_missingness(FILES["train_identity"])
    top_missing_identity = identity_missingness.head(10).to_dict()
    print(f"Top-10 missing % (train_identity):\n{identity_missingness.head(10)}")
    report["missingness_train_identity_top10"] = top_missing_identity
    report["missingness_train_identity_mean_pct"] = round(float(identity_missingness.mean()), 2)

    section("8. Missingness — train_transaction (chunked scan, large file)")
    txn_missingness = compute_missingness(FILES["train_transaction"])
    top_missing_txn = txn_missingness.head(15).to_dict()
    print(f"Top-15 missing % (train_transaction):\n{txn_missingness.head(15)}")
    report["missingness_train_transaction_top15"] = top_missing_txn
    report["missingness_train_transaction_mean_pct"] = round(float(txn_missingness.mean()), 2)
    report["missingness_train_transaction_zero_missing_cols"] = int((txn_missingness == 0).sum())

    section("9. Potential leakage / feature-availability flags")
    leakage = flag_potential_leakage_columns(schemas["train_transaction"].keys())
    print(f"Excluded always: {leakage['exclude_from_features']}")
    print(f"Unverifiable from schema alone: {len(leakage['unverifiable_from_schema_alone'])} anonymized columns (V/C/D)")
    report["leakage_flags"] = {
        "exclude_from_features": leakage["exclude_from_features"],
        "n_unverifiable_anonymized_columns": len(leakage["unverifiable_from_schema_alone"]),
    }

    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 1)
    section(f"Done in {elapsed:.1f}s")

    out_path = Path(__file__).resolve().parent.parent / "docs" / "_stage2_validation_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")

    return report


if __name__ == "__main__":
    main()
