"""Stage 9 design-phase requirement 4 — "before any real tuning, perform a
small synthetic end-to-end tracker verification. Test successful
completion, exceptions, failed runs, duplicate run IDs and evidence
traceability."

Simulates a miniature hyperparameter search (2 candidates x 2 folds, one
candidate deliberately failing) using the Stage 8 tracker's new
`search_id`/`candidate_index`/`fold_index` fields, entirely against a
TEMPORARY registry (never `docs/experiment_registry.jsonl`) and entirely
with SYNTHETIC numbers -- no real model is fit, no dataset is read, no
historical Stage 7 script is executed to "exercise" the tracker (per
explicit instruction not to do that).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_tracking import (  # noqa: E402
    DuplicateRunIdError,
    append_run_record,
    build_run_record,
    compute_file_checksum,
    new_run_id,
    query_registry,
    read_registry,
    track_run,
)


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> bool:
    all_passed = True
    with tempfile.TemporaryDirectory() as tmp:
        registry_path = Path(tmp) / "synthetic_search_registry.jsonl"
        search_id = "synthetic-search-" + new_run_id("")[-8:]

        section("1. Simulate a 2-candidate x 2-fold search: candidate 0 succeeds on both folds")
        for fold_index in (0, 1):
            with track_run(
                9, "synthetic_rf_trial", "Synthetic smoke-test CV-fold trial.", registry_path=registry_path,
                search_id=search_id, candidate_index=0, fold_index=fold_index,
            ) as record:
                record["hyperparameters"] = {"n_estimators": 100, "max_depth": 10}
                record["metrics"] = {"average_precision": 0.30 + 0.01 * fold_index}
        candidate_0_records = query_registry(registry_path, search_id=search_id, candidate_index=0)
        ok = len(candidate_0_records) == 2 and all(r["status"] == "completed" for r in candidate_0_records)
        print(f"Candidate 0 (success): {len(candidate_0_records)} fold records, all completed  [{'PASS' if ok else 'FAIL'}]")
        all_passed &= ok

        section("2. Simulate candidate 1 failing on fold 0 (exception), never attempting fold 1")
        try:
            with track_run(
                9, "synthetic_rf_trial", "Synthetic smoke-test CV-fold trial, failure path.", registry_path=registry_path,
                search_id=search_id, candidate_index=1, fold_index=0,
            ) as record:
                record["hyperparameters"] = {"n_estimators": 100, "max_depth": 10}
                raise MemoryError("synthetic simulated OOM during fold fit")
        except MemoryError:
            pass
        candidate_1_records = query_registry(registry_path, search_id=search_id, candidate_index=1)
        ok = (
            len(candidate_1_records) == 1
            and candidate_1_records[0]["status"] == "failed"
            and "synthetic simulated OOM" in candidate_1_records[0]["status_reason"]
        )
        print(f"Candidate 1 (exception on fold 0): status={candidate_1_records[0]['status']!r}, "
              f"status_reason={candidate_1_records[0]['status_reason']!r}  [{'PASS' if ok else 'FAIL'}]")
        print("Fold 1 for candidate 1 correctly has NO record -- an interrupted search does not fabricate the fold it never reached.")
        all_passed &= ok

        section("3. Duplicate run ID is refused (immutability guard)")
        duplicate_refused = False
        existing_run_id = candidate_0_records[0]["run_id"]
        duplicate_record = build_run_record(
            run_id=existing_run_id, workflow_stage=9, experiment_name="duplicate_attempt",
            experiment_purpose="deliberate duplicate for the guard test", status="completed", provenance="directly_verified",
        )
        try:
            append_run_record(duplicate_record, registry_path)
        except DuplicateRunIdError:
            duplicate_refused = True
        print(f"Duplicate run_id refused: {duplicate_refused}  [{'PASS' if duplicate_refused else 'FAIL'}]")
        all_passed &= duplicate_refused

        section("4. Evidence traceability -- a record's cited evidence file checksum is verifiable")
        evidence_file = Path(tmp) / "synthetic_evidence.json"
        evidence_file.write_text('{"synthetic": true}')
        with track_run(
            9, "synthetic_with_evidence", "Synthetic smoke-test with a real evidence file.", registry_path=registry_path,
            search_id=search_id, candidate_index=0, fold_index=None,
        ) as record:
            record["evidence_paths"] = [str(evidence_file)]
        [saved] = query_registry(registry_path, run_id=record["run_id"])
        recomputed_checksum = compute_file_checksum(evidence_file)
        traceable = saved["evidence_paths"] == [str(evidence_file)] and recomputed_checksum is not None
        print(f"Evidence path recorded and its checksum independently recomputable: {traceable}  [{'PASS' if traceable else 'FAIL'}]")
        all_passed &= traceable

        section("5. Full tally")
        all_records = read_registry(registry_path)
        print(f"Total synthetic records in the TEMPORARY registry: {len(all_records)} "
              f"(never written to the real docs/experiment_registry.jsonl)")
        print(f"ALL CHECKS PASSED: {all_passed}")

    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
