"""Stage 8 closure, carried-forward requirement 7 — "before the first real
Stage 9 tuning run, verify the tracker end-to-end using a small synthetic
experiment, including success, failure and exception paths."

Run proactively now (Stage 9 has not started) so this verification exists
as evidence before any real tuning run, not as a last-minute check.

Uses a TEMPORARY registry file (a tempfile, deleted at the end), NEVER the
real `docs/experiment_registry.jsonl` -- these are synthetic smoke-test
records, not real experiments, and must never be mixed into the
authoritative historical record (carried-forward requirement 1).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_tracking import (  # noqa: E402
    MalformedRecordError,
    read_registry,
    track_run,
)


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> bool:
    all_passed = True
    with tempfile.TemporaryDirectory() as tmp:
        registry_path = Path(tmp) / "synthetic_verification_registry.jsonl"

        section("1. Success path")
        with track_run(99, "synthetic_success", "Synthetic smoke-test experiment, success path.", registry_path=registry_path) as record:
            record["model_family"] = "SyntheticModel"
            record["metrics"] = {"synthetic_metric": 0.5}
        records = read_registry(registry_path)
        ok = len(records) == 1 and records[0]["status"] == "completed" and records[0]["metrics"]["synthetic_metric"] == 0.5
        print(f"Success path -> status={records[-1]['status']!r}, metrics={records[-1]['metrics']}  [{'PASS' if ok else 'FAIL'}]")
        all_passed &= ok

        section("2. Exception path (unhandled exception mid-run)")
        raised = False
        try:
            with track_run(99, "synthetic_exception", "Synthetic smoke-test experiment, exception path.", registry_path=registry_path) as record:
                record["model_family"] = "SyntheticModel"
                raise ValueError("synthetic simulated crash")
        except ValueError as exc:
            raised = str(exc) == "synthetic simulated crash"
        records = read_registry(registry_path)
        last = records[-1]
        ok = raised and last["status"] == "failed" and "synthetic simulated crash" in last["status_reason"] and last["model_family"] == "SyntheticModel"
        print(f"Exception path -> original exception re-raised unchanged: {raised}; logged status={last['status']!r}, status_reason={last['status_reason']!r}  [{'PASS' if ok else 'FAIL'}]")
        print("Partial progress (model_family, set before the crash) was preserved in the failed record, not discarded.")
        all_passed &= ok

        section("3. Explicit non-'completed' status path (e.g. infeasible/interrupted, no exception)")
        with track_run(99, "synthetic_infeasible", "Synthetic smoke-test experiment, explicit non-completed status.", registry_path=registry_path) as record:
            record["status"] = "infeasible_not_run"
            record["status_reason"] = "synthetic: resource budget exceeded"
        records = read_registry(registry_path)
        last = records[-1]
        ok = last["status"] == "infeasible_not_run"
        print(f"Explicit-status path -> status={last['status']!r}  [{'PASS' if ok else 'FAIL'}]")
        all_passed &= ok

        section("4. Malformed-status guard (non-completed status with NO reason must be refused)")
        guard_raised = False
        try:
            with track_run(99, "synthetic_malformed", "Synthetic smoke-test experiment, malformed-status guard.", registry_path=registry_path) as record:
                record["status"] = "failed"
                record["status_reason"] = None  # deliberately invalid
        except MalformedRecordError:
            guard_raised = True
        print(f"Malformed-status guard raised MalformedRecordError as expected: {guard_raised}  [{'PASS' if guard_raised else 'FAIL'}]")
        all_passed &= guard_raised

        section("5. Final tally")
        all_records = read_registry(registry_path)
        print(f"Total synthetic records written to the TEMPORARY registry: {len(all_records)} (never written to the real docs/experiment_registry.jsonl)")
        print(f"ALL CHECKS PASSED: {all_passed}")

    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
