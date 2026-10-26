"""Stage 9 — read-only scientific audit of the already-completed, approved
20-candidate RF hyperparameter search.

Trains NOTHING. Reads ONLY `docs/experiment_registry.jsonl` (the
authoritative structured record, per Stage 8's carried-forward
requirement 1) and reports independently-recomputed findings -- never
trusts the prior run's own printed console log as the source of truth,
per this project's established "independently re-verify" discipline
(mirrors `scripts/verify_stage7_rf_pilot.py`'s role for the RF pilot).

Never imports anything capable of reading `validation`/`final_test`, or
fitting a model -- this script's only import from the modeling stack is
read-only registry access.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_tracking import query_registry, DEFAULT_REGISTRY_PATH  # noqa: E402

SEARCH_ID = "stage9-rf-randomized-search-20x5"
N_SPLITS = 5
N_CANDIDATES = 20


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {"search_id": SEARCH_ID}

    all_records = query_registry(DEFAULT_REGISTRY_PATH, search_id=SEARCH_ID)
    fold_records = [r for r in all_records if r.get("fold_index") is not None]
    summary_records = [r for r in all_records if r.get("fold_index") is None]

    section("1. Candidate index <-> rank reconciliation")
    by_candidate: dict[int, list[dict]] = {}
    for r in fold_records:
        by_candidate.setdefault(r["candidate_index"], []).append(r)

    candidate_summaries = []
    for cand_idx in sorted(by_candidate):
        records = sorted(by_candidate[cand_idx], key=lambda r: r["fold_index"])
        completed = [r for r in records if r["status"] == "completed"]
        failed = [r for r in records if r["status"] == "failed"]
        eligible = len(completed) == N_SPLITS and len(failed) == 0
        aps = [r["metrics"]["average_precision"] for r in completed] if eligible else None
        mean_ap = statistics.mean(aps) if aps else None
        std_ap = statistics.pstdev(aps) if aps and len(aps) > 1 else (0.0 if aps else None)
        candidate_summaries.append({
            "candidate_index": cand_idx, "n_fold_records": len(records),
            "n_completed": len(completed), "n_failed": len(failed), "eligible": eligible,
            "mean_ap": mean_ap, "std_ap": std_ap,
            "hyperparameters": completed[0]["hyperparameters"] if completed else (records[0].get("hyperparameters") if records else None),
            "per_fold_ap": aps,
        })

    eligible_summaries = [c for c in candidate_summaries if c["eligible"]]
    eligible_summaries.sort(key=lambda c: c["mean_ap"], reverse=True)
    report["ranking"] = eligible_summaries

    target_ap = 0.51236
    matches = [c for c in eligible_summaries if abs(round(c["mean_ap"], 5) - target_ap) < 1e-4]
    print(f"Candidates whose INDEPENDENTLY RECOMPUTED mean AP matches {target_ap} (tolerance 1e-4): "
          f"{[(m['candidate_index'], round(m['mean_ap'], 5)) for m in matches]}")
    if len(matches) == 1:
        print(f"CONFIRMED: candidate_index={matches[0]['candidate_index']} is the unique match. "
              f"Its rank in the recomputed ranking: {eligible_summaries.index(matches[0])}. "
              f"Hyperparameters: {matches[0]['hyperparameters']}")
    report["ap_target_reconciliation"] = {"target": target_ap, "matches": [(m["candidate_index"], round(m["mean_ap"], 5)) for m in matches]}

    section("2. All 18 eligible candidates -- fold-record completeness, config consistency, fingerprint consistency")
    fingerprint_by_fold: dict[int, set] = {}
    config_consistency_ok = True
    fingerprint_consistency_ok = True
    for cand_idx in sorted(by_candidate):
        records = sorted(by_candidate[cand_idx], key=lambda r: r["fold_index"])
        completed = [r for r in records if r["status"] == "completed"]
        if len(completed) != N_SPLITS:
            continue  # not eligible, already excluded from ranking -- not an error
        # Config must be IDENTICAL across all 5 of this candidate's own folds.
        hp0 = completed[0]["hyperparameters"]
        same_config = all(r["hyperparameters"] == hp0 for r in completed)
        if not same_config:
            config_consistency_ok = False
            print(f"CONFIG MISMATCH within candidate {cand_idx}'s own 5 folds!")
        for r in completed:
            fold_idx = r["fold_index"]
            fp = r["partitions"]["val_fold"]["row_fingerprint"] + "|" + r["partitions"]["train_fold"]["row_fingerprint"]
            fingerprint_by_fold.setdefault(fold_idx, set()).add(fp)
            if not r.get("evidence_paths"):
                print(f"MISSING evidence_paths for candidate {cand_idx} fold {fold_idx}!")
    for fold_idx, fps in sorted(fingerprint_by_fold.items()):
        identical = len(fps) == 1
        fingerprint_consistency_ok = fingerprint_consistency_ok and identical
        print(f"fold {fold_idx}: {len(fps)} distinct (train_fp|val_fp) combination(s) across all candidates "
              f"({'IDENTICAL -- same rows every time' if identical else 'MISMATCH -- folds drifted between candidates!'})")
    report["config_consistency_ok"] = config_consistency_ok
    report["fingerprint_consistency_ok"] = fingerprint_consistency_ok
    print(f"\nPer-candidate config internally consistent (same 5-fold hyperparameters): {config_consistency_ok}")
    print(f"Per-fold row fingerprint identical across every candidate that reached it: {fingerprint_consistency_ok}")

    section("3. Independently recomputed fit totals")
    n_completed = sum(1 for r in fold_records if r["status"] == "completed")
    n_failed = sum(1 for r in fold_records if r["status"] == "failed")
    n_planned = N_CANDIDATES * N_SPLITS
    n_unattempted = n_planned - n_completed - n_failed
    print(f"Planned: {n_planned}  Completed: {n_completed}  Failed: {n_failed}  Unattempted: {n_unattempted}")
    report["fit_totals"] = {"planned": n_planned, "completed": n_completed, "failed": n_failed, "unattempted": n_unattempted}
    assert n_completed == 94 and n_failed == 0 and n_unattempted == 6, "Recomputed totals do not match the previously reported 94/0/6 -- investigate before trusting the report."
    print("MATCHES the previously reported 94 completed / 0 failed / 6 unattempted.")

    section("4. Paired fold-by-fold comparison, rank 0 vs rank 1")
    top2 = eligible_summaries[:2]
    rank0, rank1 = top2[0], top2[1]
    paired_diffs = [a - b for a, b in zip(rank0["per_fold_ap"], rank1["per_fold_ap"])]
    wins = sum(1 for d in paired_diffs if d > 0)
    print(f"Rank 0 = candidate {rank0['candidate_index']} {rank0['hyperparameters']}")
    print(f"Rank 1 = candidate {rank1['candidate_index']} {rank1['hyperparameters']}")
    for fold_idx, (a, b, d) in enumerate(zip(rank0["per_fold_ap"], rank1["per_fold_ap"], paired_diffs)):
        print(f"  fold {fold_idx}: rank0_AP={a:.5f}  rank1_AP={b:.5f}  diff={d:+.5f}")
    print(f"Rank 0 beat rank 1 on {wins} of {N_SPLITS} folds.")
    report["paired_comparison"] = {
        "rank0_candidate": rank0["candidate_index"], "rank1_candidate": rank1["candidate_index"],
        "paired_diffs": paired_diffs, "rank0_wins": wins, "n_folds": N_SPLITS,
    }

    section("5. Scope of the winner, stated precisely")
    print(f"Candidate {rank0['candidate_index']} is the highest-mean-AP configuration among the 18 candidates that "
          f"completed all 5 folds within the approved search's budget -- NOT a confirmed winner among all 20 "
          f"originally-declared candidates, 2 of which (indices 18, 19) were never fully evaluated and whose "
          f"relative standing remains genuinely unknown.")

    section("6. validation/final_test access re-confirmation")
    validation_touched = any(r.get("validation_consulted") not in (False, None) for r in fold_records + summary_records)
    print(f"Any record in this search reporting validation_consulted != False: {validation_touched}")
    report["validation_or_final_test_touched"] = validation_touched
    assert not validation_touched

    section("7. Provisional tuned candidate (conditional on this audit)")
    audit_passed = config_consistency_ok and fingerprint_consistency_ok and not validation_touched and len(matches) == 1
    report["audit_passed"] = audit_passed
    if audit_passed:
        print(f"AUDIT PASSED. Provisional tuned candidate: index {rank0['candidate_index']}, "
              f"hyperparameters {rank0['hyperparameters']}, mean CV AP {rank0['mean_ap']:.5f} (std {rank0['std_ap']:.5f}).")
        print("This is PROVISIONAL, not a final selection -- Stage 9 is not marked complete by this audit alone.")
    else:
        print("AUDIT DID NOT FULLY PASS -- see flags above. No provisional candidate identified.")

    out_path = Path(__file__).resolve().parent.parent / "docs" / "_stage9_search_audit_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured audit report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
