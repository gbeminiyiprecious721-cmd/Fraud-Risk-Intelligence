"""Stage 9 — APPROVED, bounded Random Forest hyperparameter search.

ONE controlled search: 20 candidates (reproducible random sampling,
random_state=42, from the declared `RF_SEARCH_SPACE`) x 5 approved
chronological folds = 100 fits. This is a hyperparameter-SELECTION
exercise (unlike the preceding feasibility rounds) -- but selection
happens ONLY by mean fold Average Precision on `train`-internal folds,
never by touching `validation`/`final_test`.

Resource/safety design, stated plainly:
- Sequential, single process -- no nested parallel searches. Each
  individual `RandomForestClassifier` fit still uses `n_jobs=-1`
  internally (ordinary multithreading within one fit), identical to
  every prior Stage 7/9 measurement -- this is NOT additional outer
  parallelism across candidates or folds.
- A hard 3,600-second TOTAL wall-clock budget, checked BEFORE starting
  each new candidate-fold fit (a fit already in progress is always
  allowed to finish -- never hard-killed mid-fit, which would destroy
  that fit's partial work for no benefit).
- Peak memory is MONITORED, not capped: `resource.getrusage().ru_maxrss`
  is read and logged after every fit. This is a same-process CUMULATIVE
  high-water mark (it can only grow across the run), NOT an
  isolated-per-fit measurement -- stated explicitly, never misrepresented
  as a per-fit figure or as an enforced ceiling. A separate, conservative
  SAFETY ceiling (45GB, ~70% of this machine's 64GB) stops the search if
  crossed, as a monitored safeguard, not a hard OS-level cap (macOS'
  `ulimit -v` is confirmed non-functional on this platform -- Stage 7's
  own documented finding).
- No preprocessing caching this run (explicit instruction) -- every
  candidate-fold pair refits `fit_transform_fold` from scratch.
- Resume-safe: before fitting any (candidate_index, fold_index) pair,
  the real registry is checked for an existing `status="completed"`
  record under this EXACT `search_id` -- if found, it is reused (not
  re-fit), and only ever from this exact search specification, never a
  different one.
- A fold-level exception (not a budget/memory/safeguard stop) is caught,
  logged as `status="failed"` by `track_run()`, and marks that ONE
  candidate incomplete -- the search continues to the next candidate
  rather than aborting entirely, preserving whatever other, independent
  candidates' results already exist. A STRUCTURAL failure (budget
  exceeded, memory ceiling crossed, or a fold failing its own
  chronology/tie/sufficient-positives re-verification) stops the ENTIRE
  search immediately, since those are not candidate-specific.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.preprocessing import load_included_columns, load_partition_df  # noqa: E402
from src.model_selection import align_labels_to_partition_df, evaluate_predictions, load_labels  # noqa: E402
from src.hyperparameter_tuning import (  # noqa: E402
    build_expanding_window_folds,
    fit_transform_fold,
    sample_rf_candidates,
    verify_fold,
)
from src.experiment_tracking import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    compute_feature_schema_fingerprint,
    query_registry,
    track_run,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

SEARCH_ID = "stage9-rf-randomized-search-20x5"
N_SPLITS = 5
N_CANDIDATES = 20
RANDOM_STATE = 42
HARD_BUDGET_SECONDS = 3600
MEMORY_CEILING_BYTES = 45 * 1024 ** 3  # ~70% of this machine's 64GB -- a monitored safeguard, not an enforced OS cap
STAGE7_REFERENCE_CONFIG = {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 20, "max_features": "sqrt"}


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}", flush=True)


def fingerprint_ids(ids: pd.Series) -> str:
    joined = "\x1f".join(str(i) for i in sorted(ids.tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def current_peak_rss_bytes() -> int:
    """macOS reports `ru_maxrss` in BYTES already (unlike Linux, which
    reports KB) -- verified empirically against `/usr/bin/time -l`'s
    "peak memory footprint" in every prior Stage 7/8/9 script on this
    platform."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main() -> dict:
    search_start = time.time()
    report: dict = {"search_id": SEARCH_ID, "n_splits": N_SPLITS, "n_candidates": N_CANDIDATES, "hard_budget_seconds": HARD_BUDGET_SECONDS}

    section("0. Preflight: final_test refusal check")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    try:
        get_partition(assignment, ["train", "final_test"])
        report["final_test_refused"] = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        report["final_test_refused"] = True

    section("1. Declare the COMPLETE candidate list BEFORE any fits")
    candidates = sample_rf_candidates(N_CANDIDATES, random_state=RANDOM_STATE)
    for i, c in enumerate(candidates):
        print(f"candidate {i}: {c}")
    report["declared_candidates"] = candidates
    stage7_in_sample = any(
        c["n_estimators"] == STAGE7_REFERENCE_CONFIG["n_estimators"]
        and c["max_depth"] == STAGE7_REFERENCE_CONFIG["max_depth"]
        and c["min_samples_leaf"] == STAGE7_REFERENCE_CONFIG["min_samples_leaf"]
        and c["max_features"] == STAGE7_REFERENCE_CONFIG["max_features"]
        for c in candidates
    )
    report["stage7_reference_config_in_sample"] = stage7_in_sample
    print(f"Stage 7 reference config present among the 20 sampled candidates: {stage7_in_sample}")
    declaration_path = DOCS_DIR / "_stage9_search_declaration_raw.json"
    with open(declaration_path, "w") as f:
        json.dump({"search_id": SEARCH_ID, "candidates": candidates, "n_splits": N_SPLITS,
                    "random_state": RANDOM_STATE, "stage7_reference_config_in_sample": stage7_in_sample}, f, indent=2)
    print(f"Declaration written to {declaration_path} before any fit.")

    section("2. Load the COMPLETE approved train partition once; build + pre-verify the 5 folds once")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    numeric_columns = columns["numeric"]
    categorical_columns = columns["categorical"]
    all_cols = numeric_columns + categorical_columns

    train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols)
    labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
    train_df = align_labels_to_partition_df(train_df, labels)
    if "TransactionDT" not in train_df.columns:
        raise ValueError("TransactionDT missing from the loaded partition -- expected it to be a register-included column.")
    print(f"Train partition: {len(train_df)} rows (expected 438,125)")

    df_sorted, folds = build_expanding_window_folds(train_df, N_SPLITS)
    fold_checks = []
    for j, (train_idx, val_idx) in enumerate(folds):
        fc = verify_fold(df_sorted, train_idx, val_idx)
        print(f"fold {j} pre-flight verification: {fc}")
        fold_checks.append(fc)
        if not fc["all_checks_passed"]:
            print(f"SCIENTIFIC SAFEGUARD FAILURE on fold {j} -- aborting before any fit.")
            report["aborted"] = True
            report["aborted_reason"] = f"fold {j} failed pre-flight verification: {fc}"
            _write_report(report)
            return report
    report["fold_checks"] = fold_checks

    fold_fingerprints = [
        {"train_fp": fingerprint_ids(df_sorted.loc[tr, "TransactionID"]), "val_fp": fingerprint_ids(df_sorted.loc[va, "TransactionID"])}
        for tr, va in folds
    ]

    section("3. Sequential candidate x fold loop -- no caching, no parallel candidates, resume-safe")
    results = []
    stopped_early = False
    stop_reason = None

    for cand_idx, params in enumerate(candidates):
        if stopped_early:
            break
        candidate_fold_results = []
        candidate_incomplete = False

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            elapsed = time.time() - search_start
            if elapsed >= HARD_BUDGET_SECONDS:
                stopped_early = True
                stop_reason = f"hard {HARD_BUDGET_SECONDS}s total search budget reached before candidate {cand_idx} fold {fold_idx} (elapsed={elapsed:.1f}s)"
                print(stop_reason)
                break

            # Resume-safety: reuse an existing COMPLETED record from this
            # EXACT search_id/candidate_index/fold_index if one exists --
            # never from a different search_id (a different spec/fold def).
            existing = query_registry(DEFAULT_REGISTRY_PATH, search_id=SEARCH_ID, candidate_index=cand_idx, fold_index=fold_idx)
            existing_completed = [r for r in existing if r["status"] == "completed"]
            if existing_completed:
                saved = existing_completed[0]
                print(f"candidate {cand_idx} fold {fold_idx}: REUSED existing completed record {saved['run_id']} (resume-safe skip, no re-fit)")
                candidate_fold_results.append({
                    "fold_index": fold_idx, "metrics": saved["metrics"],
                    "preprocessing_seconds": saved["runtime_settings"].get("preprocessing_seconds") if saved.get("runtime_settings") else None,
                    "fit_seconds": saved["training_time_seconds"], "predict_seconds": saved["inference_time_seconds"],
                    "train_prevalence": saved["runtime_settings"].get("train_prevalence") if saved.get("runtime_settings") else None,
                    "val_prevalence": saved["runtime_settings"].get("val_prevalence") if saved.get("runtime_settings") else None,
                    "reused": True,
                })
                continue

            # Re-verify THIS specific fold again (cheap, per explicit
            # instruction), even though folds are fixed across candidates.
            fc = verify_fold(df_sorted, train_idx, val_idx)
            if not fc["all_checks_passed"]:
                stopped_early = True
                stop_reason = f"scientific safeguard re-verification failed for candidate {cand_idx} fold {fold_idx}: {fc}"
                print(stop_reason)
                break

            try:
                with track_run(
                    9, "stage9_rf_search_candidate_fold", f"Approved RF search: candidate {cand_idx} ({params}), fold {fold_idx}.",
                    search_id=SEARCH_ID, candidate_index=cand_idx, fold_index=fold_idx,
                ) as record:
                    t0 = time.perf_counter()
                    fitted = fit_transform_fold(df_sorted, train_idx, val_idx, numeric_columns, categorical_columns)
                    preprocessing_seconds = time.perf_counter() - t0

                    model = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, class_weight=None, **params)
                    t0 = time.perf_counter()
                    model.fit(fitted["X_train"], fitted["y_train"])
                    fit_seconds = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    scores = model.predict_proba(fitted["X_val"])[:, 1]
                    predict_seconds = time.perf_counter() - t0
                    metrics = evaluate_predictions(fitted["y_val"], scores)
                    train_prevalence = float(fitted["y_train"].mean())
                    val_prevalence = float(fitted["y_val"].mean())
                    peak_rss_bytes = current_peak_rss_bytes()

                    print(f"candidate {cand_idx} fold {fold_idx}: preproc={preprocessing_seconds:.2f}s fit={fit_seconds:.2f}s "
                          f"predict={predict_seconds:.3f}s AP={metrics['average_precision']} cum_peak_rss={peak_rss_bytes/1e9:.2f}GB")

                    record["model_family"] = "RandomForestClassifier"
                    record["hyperparameters"] = {**params, "n_jobs": -1, "random_state": RANDOM_STATE, "class_weight": None}
                    record["random_seed"] = RANDOM_STATE
                    record["preprocessing_config"] = {"pipeline": "fit_transform_fold (per-fold-refit, no caching this run)"}
                    record["feature_count"] = fitted["fitted_feature_count"]
                    record["feature_schema_fingerprint"] = compute_feature_schema_fingerprint(
                        fitted["numeric_with_indicator"] + fitted["numeric_without_indicator"] + categorical_columns
                    )
                    record["partitions"] = {
                        "train_fold": {"id": f"candidate {cand_idx} fold {fold_idx} train", "n_rows": len(train_idx),
                                       "row_fingerprint": fold_fingerprints[fold_idx]["train_fp"], "dt_range": fc["train_dt_range"]},
                        "val_fold": {"id": f"candidate {cand_idx} fold {fold_idx} val", "n_rows": len(val_idx),
                                     "row_fingerprint": fold_fingerprints[fold_idx]["val_fp"], "dt_range": fc["val_dt_range"]},
                    }
                    record["training_time_seconds"] = round(fit_seconds, 3)
                    record["inference_time_seconds"] = round(predict_seconds, 3)
                    record["peak_memory_bytes"] = int(peak_rss_bytes)
                    record["runtime_settings"] = {
                        "preprocessing_seconds": round(preprocessing_seconds, 3),
                        "train_prevalence": round(train_prevalence, 5), "val_prevalence": round(val_prevalence, 5),
                        "peak_memory_caveat": "cumulative same-process high-water mark (ru_maxrss) across the whole sequential search, NOT isolated to this single fit -- monitored, never an enforced cap.",
                    }
                    record["metrics"] = metrics
                    record["validation_consulted"] = False
                    record["validation_purpose"] = "evaluated on this fold's own held-out rows only; the external validation partition was never read"
                    record["evidence_paths"] = [str(declaration_path.relative_to(ROOT))]

                candidate_fold_results.append({
                    "fold_index": fold_idx, "metrics": metrics, "preprocessing_seconds": preprocessing_seconds,
                    "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                    "train_prevalence": train_prevalence, "val_prevalence": val_prevalence, "reused": False,
                })
            except Exception as exc:
                candidate_incomplete = True
                print(f"candidate {cand_idx} fold {fold_idx} FAILED (isolated fit exception, logged, search continues to next candidate): {exc}")
                continue

            if peak_rss_bytes > MEMORY_CEILING_BYTES:
                stopped_early = True
                stop_reason = f"cumulative process RSS {peak_rss_bytes / 1e9:.1f}GB exceeded the {MEMORY_CEILING_BYTES / 1e9:.0f}GB monitored safety ceiling after candidate {cand_idx} fold {fold_idx}"
                print(stop_reason)
                break

        all_five = len(candidate_fold_results) == N_SPLITS and not candidate_incomplete
        results.append({"candidate_index": cand_idx, "params": params, "fold_results": candidate_fold_results, "all_five_folds_successful": all_five})

    report["stopped_early"] = stopped_early
    report["stop_reason"] = stop_reason
    report["results"] = results
    report["total_wall_seconds"] = round(time.time() - search_start, 1)

    section("4. Rank eligible candidates (all 5 folds successful) by mean fold AP")
    eligible = [r for r in results if r["all_five_folds_successful"]]
    for r in eligible:
        aps = [f["metrics"]["average_precision"] for f in r["fold_results"]]
        r["mean_ap"] = sum(aps) / len(aps)
        r["std_ap"] = (sum((a - r["mean_ap"]) ** 2 for a in aps) / len(aps)) ** 0.5
    eligible.sort(key=lambda r: r["mean_ap"], reverse=True)
    report["n_eligible_candidates"] = len(eligible)
    report["n_incomplete_candidates"] = len(results) - len(eligible)
    for rank, r in enumerate(eligible):
        print(f"rank {rank}: candidate {r['candidate_index']} {r['params']} mean_AP={r['mean_ap']:.5f} std={r['std_ap']:.5f}")

    section("5. Search-level summary record")
    with track_run(
        9, "stage9_rf_search_summary", "Approved RF search -- overall summary (not a selection decision by itself).",
        search_id=SEARCH_ID,
    ) as summary_record:
        summary_record["status"] = "completed" if not stopped_early else "interrupted"
        summary_record["status_reason"] = stop_reason if stopped_early else "All 100 planned fits attempted within budget."
        summary_record["runtime_settings"] = {
            "n_eligible_candidates": len(eligible), "n_incomplete_candidates": len(results) - len(eligible),
            "total_wall_seconds": report["total_wall_seconds"], "stage7_reference_config_in_sample": stage7_in_sample,
        }
        summary_record["validation_consulted"] = False

    out_path = DOCS_DIR / "_stage9_hyperparameter_search_raw.json"
    _write_report(report, out_path)
    print(f"\nFull structured report written to {out_path}")
    return report


def _write_report(report: dict, out_path: Path | None = None) -> None:
    out_path = out_path or (DOCS_DIR / "_stage9_hyperparameter_search_raw.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
