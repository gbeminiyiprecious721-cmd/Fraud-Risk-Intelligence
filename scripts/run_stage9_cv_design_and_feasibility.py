"""Stage 9 — chronological CV design verification + bounded feasibility probe.

TRAIN-ONLY throughout. Never reads `validation` or `final_test`. Does not
execute a real hyperparameter search -- that remains unapproved. This
script:

  1. Measures the real `train` partition's duplicate-timestamp structure
     and checks sklearn's `TimeSeriesSplit` against it across several
     candidate `n_splits` values (structural verification, no model fit).
  2. Builds this project's deterministic, tie-safe expanding-window folds
     on the COMPLETE `train` partition at the proposed `n_splits=5`, and
     verifies every fold against this stage's requirements (chronology,
     no tie-straddle, sufficient positives, documented size/range/prevalence).
  3. Runs a SMALL, BOUNDED feasibility probe: fits the Stage 7 reference
     Random Forest config on each fold of the EXISTING 15,000-row
     chronological pilot subset (already used throughout Stage 7, no new
     approval needed to reuse it), measuring REAL per-fold
     preprocessing+fit time, and logs each fold's probe fit to the real
     experiment registry (a genuine small experiment, not a synthetic
     test). Extrapolates a structural, explicitly-labeled LOWER-BOUND cost
     estimate for the PROPOSED (not yet approved) 20-candidate x 5-fold
     search -- a full-`train`-scale search would cost more than this
     bounded probe measures, stated plainly, not glossed over.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.preprocessing import load_included_columns  # noqa: E402
from src.model_selection import build_bounded_random_forest, build_chronological_pilot_subset  # noqa: E402
from src.hyperparameter_tuning import (  # noqa: E402
    assess_timeseries_split_compatibility,
    build_expanding_window_folds,
    estimate_search_cost,
    find_tie_groups,
    fit_transform_fold,
    sort_chronologically,
    verify_fold,
)
from src.experiment_tracking import track_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

PROPOSED_N_SPLITS = 5
PROBE_N_SPLITS = 3  # smaller, appropriate for the bounded 15,000-row probe subset
PROBE_PILOT_N_ROWS = 15_000
PROPOSED_N_CANDIDATES = 20
SEARCH_ID = "stage9-rf-cv-design-feasibility"


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {}

    section("0. final_test refusal re-check")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    try:
        get_partition(assignment, ["train", "final_test"])
        report["final_test_refused"] = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        report["final_test_refused"] = True

    section("1. Real tie structure of the COMPLETE train partition")
    train_ids = set(get_partition(assignment, ["train"])["TransactionID"])
    full_train = pd.read_csv(TRAIN_TXN, usecols=["TransactionID", "TransactionDT", "isFraud"])
    full_train = full_train.loc[full_train["TransactionID"].isin(train_ids)]
    full_train_sorted = sort_chronologically(full_train)
    tie_report = find_tie_groups(full_train_sorted)
    print(tie_report)
    report["tie_structure"] = tie_report

    section("2. TimeSeriesSplit compatibility across candidate n_splits (NOT assumed safe)")
    ts_assessment = {}
    for n_splits in (3, 4, 5, 6, 7, 8, 10):
        res = assess_timeseries_split_compatibility(full_train_sorted, n_splits)
        print(f"n_splits={n_splits}: any_tie_straddled={res['any_tie_straddled']}")
        ts_assessment[n_splits] = res["any_tie_straddled"]
    report["timeseries_split_any_tie_straddled_by_n_splits"] = ts_assessment
    report["timeseries_split_safe_at_proposed_n_splits"] = not ts_assessment[PROPOSED_N_SPLITS]

    section(f"3. Build this project's deterministic, tie-safe expanding-window folds (n_splits={PROPOSED_N_SPLITS}) on the COMPLETE train partition")
    _, full_folds = build_expanding_window_folds(full_train_sorted, PROPOSED_N_SPLITS)
    fold_reports = []
    for i, (train_idx, val_idx) in enumerate(full_folds):
        fr = verify_fold(full_train_sorted, train_idx, val_idx)
        print(f"fold {i}: {fr}")
        fold_reports.append(fr)
    report["full_train_fold_verification"] = fold_reports
    report["all_full_train_folds_passed"] = all(f["all_checks_passed"] for f in fold_reports)

    section(f"4. Bounded feasibility probe -- {PROBE_PILOT_N_ROWS}-row chronological pilot, {PROBE_N_SPLITS} folds, Stage 7 reference RF config")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    numeric_columns = columns["numeric"]
    categorical_columns = columns["categorical"]
    all_cols = numeric_columns + categorical_columns

    pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=PROBE_PILOT_N_ROWS)
    pilot_df_sorted, probe_folds = build_expanding_window_folds(pilot_df, PROBE_N_SPLITS)

    probe_fold_times = []
    for fold_index, (train_idx, val_idx) in enumerate(probe_folds):
        fr = verify_fold(pilot_df_sorted, train_idx, val_idx, min_val_positive=5)
        with track_run(
            9, "stage9_cv_feasibility_probe_fold", f"Bounded feasibility probe, fold {fold_index} of {PROBE_N_SPLITS} -- Stage 7 reference RF config, {PROBE_PILOT_N_ROWS}-row pilot.",
            search_id=SEARCH_ID, candidate_index=0, fold_index=fold_index,
        ) as record:
            t0 = time.perf_counter()
            fitted = fit_transform_fold(
                pilot_df_sorted, train_idx, val_idx, numeric_columns, categorical_columns
            )
            preprocessing_seconds = time.perf_counter() - t0

            model = build_bounded_random_forest()
            t0 = time.perf_counter()
            model.fit(fitted["X_train"], fitted["y_train"])
            fit_seconds = time.perf_counter() - t0

            total_fold_seconds = preprocessing_seconds + fit_seconds
            print(f"fold {fold_index}: preprocessing={preprocessing_seconds:.3f}s fit={fit_seconds:.3f}s total={total_fold_seconds:.3f}s "
                  f"train_n={len(train_idx)} val_n={len(val_idx)} feature_count={fitted['fitted_feature_count']} fold_checks_passed={fr['all_checks_passed']}")
            probe_fold_times.append(total_fold_seconds)

            record["model_family"] = "RandomForestClassifier"
            record["hyperparameters"] = {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 20, "random_state": 42}
            record["random_seed"] = 42
            record["preprocessing_config"] = {"pipeline": "fit_transform_fold (per-fold-refit, Stage 9 design)"}
            record["partitions"] = {"train_fold": {"id": f"pilot-subset fold {fold_index} train", "n_rows": len(train_idx)}, "val_fold": {"id": f"pilot-subset fold {fold_index} val", "n_rows": len(val_idx)}}
            record["feature_count"] = fitted["fitted_feature_count"]
            record["training_time_seconds"] = round(fit_seconds, 3)
            record["runtime_settings"] = {"preprocessing_seconds": round(preprocessing_seconds, 3), "fold_requirements_passed": fr["all_checks_passed"]}
            record["validation_consulted"] = False
            record["validation_purpose"] = "this probe never reads the real `validation` partition -- only a train-only CV fold's own held-out rows"

    report["probe_fold_seconds"] = [round(t, 3) for t in probe_fold_times]
    avg_probe_fold_seconds = sum(probe_fold_times) / len(probe_fold_times)
    report["avg_probe_fold_seconds"] = round(avg_probe_fold_seconds, 3)

    section("5. Structural cost estimate for the PROPOSED (not yet approved) search")
    cost = estimate_search_cost(avg_probe_fold_seconds, PROPOSED_N_CANDIDATES, PROPOSED_N_SPLITS, safety_margin=1.5)
    print(cost)
    print("\nLOWER-BOUND CAVEAT, stated explicitly: this estimate is built from per-fold costs measured on the "
          f"{PROBE_PILOT_N_ROWS}-row bounded pilot. The proposed real search would run on larger folds (up to the "
          "complete ~438,125-row train partition in its final expanding-window fold), which would cost MORE per fit, "
          "not less. This is a feasibility floor, not a promised runtime.")
    report["cost_estimate_from_bounded_probe"] = cost

    with track_run(
        9, "stage9_search_feasibility_decision", "Structural feasibility assessment for the proposed 20-candidate x 5-fold RF search -- not an actual search.",
        search_id=SEARCH_ID,
    ) as decision_record:
        decision_record["status"] = "diagnostic"
        decision_record["status_reason"] = "Design/feasibility assessment only, per explicit instruction -- the real search has not been approved or executed."
        decision_record["model_family"] = "RandomForestClassifier"
        decision_record["runtime_settings"] = cost
        decision_record["validation_consulted"] = False

    out_path = DOCS_DIR / "_stage9_cv_design_and_feasibility_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
