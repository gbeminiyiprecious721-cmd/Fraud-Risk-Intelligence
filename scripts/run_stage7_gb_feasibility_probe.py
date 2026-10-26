"""Stage 7 — Gradient Boosting feasibility assessment (review item 2).

Fits the approved, UNCHANGED GradientBoostingClassifier config
(n_estimators=100, max_depth=3, random_state=42) on three chronological
TRAIN subset sizes (15,000 / 30,000 / 60,000 -- all still well short of
the full 438,125 rows) to measure the ACTUAL fit-time scaling exponent,
rather than assuming linear scaling with row count.

Fits a power law (fit_seconds ~ n_rows^k) via log-log least squares on
these 3 measured points, then extrapolates to the full 438,125-row
partition. This is a feasibility ESTIMATE, not a full-training run --
no fit on the full partition happens in this script.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    build_preprocessing_pipeline,
    load_included_columns,
    measure_train_missingness,
    select_indicator_columns,
)
from src.model_selection import build_bounded_gradient_boosting, build_chronological_pilot_subset  # noqa: E402
from src.experiment_tracking import track_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

PROBE_SIZES = [15_000, 30_000, 60_000]
FULL_TRAIN_N_ROWS = 438_125
PRACTICAL_BUDGET_SECONDS = 600  # same 10-minute budget used for the RF full-train run


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {}

    section("0. Column scope (same register-driven logic as every Stage 7 script)")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    section(f"1. Measure GradientBoostingClassifier fit time at {PROBE_SIZES} chronological TRAIN rows")
    measured = []
    for n_rows in PROBE_SIZES:
        # Stage 8 automatic logging: one tracked run per measured probe
        # size, wrapping the EXISTING fit below unchanged.
        with track_run(7, f"gb_feasibility_probe_{n_rows}_rows", "One measured point of the GBM full-data feasibility power-law fit.") as record:
            pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=n_rows)
            ct = build_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
            X = ct.fit_transform(pilot_df[all_cols])
            y = pilot_df["isFraud"].to_numpy()

            model = build_bounded_gradient_boosting()
            t0 = time.perf_counter()
            model.fit(X, y)
            fit_s = time.perf_counter() - t0
            print(f"n_rows={n_rows:6d}  shape={X.shape}  fit_seconds={fit_s:.3f}")
            measured.append({"n_rows": n_rows, "fit_seconds": fit_s, "shape": list(X.shape)})

            record["model_family"] = "GradientBoostingClassifier"
            record["hyperparameters"] = {k: v for k, v in model.get_params().items() if k in ("n_estimators", "max_depth", "random_state")}
            record["random_seed"] = model.get_params().get("random_state")
            record["preprocessing_config"] = {"pipeline": "build_preprocessing_pipeline (approved, unscaled)"}
            record["partitions"] = {"train_pilot": {"id": f"train (earliest {n_rows} rows, chronological)", "n_rows": n_rows}}
            record["feature_count"] = X.shape[1]
            record["training_time_seconds"] = round(fit_s, 3)
            record["validation_consulted"] = False
            record["validation_purpose"] = "this probe measures fit time only -- no validation scoring occurs"

    report["measured"] = measured

    section("2. Fit a power law (fit_seconds ~ n_rows^k) on the 3 measured points -- NOT assumed linear")
    log_n = np.log([m["n_rows"] for m in measured])
    log_t = np.log([m["fit_seconds"] for m in measured])
    k, log_c = np.polyfit(log_n, log_t, deg=1)
    c = np.exp(log_c)
    print(f"Fitted scaling exponent k={k:.3f} (k=1.0 would be exactly linear; k>1.0 means WORSE than linear)")
    print(f"Model: fit_seconds ~= {c:.6g} * n_rows^{k:.3f}")

    naive_linear_estimate = measured[0]["fit_seconds"] * (FULL_TRAIN_N_ROWS / measured[0]["n_rows"])
    power_law_estimate = c * (FULL_TRAIN_N_ROWS ** k)
    print(f"\nNaive LINEAR extrapolation from the 15,000-row point alone: {naive_linear_estimate:.1f}s")
    print(f"Power-law extrapolation (using the measured exponent k={k:.3f}): {power_law_estimate:.1f}s")

    report["fitted_exponent_k"] = round(float(k), 4)
    report["naive_linear_estimate_seconds"] = round(float(naive_linear_estimate), 1)
    report["power_law_estimate_seconds"] = round(float(power_law_estimate), 1)
    report["full_train_n_rows"] = FULL_TRAIN_N_ROWS
    report["practical_budget_seconds"] = PRACTICAL_BUDGET_SECONDS

    section("3. Feasibility decision")
    # Use the power-law (empirically grounded) estimate, not the naive linear one,
    # and apply a safety margin since this is an EXTRAPOLATION past the measured range.
    safety_margin = 1.5
    conservative_estimate = power_law_estimate * safety_margin
    feasible = conservative_estimate <= PRACTICAL_BUDGET_SECONDS
    print(f"Power-law estimate with a {safety_margin}x safety margin (extrapolating {FULL_TRAIN_N_ROWS/measured[-1]['n_rows']:.1f}x past "
          f"the largest measured point): {conservative_estimate:.1f}s")
    print(f"Practical budget: {PRACTICAL_BUDGET_SECONDS}s")
    print(f"FEASIBLE within budget: {feasible}")
    report["conservative_estimate_seconds"] = round(float(conservative_estimate), 1)
    report["feasible_within_budget"] = bool(feasible)

    out_path = DOCS_DIR / "_stage7_gb_feasibility_probe_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")

    # Stage 8 automatic logging: the full-data run ITSELF (never executed)
    # gets its own record, matching the backfilled historical convention --
    # a resource-budget decision, not a predictive-performance judgment.
    decision_status = "infeasible_not_run" if not feasible else "diagnostic"
    decision_reason = (
        f"Power-law extrapolation (exponent k={report['fitted_exponent_k']}, worse than linear) with a "
        f"{safety_margin}x safety margin estimated {report['conservative_estimate_seconds']}s against a "
        f"{PRACTICAL_BUDGET_SECONDS}s practical budget."
        + (" Exceeds the budget -- not launched." if not feasible else " Within budget -- see Stage 7 report for whether it was actually launched.")
    )
    with track_run(
        7, "gb_full_data_training", "The full-data (438,125-row) Gradient Boosting benchmark itself.",
    ) as decision_record:
        decision_record["status"] = decision_status
        decision_record["status_reason"] = decision_reason
        decision_record["model_family"] = "GradientBoostingClassifier"
        decision_record["hyperparameters"] = {"n_estimators": 100, "max_depth": 3, "random_state": 42}
        decision_record["partitions"] = {"train": {"id": "train (complete)", "n_rows": FULL_TRAIN_N_ROWS}}
        decision_record["runtime_settings"] = {
            "practical_budget_seconds": PRACTICAL_BUDGET_SECONDS,
            "conservative_estimate_seconds": report["conservative_estimate_seconds"],
            "fitted_scaling_exponent_k": report["fitted_exponent_k"],
        }
        decision_record["validation_consulted"] = False
        decision_record["evidence_paths"] = ["docs/_stage7_gb_feasibility_probe_raw.json"]

    return report


if __name__ == "__main__":
    main()
