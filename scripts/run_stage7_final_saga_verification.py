"""Stage 7 — final bounded SAGA convergence verification.

ONE experiment, exactly as specified, no parameter search:

    LogisticRegression(solver="saga", penalty="l2", C=1.0, max_iter=3000,
                        random_state=42, class_weight=None)

on the ORIGINAL 542-feature (Arm A) representation, the identical
15,000-earliest-chronological-row `train` subset used by every prior
Stage 7 script. `penalty="l2"`, `C=1.0`, and sklearn's default `tol`
(1e-4) are the SAME values every prior round used -- carried forward
unchanged, not re-chosen here.

Strictly training-only: validation is never loaded in this script, so
there is no way for a validation label to influence anything here, by
construction, not just by discipline. `final_test` is never requested.

Meant to be run under a 120-second OS-level wall-clock budget (the
harness's own process timeout, which can actually kill a stuck process --
unlike Python's `signal.alarm`, which cannot interrupt a saga/liblinear C
loop, per the lesson from the original Stage 7 diagnostic).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.feature_engineering import collapsed_indicator_representatives, find_duplicate_indicator_groups  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    load_included_columns,
    measure_train_missingness,
    select_indicator_columns,
)
from src.model_selection import (  # noqa: E402
    build_chronological_pilot_subset,
    build_scaled_pipeline_with_collapsed_indicators,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

PILOT_N_ROWS = 15_000  # identical to every prior Stage 7 driver script
MAX_ITER = 3000


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {}

    section("0. Rebuild the SAME 15,000-row chronological pilot subset (Arm A representation only)")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=PILOT_N_ROWS)
    y_train = pilot_df["isFraud"].to_numpy()
    print(f"Pilot subset: {len(pilot_df)} rows, fraud prevalence {y_train.mean():.4%} (identical subset, every prior Stage 7 script)")
    report["pilot_n_rows"] = len(pilot_df)
    report["pilot_fraud_prevalence"] = round(float(y_train.mean()), 5)

    section("0b. Confirm final_test still refused -- validation is never even loaded in this script")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    try:
        get_partition(assignment, ["train", "final_test"])
        report["final_test_refused"] = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        report["final_test_refused"] = True

    section("1. Build Arm A's ORIGINAL 542-feature representation (unchanged indicator-collapse prototype)")
    indicator_groups = find_duplicate_indicator_groups(pilot_df, numeric_with_indicator)
    representatives = collapsed_indicator_representatives(indicator_groups)
    ct = build_scaled_pipeline_with_collapsed_indicators(numeric_no_indicator, numeric_with_indicator, categorical, representatives)

    t0 = time.perf_counter()
    X_train = ct.fit_transform(pilot_df[all_cols])
    preprocessing_seconds = time.perf_counter() - t0
    print(f"Preprocessing (fit_transform) runtime: {preprocessing_seconds:.3f}s, shape {X_train.shape}")
    report["preprocessing_seconds"] = round(preprocessing_seconds, 3)
    report["matrix_shape"] = list(X_train.shape)

    section(f"2. ONE fit: LogisticRegression(solver='saga', penalty='l2', C=1.0, max_iter={MAX_ITER}, random_state=42, class_weight=None)")
    print("tol left at sklearn's own default (1e-4) -- same as every prior round; not re-chosen here.")
    model = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        tol=1e-4,
        max_iter=MAX_ITER,
        random_state=42,
        class_weight=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0
        convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]

    n_iter = int(model.n_iter_[0])
    hit_max_iter = n_iter >= MAX_ITER
    raised_warning = len(convergence_warnings) > 0
    # sklearn's own definition: converged iff it stopped (tol reached) BEFORE
    # exhausting max_iter, i.e. no ConvergenceWarning was raised.
    converged = not raised_warning

    train_log_loss = log_loss(y_train, model.predict_proba(X_train))

    print(f"fit_seconds={fit_seconds:.3f}")
    print(f"n_iter_={n_iter} (max_iter={MAX_ITER}, hit_cap={hit_max_iter})")
    print(f"ConvergenceWarning raised: {raised_warning}")
    if convergence_warnings:
        print(f"  message: {convergence_warnings[0]}")
    print(f"Solver-reported convergence status: {'CONVERGED (stopped before max_iter, tol satisfied)' if converged else 'DID NOT CONVERGE (exhausted max_iter)'}")
    print(f"Final train log-loss (diagnostic only, not a claim of convergence by itself): {train_log_loss:.6f}")

    report["config"] = {
        "solver": "saga", "penalty": "l2", "C": 1.0, "tol": 1e-4,
        "max_iter": MAX_ITER, "random_state": 42, "class_weight": None,
    }
    report["fit_seconds"] = round(fit_seconds, 3)
    report["n_iter"] = n_iter
    report["hit_max_iter_cap"] = hit_max_iter
    report["convergence_warning_raised"] = raised_warning
    report["converged"] = converged
    report["final_train_log_loss"] = round(float(train_log_loss), 6)

    out_path = DOCS_DIR / "_stage7_final_saga_verification_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
