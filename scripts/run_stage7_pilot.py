"""Stage 7 — bounded chronological pilot (corrected linear-model branch).

A DIAGNOSTIC pilot, not a final model-selection result: fits on a small,
genuinely-earliest-in-time slice of `train` (never a random sample) to
check whether the corrected preprocessing (Stage 6's collapsed
missingness-indicator representation, unscaled binary indicators, scaled
continuous values) plus a different solver (saga, not liblinear) actually
converges in reasonable time -- before ever attempting the full
438,125-row fit again.

Partition discipline: `train`-only for fitting (via `get_partition()`),
real `validation` for scoring (never redefined, never a fresh split) --
`final_test` is never requested with `allow_final_test=True` anywhere in
this script.

Run under an OS-level timeout, not an in-process one:
    timeout 180 .venv/bin/python3 scripts/run_stage7_pilot.py     # if GNU coreutils available
or, inside this harness, via the tool's own process-level timeout, which
can actually kill a stuck process (Python's signal.alarm cannot interrupt
a liblinear/saga C loop -- see docs/HANDOFF.md "Stage 7 -- diagnosed,
unresolved" for why this matters here specifically).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.feature_engineering import collapsed_indicator_representatives, find_duplicate_indicator_groups  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    load_included_columns,
    load_partition_df,
    measure_train_missingness,
    select_indicator_columns,
)
from src.model_selection import (  # noqa: E402
    RANDOM_STATE,
    build_chronological_pilot_subset,
    build_scaled_pipeline_with_collapsed_indicators,
    evaluate_predictions,
    load_labels,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

PILOT_N_ROWS = 15_000  # within the requested 10,000-20,000 range, per instruction


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> dict:
    report: dict = {}

    section("1. Column scope (unchanged from Stage 5/7 -- same register-driven logic)")
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical
    print(f"numeric_no_indicator={len(numeric_no_indicator)} numeric_with_indicator={len(numeric_with_indicator)} categorical={len(categorical)}")

    section(f"2. Chronological pilot subset -- earliest {PILOT_N_ROWS} TRAIN rows (never random, never validation/final_test)")
    t0 = time.perf_counter()
    pilot_df = build_chronological_pilot_subset(ASSIGNMENT_PATH, TRAIN_TXN, all_cols, n_rows=PILOT_N_ROWS)
    load_s = time.perf_counter() - t0
    prevalence = pilot_df["isFraud"].mean()
    print(f"Pilot subset: {len(pilot_df)} rows, fraud prevalence {prevalence:.4%}, loaded in {load_s:.2f}s")
    print(f"TransactionDT range: {pilot_df['TransactionDT'].min()} to {pilot_df['TransactionDT'].max()}")
    report["pilot_subset"] = {
        "n_rows_requested": PILOT_N_ROWS,
        "n_rows_actual": len(pilot_df),
        "fraud_prevalence": round(float(prevalence), 5),
        "n_fraud_rows": int(pilot_df["isFraud"].sum()),
        "dt_min": int(pilot_df["TransactionDT"].min()),
        "dt_max": int(pilot_df["TransactionDT"].max()),
        "load_seconds": round(load_s, 3),
    }

    section("2b. Confirm final_test is still refused (same guardrail, exercised from this script too)")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    try:
        get_partition(assignment, ["train", "final_test"])
        report["final_test_refused"] = False
    except FinalTestAccessError as exc:
        print(f"Correctly refused: {exc}")
        report["final_test_refused"] = True

    section("3. Recompute indicator-duplicate groups FRESH on the PILOT subset (not reused from the full 438K-row figure)")
    print("Caveat, stated explicitly: a representative-column pattern found on 15,000 rows is diagnostic "
          "only and may differ from the full train set's pattern (438,125 rows) -- not assumed identical.")
    groups = find_duplicate_indicator_groups(pilot_df, numeric_with_indicator)
    representatives = collapsed_indicator_representatives(groups)
    print(f"{len(numeric_with_indicator)} indicator columns -> {len(representatives)} unique patterns on this pilot subset "
          f"(full-train figure from Stage 5/6/7: 26)")
    report["indicator_collapse_on_pilot"] = {
        "n_indicator_columns": len(numeric_with_indicator),
        "n_unique_patterns_on_pilot_subset": len(representatives),
        "n_unique_patterns_on_full_train_previously_measured": 26,
    }

    section("4. Corrected experimental pipeline: collapsed indicators (unscaled) + scaled continuous values + deterministic one-hot")
    ct = build_scaled_pipeline_with_collapsed_indicators(numeric_no_indicator, numeric_with_indicator, categorical, representatives)
    t0 = time.perf_counter()
    X_train = ct.fit_transform(pilot_df[all_cols])
    fit_transform_s = time.perf_counter() - t0
    print(f"Pilot pipeline fit_transform: {fit_transform_s:.2f}s, shape {X_train.shape}")
    report["pilot_pipeline"] = {
        "fit_transform_seconds": round(fit_transform_s, 3),
        "output_shape": list(X_train.shape),
        "is_sparse": hasattr(X_train, "nnz"),
        "density": round(X_train.nnz / (X_train.shape[0] * X_train.shape[1]), 4) if hasattr(X_train, "nnz") else None,
    }

    section("5. Transform REAL validation (unchanged partition) with the pilot-fitted pipeline")
    val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    y_val = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")
    t0 = time.perf_counter()
    X_val = ct.transform(val_df[all_cols])
    val_transform_s = time.perf_counter() - t0
    print(f"Validation transform: {val_transform_s:.2f}s, shape {X_val.shape}")
    report["validation_transform_seconds"] = round(val_transform_s, 3)

    section("6. LogisticRegression(solver='saga') -- two class_weight configs, BOTH recorded with rationale")
    configs = [
        {
            "name": "saga_balanced",
            "class_weight": "balanced",
            "rationale": "Preserves the prior project decision (used throughout Stages 5-7 so far) -- "
                         "tested, not silently kept or silently dropped.",
        },
        {
            "name": "saga_unweighted",
            "class_weight": None,
            "rationale": "External-repo finding (Stage 7 architecture-reconciliation audit): one verified "
                         "IEEE-CIS repo reported unweighted LightGBM beating its own weighted/SMOTE variants. "
                         "Tested here as a documented alternative, not assumed better -- solver is saga in "
                         "both arms, so this isolates the class_weight question specifically.",
        },
    ]

    results = {}
    for cfg in configs:
        print(f"\n--- {cfg['name']} (class_weight={cfg['class_weight']!r}) ---")
        print(f"Rationale: {cfg['rationale']}")
        model = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,  # default, NOT tuned -- Stage 9's job; stated explicitly, not silently assumed optimal
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight=cfg["class_weight"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            t0 = time.perf_counter()
            model.fit(X_train, pilot_df["isFraud"].to_numpy())
            fit_s = time.perf_counter() - t0
            convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]

        converged = len(convergence_warnings) == 0
        n_iter = int(model.n_iter_[0])
        print(f"fit_seconds={fit_s:.3f}  n_iter_={n_iter} (max_iter=1000)  converged={converged}")
        if convergence_warnings:
            print(f"ConvergenceWarning raised: {convergence_warnings[0]}")

        scores = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
        print(f"Validation AP={metrics['average_precision']} ROC-AUC={metrics['roc_auc']} "
              f"(DIAGNOSTIC ONLY -- model fit on {len(pilot_df)} rows, not a final-model estimate)")

        results[cfg["name"]] = {
            "class_weight": str(cfg["class_weight"]),
            "rationale": cfg["rationale"],
            "solver": "saga",
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 1000,
            "fit_seconds": round(fit_s, 3),
            "n_iter": n_iter,
            "converged": converged,
            "convergence_warning": convergence_warnings[0] if convergence_warnings else None,
            "validation_average_precision": metrics["average_precision"],
            "validation_roc_auc": metrics["roc_auc"],
        }

    report["saga_configs"] = results

    out_path = DOCS_DIR / "_stage7_pilot_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
