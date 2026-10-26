"""Stage 9 — real, full-scale five-fold feasibility experiment.

Approved scope: ONE fixed Random Forest configuration (the Stage 7
reference config, unchanged) executed across all 5 of the already-built,
deterministic expanding-window chronological CV folds, AT FULL SCALE (the
complete `train` partition's own folds -- not the bounded 15,000-row
probe from the prior design round). This is a resource-feasibility
measurement, NOT a hyperparameter search: no candidate varies, no result
here selects or alters any configuration.

Invoked ONCE PER FOLD (`--fold_index 0..4`), each invocation under its own
`/usr/bin/time -l` process and the harness's own 600-second wall-clock
timeout, so each fold's peak memory is measured cleanly (no cross-fold
high-water-mark contamination -- the same discipline used throughout this
project since Stage 5) and a timeout/failure on one fold stops the
experiment rather than silently continuing to the next.

TRAIN-ONLY. Never reads `validation` or `final_test`. The approved Stage 5
preprocessing pipeline (`src/preprocessing.py`) is read, never modified;
each fold refits it independently on that fold's own training rows only
(`src/hyperparameter_tuning.fit_transform_fold`, unmodified from the prior
Stage 9 design round).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.preprocessing import load_included_columns, load_partition_df  # noqa: E402
from src.model_selection import align_labels_to_partition_df, build_bounded_random_forest, evaluate_predictions, load_labels  # noqa: E402
from src.hyperparameter_tuning import build_expanding_window_folds, fit_transform_fold, verify_fold  # noqa: E402
from src.experiment_tracking import compute_feature_schema_fingerprint, track_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

N_SPLITS = 5
SEARCH_ID = "stage9-five-fold-full-scale-feasibility"
RF_REFERENCE_CONFIG = {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 20, "random_state": 42}


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def fingerprint_ids(ids: pd.Series) -> str:
    """sha256 over the SORTED TransactionID list -- a partition fingerprint
    that changes if, and only if, the actual set of rows changes (order-
    independent, content-sensitive)."""
    joined = "\x1f".join(str(i) for i in sorted(ids.tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def main(fold_index: int) -> dict:
    report: dict = {"fold_index": fold_index, "n_splits": N_SPLITS}

    with track_run(
        9, "stage9_fivefold_feasibility_fold", f"Real, full-scale feasibility fold {fold_index} of {N_SPLITS} -- fixed Stage 7 reference RF config, resource measurement only.",
        search_id=SEARCH_ID, candidate_index=0, fold_index=fold_index,
    ) as record:
        section("0. final_test refusal re-check")
        assignment = pd.read_csv(ASSIGNMENT_PATH)
        try:
            get_partition(assignment, ["train", "final_test"])
            report["final_test_refused"] = False
        except FinalTestAccessError as exc:
            print(f"Correctly refused: {exc}")
            report["final_test_refused"] = True

        section("1. Load the COMPLETE approved train partition (all feature columns + TransactionDT + isFraud)")
        columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
        numeric_columns = columns["numeric"]
        categorical_columns = columns["categorical"]
        all_cols = numeric_columns + categorical_columns

        train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols)
        labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
        train_df = align_labels_to_partition_df(train_df, labels)

        # TransactionDT is itself an "include"-status register column (the
        # register documents it as a legitimate feature, not just a sort
        # key), so it is ALREADY present in `all_cols`/`train_df` via
        # `load_partition_df` above -- no separate merge is needed or
        # correct here (a second read-and-merge would collide on the
        # column name and get silently suffixed by pandas).
        if "TransactionDT" not in train_df.columns:
            raise ValueError("TransactionDT missing from the loaded partition -- expected it to be a register-included column.")
        print(f"Full train partition (features incl. TransactionDT + isFraud): {len(train_df)} rows (expected 438,125)")
        report["train_n_rows"] = len(train_df)

        section(f"2. Rebuild the 5 deterministic expanding-window folds fresh, re-verify fold {fold_index}")
        df_sorted, folds = build_expanding_window_folds(train_df, N_SPLITS)
        if fold_index >= len(folds):
            raise ValueError(f"fold_index {fold_index} out of range (n_splits={N_SPLITS})")
        train_idx, val_idx = folds[fold_index]
        fold_check = verify_fold(df_sorted, train_idx, val_idx)
        print(f"Fold {fold_index} re-verification: {fold_check}")
        report["fold_verification"] = fold_check
        if not fold_check["all_checks_passed"]:
            raise RuntimeError(f"Fold {fold_index} failed re-verification -- stopping rather than proceeding: {fold_check}")

        train_fp = fingerprint_ids(df_sorted.loc[train_idx, "TransactionID"])
        val_fp = fingerprint_ids(df_sorted.loc[val_idx, "TransactionID"])
        report["train_fold_row_fingerprint"] = train_fp
        report["val_fold_row_fingerprint"] = val_fp
        record["partitions"] = {
            "train_fold": {"id": f"train fold {fold_index} (expanding window)", "n_rows": len(train_idx), "row_fingerprint": train_fp},
            "val_fold": {"id": f"val fold {fold_index} (expanding window)", "n_rows": len(val_idx), "row_fingerprint": val_fp},
        }

        section("3. Fit preprocessing INDEPENDENTLY on this fold's own training rows (never train-wide)")
        t0 = time.perf_counter()
        fitted = fit_transform_fold(df_sorted, train_idx, val_idx, numeric_columns, categorical_columns)
        preprocessing_seconds = time.perf_counter() - t0
        print(f"preprocessing_seconds={preprocessing_seconds:.3f}  fitted_feature_count={fitted['fitted_feature_count']}")
        report["preprocessing_seconds"] = round(preprocessing_seconds, 3)
        report["fitted_feature_count"] = fitted["fitted_feature_count"]
        report["numeric_with_indicator_count"] = len(fitted["numeric_with_indicator"])
        record["feature_count"] = fitted["fitted_feature_count"]
        record["feature_schema_fingerprint"] = compute_feature_schema_fingerprint(
            fitted["numeric_with_indicator"] + fitted["numeric_without_indicator"] + categorical_columns
        )
        record["preprocessing_config"] = {"pipeline": "fit_transform_fold (per-fold-refit, Stage 9 design, unmodified)"}

        section("4. Fit the FIXED Stage 7 reference Random Forest -- same config every fold, no variation")
        model = build_bounded_random_forest()
        print(f"RandomForestClassifier({RF_REFERENCE_CONFIG}, n_jobs=-1)")
        t0 = time.perf_counter()
        model.fit(fitted["X_train"], fitted["y_train"])
        fit_seconds = time.perf_counter() - t0
        print(f"fit_seconds={fit_seconds:.3f}")
        report["fit_seconds"] = round(fit_seconds, 3)
        record["model_family"] = "RandomForestClassifier"
        record["hyperparameters"] = RF_REFERENCE_CONFIG
        record["random_seed"] = RF_REFERENCE_CONFIG["random_state"]
        record["training_time_seconds"] = round(fit_seconds, 3)

        section("5. Evaluate on THIS FOLD'S OWN held-out validation rows ONLY (never the external `validation` partition)")
        t0 = time.perf_counter()
        scores = model.predict_proba(fitted["X_val"])[:, 1]
        predict_seconds = time.perf_counter() - t0
        metrics = evaluate_predictions(fitted["y_val"], scores)
        train_prevalence = float(fitted["y_train"].mean())
        val_prevalence = float(fitted["y_val"].mean())
        print(f"predict_seconds={predict_seconds:.3f}")
        print(f"train_prevalence={train_prevalence:.5f}  val_prevalence={val_prevalence:.5f}")
        print(f"AP={metrics['average_precision']}  ROC-AUC={metrics['roc_auc']}")
        report["predict_seconds"] = round(predict_seconds, 3)
        report["train_prevalence"] = round(train_prevalence, 5)
        report["val_prevalence"] = round(val_prevalence, 5)
        report["metrics"] = metrics
        record["inference_time_seconds"] = round(predict_seconds, 3)
        record["metrics"] = metrics
        record["validation_consulted"] = False
        record["validation_purpose"] = "evaluated on this fold's OWN held-out rows inside train only -- the external validation partition was never read"

        total_seconds = preprocessing_seconds + fit_seconds + predict_seconds
        report["total_fold_seconds"] = round(total_seconds, 3)
        print(f"\ntotal_fold_seconds={total_seconds:.3f}")

        out_path = DOCS_DIR / f"_stage9_fivefold_feasibility_fold{fold_index}_raw.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull structured report written to {out_path}")
        record["evidence_paths"] = [f"docs/_stage9_fivefold_feasibility_fold{fold_index}_raw.json"]

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_index", type=int, required=True, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    main(args.fold_index)
