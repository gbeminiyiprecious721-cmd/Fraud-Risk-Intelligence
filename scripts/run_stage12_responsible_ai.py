"""Stage 12 -- Responsible AI assessment, real analysis.

TRAIN-ONLY throughout. Never reads `validation` or `final_test`. Reuses
the exact same mechanism Stage 11 used for its permutation-importance
cross-check: an AUXILIARY fit of the frozen Stage 9/10 Random Forest
configuration on Stage 9's own fold-0 training rows, scored on fold-0's
own held-out validation rows -- a genuinely held-out, train-only
evaluation set. This auxiliary fit's own predictive numbers are NOT a
Stage 10 result and must never be reported as one.

Subgroup assessment uses ProductCD, card4, and card6 -- real transaction
and payment-method fields, NOT verified demographic or protected-
characteristic attributes. This dataset contains no such attribute.

Robustness checks distinguish INPUT VALIDATION (does the fitted
pipeline + model accept an unusual row without crashing, and return a
finite probability in [0, 1]?) from PREDICTIVE ROBUSTNESS (does AP hold
up moving forward in time?) -- the latter is read from ALREADY-SAVED
Stage 9 registry evidence, not recomputed here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.preprocessing import build_preprocessing_pipeline, load_included_columns, load_partition_df, measure_train_missingness, select_indicator_columns, DEFAULT_INDICATOR_THRESHOLD  # noqa: E402
from src.model_selection import align_labels_to_partition_df, evaluate_predictions, load_labels  # noqa: E402
from src.hyperparameter_tuning import build_expanding_window_folds  # noqa: E402
from src.model_evaluation import build_frozen_model, segment_report, verify_frozen_config_unchanged  # noqa: E402
from src.responsible_ai import (  # noqa: E402
    build_edge_case_rows,
    malformed_dtype_case,
    run_edge_case_predictions,
    schema_mismatch_case,
    segment_report_with_lift,
)
from src.experiment_tracking import query_registry, track_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"
REGISTRY_PATH = DOCS_DIR / "experiment_registry.jsonl"

SEARCH_ID = "stage12-responsible-ai"
SUBGROUP_COLUMNS = ["ProductCD", "card4", "card6"]


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}", flush=True)


def main() -> dict:
    report: dict = {"search_id": SEARCH_ID}

    with track_run(12, "stage12_responsible_ai_main", "Subgroup/robustness Responsible AI assessment on an auxiliary fold-0 fit of the frozen RF, train-only.", search_id=SEARCH_ID) as record:
        section("0. final_test refusal + frozen-config integrity")
        assignment = pd.read_csv(ASSIGNMENT_PATH)
        try:
            get_partition(assignment, ["train", "final_test"])
            report["final_test_refused"] = False
        except FinalTestAccessError as exc:
            print(f"Correctly refused: {exc}")
            report["final_test_refused"] = True
        assert verify_frozen_config_unchanged()

        section("1. Load the complete train partition (needed to rebuild Stage 9's fold-0 split exactly)")
        columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
        numeric_columns, categorical_columns = columns["numeric"], columns["categorical"]
        missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, numeric_columns)
        indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
        all_cols = indicator_split["without_indicator"] + indicator_split["with_indicator"] + categorical_columns

        train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols)
        labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
        train_df = align_labels_to_partition_df(train_df, labels)
        print(f"train: {len(train_df)} rows")

        section("2. AUXILIARY frozen-config fit on Stage 9's fold-0 train rows, scored on fold-0's own validation rows")
        df_sorted, folds = build_expanding_window_folds(train_df, 5)
        fold0_train_idx, fold0_val_idx = folds[0]
        ct = build_preprocessing_pipeline(indicator_split["without_indicator"], indicator_split["with_indicator"], categorical_columns)
        t0 = time.perf_counter()
        X_fold0_train = ct.fit_transform(df_sorted.loc[fold0_train_idx, all_cols])
        preprocessing_seconds = time.perf_counter() - t0
        X_fold0_val = ct.transform(df_sorted.loc[fold0_val_idx, all_cols])
        y_fold0_train = df_sorted.loc[fold0_train_idx, "isFraud"].to_numpy()
        y_fold0_val = df_sorted.loc[fold0_val_idx, "isFraud"].to_numpy()

        aux_model = build_frozen_model()
        t0 = time.perf_counter()
        aux_model.fit(X_fold0_train, y_fold0_train)
        fit_seconds = time.perf_counter() - t0
        X_fold0_val_dense = X_fold0_val.toarray() if hasattr(X_fold0_val, "toarray") else X_fold0_val
        proba_val = aux_model.predict_proba(X_fold0_val_dense)[:, 1]
        aux_metrics = evaluate_predictions(y_fold0_val, proba_val)
        print(f"Auxiliary fit (fold-0 train, {len(fold0_train_idx)} rows, preprocessing {preprocessing_seconds:.3f}s, fit {fit_seconds:.3f}s) "
              f"-- held-out AP on fold-0 validation ({len(fold0_val_idx)} rows): {aux_metrics['average_precision']}")
        record["model_family"] = "RandomForestClassifier"
        record["metrics"] = aux_metrics
        record["validation_consulted"] = False
        report["auxiliary_fit_metrics"] = aux_metrics

        section("3. Subgroup assessment -- ProductCD / card4 / card6 (NOT demographic or protected-characteristic groups)")
        val_df = df_sorted.loc[fold0_val_idx]
        subgroup_reports: dict[str, dict] = {}
        for col in SUBGROUP_COLUMNS:
            labels_col = val_df[col].fillna("MISSING").to_numpy()
            seg_report = segment_report(y_fold0_val, proba_val, labels_col, min_samples=100, min_positives=10)
            flattened = segment_report_with_lift(seg_report)
            subgroup_reports[col] = flattened.to_dict("index")
            print(f"\n-- {col} --")
            print(flattened[["n_samples", "n_positive", "sufficient_sample", "prevalence", "average_precision", "ap_to_prevalence_lift"]].to_string())
        report["subgroup_reports"] = subgroup_reports

        section("4. Robustness -- input-validation edge cases (pipeline-level, NOT a predictive-robustness claim)")
        template_row = val_df.iloc[0]
        edge_cases = build_edge_case_rows(template_row, numeric_columns=indicator_split["without_indicator"] + indicator_split["with_indicator"])
        edge_results = run_edge_case_predictions(aux_model, ct, edge_cases, all_cols)
        for name, res in edge_results.items():
            print(f"  {name}: raised={res['raised']} proba={res['predicted_proba']}")
        report["edge_case_results"] = edge_results

        section("5. Robustness -- feature-schema mismatch (drop a required column entirely)")
        schema_result = schema_mismatch_case(ct, template_row.to_frame().T, all_cols, drop_column="TransactionAmt")
        print(f"  dropped TransactionAmt entirely: raised={schema_result['raised']} ({schema_result['exception_type']})")
        report["schema_mismatch_result"] = schema_result

        section("6. Robustness -- malformed dtype (a numeric field arriving as text)")
        malformed_row = malformed_dtype_case(template_row, "TransactionAmt")
        malformed_result = run_edge_case_predictions(aux_model, ct, {"malformed_transactionamt": malformed_row}, all_cols)
        print(f"  TransactionAmt='not_a_number': {malformed_result['malformed_transactionamt']}")
        report["malformed_dtype_result"] = malformed_result["malformed_transactionamt"]

        section("7. Robustness -- early-vs-late chronological performance, from ALREADY-SAVED Stage 9 evidence (no new fit)")
        stage9_fold_records = query_registry(REGISTRY_PATH, search_id="stage9-rf-randomized-search-20x5", candidate_index=4)
        fold_summary = []
        for rec in sorted(stage9_fold_records, key=lambda r: r.get("fold_index", -1)):
            if rec.get("fold_index") is None:
                continue
            m = rec["metrics"]
            vf = rec["partitions"]["val_fold"]
            fold_summary.append({
                "fold_index": rec["fold_index"], "val_dt_range": vf["dt_range"], "n_rows": vf["n_rows"],
                "n_positive": m["n_positive"], "prevalence": m["actual_prevalence"], "average_precision": m["average_precision"],
            })
        for row in fold_summary:
            print(f"  fold {row['fold_index']}: dt_range={row['val_dt_range']} AP={row['average_precision']:.5f} prevalence={row['prevalence']:.5f}")
        print("  Stage 10 final_test AP (chronologically AFTER every fold above): 0.42784")
        report["early_vs_late_chronological_evidence"] = fold_summary
        report["final_test_ap_for_comparison_only"] = 0.42784

    out_path = DOCS_DIR / "_stage12_responsible_ai_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
