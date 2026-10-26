"""Stage 11 — Explainability, real analysis.

TRAIN-ONLY throughout. Never reads `validation` or `final_test`. Explains
the FROZEN Stage 9/10 Random Forest configuration, refit on the complete
`train` partition (identical to how Stage 10 obtained the model whose
AP=0.42784 was reported) -- SHAP explanations describe THAT model.

A SEPARATE, clearly-labeled AUXILIARY fit (same frozen hyperparameters,
on Stage 9's fold-0 training rows only) is used ONLY for the permutation-
importance cross-check, so that cross-check has a genuinely held-out
chronological evaluation set (fold-0's own validation rows) rather than
scoring on rows the explained model was itself fit on -- this auxiliary
fit's own predictive numbers are NOT reported as a Stage 10 result and
must never be confused with the real, approved final_test evaluation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import FinalTestAccessError, get_partition  # noqa: E402
from src.preprocessing import build_preprocessing_pipeline, load_included_columns, load_partition_df, measure_train_missingness, select_indicator_columns, DEFAULT_INDICATOR_THRESHOLD  # noqa: E402
from src.model_selection import align_labels_to_partition_df, evaluate_predictions, find_highly_correlated_value_groups, load_labels  # noqa: E402
from src.hyperparameter_tuning import build_expanding_window_folds  # noqa: E402
from src.model_evaluation import FROZEN_RF_CONFIG, build_frozen_model, verify_frozen_config_unchanged  # noqa: E402
from src.explainability import (  # noqa: E402
    bounded_permutation_importance,
    build_feature_lineage,
    build_tree_explainer,
    compute_shap_values,
    correlated_group_permutation_importance,
    explanation_sample_fingerprint,
    global_importance_ranking,
    perturb_numeric_features_and_recompute,
    select_explanation_sample,
    verify_shap_additivity,
)
from src.experiment_tracking import track_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
NOTEBOOKS_DIR = ROOT / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"

SEARCH_ID = "stage11-explainability"
N_FRAUD_SAMPLE = 100
N_NON_FRAUD_SAMPLE = 100
FEASIBILITY_PROBE_N = 10
TOP_K_PERMUTATION_FEATURES = 25


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}", flush=True)


def main() -> dict:
    report: dict = {"search_id": SEARCH_ID}

    with track_run(11, "stage11_explainability_main", "SHAP + permutation-importance explainability analysis on the frozen Stage 9/10 RF, train-only.", search_id=SEARCH_ID) as record:
        section("0. final_test refusal + frozen-config integrity")
        assignment = pd.read_csv(ASSIGNMENT_PATH)
        try:
            get_partition(assignment, ["train", "final_test"])
            report["final_test_refused"] = False
        except FinalTestAccessError as exc:
            print(f"Correctly refused: {exc}")
            report["final_test_refused"] = True
        assert verify_frozen_config_unchanged()

        section("1. Load the COMPLETE train partition and fit the EXACT frozen model (same as Stage 10)")
        columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
        numeric_columns, categorical_columns = columns["numeric"], columns["categorical"]
        missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, numeric_columns)
        indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
        all_cols = indicator_split["without_indicator"] + indicator_split["with_indicator"] + categorical_columns

        train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols)
        labels = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
        train_df = align_labels_to_partition_df(train_df, labels)
        print(f"train: {len(train_df)} rows")

        ct = build_preprocessing_pipeline(indicator_split["without_indicator"], indicator_split["with_indicator"], categorical_columns)
        t0 = time.perf_counter()
        X_train = ct.fit_transform(train_df[all_cols])
        preprocessing_seconds = time.perf_counter() - t0
        feature_names = list(ct.get_feature_names_out())
        print(f"preprocessing_seconds={preprocessing_seconds:.3f}  fitted_feature_count={X_train.shape[1]}")

        model = build_frozen_model()
        t0 = time.perf_counter()
        model.fit(X_train, train_df["isFraud"].to_numpy())
        fit_seconds = time.perf_counter() - t0
        print(f"fit_seconds={fit_seconds:.3f}")
        record["model_family"] = "RandomForestClassifier"
        record["hyperparameters"] = FROZEN_RF_CONFIG
        record["feature_count"] = X_train.shape[1]
        record["training_time_seconds"] = round(fit_seconds, 3)

        section("2. Deterministic, stratified, train-only explanation sample")
        sample_df = select_explanation_sample(train_df, n_fraud=N_FRAUD_SAMPLE, n_non_fraud=N_NON_FRAUD_SAMPLE, random_state=42)
        sample_fp = explanation_sample_fingerprint(sample_df)
        print(f"sample: {len(sample_df)} rows ({N_FRAUD_SAMPLE} fraud + {N_NON_FRAUD_SAMPLE} non-fraud), fingerprint={sample_fp[:16]}...")
        X_sample = ct.transform(sample_df[all_cols])
        X_sample_dense = X_sample.toarray() if hasattr(X_sample, "toarray") else X_sample
        report["sample_n_fraud"] = N_FRAUD_SAMPLE
        report["sample_n_non_fraud"] = N_NON_FRAUD_SAMPLE
        report["sample_fingerprint"] = sample_fp
        report["sample_selection_procedure"] = "Stratified (100 fraud + 100 non-fraud), sorted by TransactionID, sampled with random_state=42, from the train partition only -- selected before any SHAP value was computed, never chosen for how a plot looked."

        section("3. SHAP feasibility probe (small subset) before committing to the full sample")
        explainer = build_tree_explainer(model)
        t0 = time.perf_counter()
        probe_result = compute_shap_values(explainer, X_sample_dense[:FEASIBILITY_PROBE_N])
        probe_seconds = time.perf_counter() - t0
        per_row_estimate = probe_seconds / FEASIBILITY_PROBE_N
        full_estimate = per_row_estimate * len(sample_df)
        print(f"Feasibility probe: {FEASIBILITY_PROBE_N} rows in {probe_seconds:.3f}s ({per_row_estimate:.4f}s/row) -- estimated {full_estimate:.1f}s for the full {len(sample_df)}-row sample")
        report["feasibility_probe"] = {"n_rows": FEASIBILITY_PROBE_N, "seconds": round(probe_seconds, 3), "estimated_full_sample_seconds": round(full_estimate, 1)}

        section("4. Full SHAP computation on the sample + additivity verification")
        t0 = time.perf_counter()
        shap_result = compute_shap_values(explainer, X_sample_dense)
        shap_seconds = time.perf_counter() - t0
        print(f"SHAP computation: {shap_seconds:.3f}s for {len(sample_df)} rows. Output space: {shap_result['output_space']}")
        model_output = model.predict_proba(X_sample_dense)[:, 1]
        additivity = verify_shap_additivity(shap_result["shap_values"], shap_result["expected_value"], model_output, tolerance=1e-3)
        print(f"Additivity check: {additivity}")
        report["shap_seconds"] = round(shap_seconds, 3)
        report["output_space"] = shap_result["output_space"]
        report["additivity"] = additivity
        assert additivity["passed"], "SHAP additivity check failed -- stopping before reporting any SHAP-derived claim."

        section("5. Global importance ranking + feature lineage (transformed -> original)")
        ranking = global_importance_ranking(shap_result["shap_values"], feature_names)
        lineage = build_feature_lineage(feature_names, categorical_columns)
        ranking_with_lineage = ranking.merge(lineage, left_on="feature", right_on="transformed_feature")
        top20 = ranking_with_lineage.head(20)
        print(top20[["feature", "original_column", "feature_type", "mean_abs_shap"]].to_string(index=False))
        report["global_ranking_top20"] = top20[["feature", "original_column", "feature_type", "category", "mean_abs_shap"]].to_dict("records")

        # Aggregate by ORIGINAL column (summing one-hot categories, e.g. all
        # ProductCD_* columns collapse to one "ProductCD" importance figure)
        by_original = ranking_with_lineage.groupby("original_column", dropna=True)["mean_abs_shap"].sum().sort_values(ascending=False)
        print("\nTop 15 by ORIGINAL column (one-hot categories summed):")
        print(by_original.head(15).to_string())
        report["global_ranking_by_original_column_top15"] = by_original.head(15).to_dict()

        section("6. Global summary plot (saved, not displayed interactively)")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        top15_plot = by_original.head(15).iloc[::-1]
        ax.barh(top15_plot.index.astype(str), top15_plot.values, color="#4C72B0")
        ax.set_xlabel("Sum of mean |SHAP value| across one-hot categories (class-1 / fraud-probability output)")
        ax.set_title("Stage 11 — Global feature importance (top 15 original columns)\nFrozen RandomForestClassifier, train-only, n=200 stratified sample")
        fig.tight_layout()
        plot_path = FIGURES_DIR / "stage11_global_shap_importance.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"Saved: {plot_path}")
        report["global_plot_path"] = str(plot_path.relative_to(ROOT))

        section("7. Local explanations -- a handful of specific transactions")
        fraud_mask = sample_df["isFraud"].to_numpy() == 1
        proba = model_output
        # Fixed, pre-stated selection rule (not cherry-picked for a nice story):
        # the highest-confidence correctly-flagged fraud case, the lowest-
        # confidence MISSED fraud case (false negative), and one borderline
        # (~0.5) case -- three different, honestly-labeled situations.
        fraud_idx = np.where(fraud_mask)[0]
        local_cases = {}
        tp_idx = fraud_idx[np.argmax(proba[fraud_idx])]
        fn_idx = fraud_idx[np.argmin(proba[fraud_idx])]
        borderline_idx = int(np.argmin(np.abs(proba - 0.5)))
        for label, idx in (("highest_confidence_true_positive", tp_idx), ("missed_fraud_lowest_score", fn_idx), ("borderline_case", borderline_idx)):
            row_shap = shap_result["shap_values"][idx]
            top_contributors = pd.DataFrame({"feature": feature_names, "shap_value": row_shap}).merge(lineage, left_on="feature", right_on="transformed_feature")
            top_contributors = top_contributors.reindex(top_contributors["shap_value"].abs().sort_values(ascending=False).index).head(10)
            local_cases[label] = {
                "transaction_id": int(sample_df.iloc[idx]["TransactionID"]), "predicted_proba": float(proba[idx]),
                "actual_label": int(sample_df.iloc[idx]["isFraud"]),
                "top_10_contributors": top_contributors[["original_column", "feature_type", "category", "shap_value"]].to_dict("records"),
            }
            print(f"\n{label}: TransactionID={local_cases[label]['transaction_id']} proba={proba[idx]:.4f} actual={int(sample_df.iloc[idx]['isFraud'])}")
            print(top_contributors[["original_column", "feature_type", "shap_value"]].head(5).to_string(index=False))
        report["local_explanations"] = local_cases

        section("8. Stability check: small, realistic numeric-only perturbation on the local examples")
        numeric_feature_mask = [i for i, f in enumerate(feature_names) if f.startswith("numeric_plain__") or (f.startswith("numeric_indicator__") and "missingindicator" not in f)]
        stability_results = {}
        for label, idx in (("highest_confidence_true_positive", tp_idx), ("borderline_case", borderline_idx)):
            stability_results[label] = perturb_numeric_features_and_recompute(
                model, explainer, X_sample_dense[idx], numeric_col_indices=numeric_feature_mask, relative_noise_scale=0.01, n_trials=10, random_state=42,
            )
            print(f"{label}: {stability_results[label]}")
        report["stability_checks"] = stability_results

        section("9. Correlated V-column groups (reused from Stage 7's own finder)")
        v_columns = [c for c in indicator_split["without_indicator"] + indicator_split["with_indicator"] if c.startswith("V")]
        v_values = train_df[v_columns].fillna(train_df[v_columns].median()).to_numpy()
        groups = find_highly_correlated_value_groups(v_values, v_columns, threshold=0.99)
        print(f"Found {len(groups)} correlated V-column groups (threshold 0.99) among {len(v_columns)} V-columns.")
        report["n_correlated_v_groups"] = len(groups)
        report["correlated_v_group_sizes"] = {k: len(v) for k, v in list(groups.items())[:10]}

    section("10. Permutation-importance cross-check -- AUXILIARY fit on Stage 9's fold-0, held out on fold-0's own validation rows")
    with track_run(11, "stage11_permutation_importance_auxiliary_fit", "Auxiliary frozen-config fit on Stage 9 fold-0 train rows, for permutation-importance cross-check ONLY -- not a Stage 10 result.", search_id=SEARCH_ID) as aux_record:
        df_sorted, folds = build_expanding_window_folds(train_df, 5)
        fold0_train_idx, fold0_val_idx = folds[0]
        aux_model = build_frozen_model()
        X_fold0_train = ct.fit_transform(df_sorted.loc[fold0_train_idx, all_cols])
        X_fold0_val = ct.transform(df_sorted.loc[fold0_val_idx, all_cols])
        y_fold0_train = df_sorted.loc[fold0_train_idx, "isFraud"].to_numpy()
        y_fold0_val = df_sorted.loc[fold0_val_idx, "isFraud"].to_numpy()
        aux_model.fit(X_fold0_train, y_fold0_train)
        aux_metrics = evaluate_predictions(y_fold0_val, aux_model.predict_proba(X_fold0_val)[:, 1])
        print(f"Auxiliary fit (fold-0 train, {len(fold0_train_idx)} rows) -- held-out AP on fold-0 validation ({len(fold0_val_idx)} rows): {aux_metrics['average_precision']}")
        aux_record["model_family"] = "RandomForestClassifier"
        aux_record["metrics"] = aux_metrics
        aux_record["validation_consulted"] = False

        top_feature_names = ranking["feature"].head(TOP_K_PERMUTATION_FEATURES).tolist()
        fold0_feature_names = list(ct.get_feature_names_out())
        top_feature_indices = [fold0_feature_names.index(f) for f in top_feature_names if f in fold0_feature_names]
        X_fold0_val_dense = X_fold0_val.toarray() if hasattr(X_fold0_val, "toarray") else X_fold0_val
        perm_result = bounded_permutation_importance(aux_model, X_fold0_val_dense, y_fold0_val, feature_indices=top_feature_indices, n_repeats=5, random_state=42)
        print(f"Permutation importance (top {len(top_feature_indices)} SHAP-ranked features, fold-0 val, baseline AP={perm_result['baseline_score']:.5f}):")
        for i, feat_idx in enumerate(top_feature_indices[:10]):
            print(f"  {fold0_feature_names[feat_idx]}: mean_AP_drop={perm_result['importances_mean'][i]:.5f} (+/- {perm_result['importances_std'][i]:.5f})")

        v_group_indices = {k: [fold0_feature_names.index(f"numeric_indicator__{c}") for c in v if f"numeric_indicator__{c}" in fold0_feature_names] for k, v in list(groups.items())[:3] if len(v) > 1}
        v_group_indices = {k: v for k, v in v_group_indices.items() if v}
        correlated_result = correlated_group_permutation_importance(aux_model, X_fold0_val_dense, y_fold0_val, v_group_indices, n_repeats=5, random_state=42) if v_group_indices else {}
        print(f"\nCorrelated-group permutation importance (whole group permuted together): {correlated_result}")

    report["auxiliary_fit_metrics"] = aux_metrics
    report["permutation_importance_top_features"] = {fold0_feature_names[i]: {"mean_ap_drop": perm_result["importances_mean"][j], "std_ap_drop": perm_result["importances_std"][j]} for j, i in enumerate(top_feature_indices)}
    report["correlated_group_permutation_importance"] = correlated_result

    out_path = DOCS_DIR / "_stage11_explainability_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
