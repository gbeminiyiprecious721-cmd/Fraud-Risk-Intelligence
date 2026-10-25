"""Stage 7 driver — baseline/candidate models + both ablations, real data.

Each `--run` value is meant to be invoked as a SEPARATE process (under
`/usr/bin/time -l`) for clean, isolated peak-memory measurement -- see
Stage 5's own documented lesson (docs/PREPROCESSING_REPORT.md §13.2) about
same-process high-water-mark contamination. Every run fits on `train`
ONLY and scores on `validation` ONLY; `final_test` is never requested.

Usage:
    .venv/bin/python3 scripts/run_stage7_model_selection.py --run baselines
    .venv/bin/python3 scripts/run_stage7_model_selection.py --run logreg_family
    .venv/bin/python3 scripts/run_stage7_model_selection.py --run tree_family
    .venv/bin/python3 scripts/run_stage7_model_selection.py --run feature_ablation_b
    .venv/bin/python3 scripts/run_stage7_model_selection.py --run indicator_ablation_collapsed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leakage_boundaries import get_partition  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    build_entity_history_train_and_validation,
    compute_interaction_and_amount_features,
    compute_temporal_phase_features,
    find_duplicate_indicator_groups,
    collapsed_indicator_representatives,
    fit_frequency_encoder_on_train,
    apply_frequency_encoder,
)
from src.preprocessing import (  # noqa: E402
    DEFAULT_INDICATOR_THRESHOLD,
    load_included_columns,
    load_partition_df,
    measure_train_missingness,
    select_indicator_columns,
)
from src.model_selection import (  # noqa: E402
    RANDOM_STATE,
    build_engineered_feature_block,
    build_scaled_pipeline_with_collapsed_indicators,
    build_scaled_preprocessing_pipeline,
    build_simple_baseline_pipeline,
    estimate_dense_memory_bytes,
    evaluate_predictions,
    hstack_sparse,
    load_labels,
    no_skill_baseline,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "ieee-fraud-detection"
TRAIN_TXN = DATA_DIR / "train_transaction.csv"
DOCS_DIR = ROOT / "docs"
REGISTER_PATH = DOCS_DIR / "feature_availability_register.csv"
ASSIGNMENT_PATH = DOCS_DIR / "_stage4_split_assignment.csv"
RESULTS_DIR = DOCS_DIR
RESULTS_DIR.mkdir(exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def _column_scope():
    columns = load_included_columns(REGISTER_PATH, TRAIN_TXN)
    missingness = measure_train_missingness(ASSIGNMENT_PATH, TRAIN_TXN, columns["numeric"])
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    return columns, indicator_split


def _timed_fit(model, X, y):
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def _timed_predict_proba(model, X):
    t0 = time.perf_counter()
    scores = model.predict_proba(X)[:, 1]
    return scores, time.perf_counter() - t0


def run_baselines() -> dict:
    section("Baselines: no-skill + simple Logistic Regression (TransactionAmt/ProductCD/card4)")
    y_train = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
    y_val = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")

    no_skill = no_skill_baseline(y_train["isFraud"].to_numpy(), y_val["isFraud"].to_numpy())
    print("No-skill baseline:", no_skill)

    cols = ["TransactionID", "TransactionAmt", "ProductCD", "card4"]
    train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", cols[1:])
    val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", cols[1:])

    from sklearn.linear_model import LogisticRegression

    ct = build_simple_baseline_pipeline()
    X_train = ct.fit_transform(train_df[["TransactionAmt", "ProductCD", "card4"]])
    X_val = ct.transform(val_df[["TransactionAmt", "ProductCD", "card4"]])

    # Only 3 raw columns feed this model (~10-12 after encoding) -- none of the
    # 271-column missingness-indicator multicollinearity that stalls the
    # full-feature run below exists here, which is why this fit completes in
    # ~0.2s on the real data while that one does not. Default liblinear solver
    # is fine at this width; this baseline is NOT affected by the Stage 7
    # diagnostic finding (see docs/ML_WORKFLOW_STATE.md Decisions log,
    # "Stage 7, diagnostic round").
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    model, fit_s = _timed_fit(model, X_train, y_train["isFraud"].to_numpy())
    scores, predict_s = _timed_predict_proba(model, X_val)
    simple_logreg = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
    simple_logreg["fit_seconds"] = round(fit_s, 3)
    simple_logreg["predict_seconds"] = round(predict_s, 3)
    print("Simple LogReg baseline:", simple_logreg)

    return {"no_skill": no_skill, "simple_logreg": simple_logreg}


def run_logreg_family() -> dict:
    section("Logistic Regression family: Stage 5 (scaled) pipeline, L2 and L1")
    columns, indicator_split = _column_scope()
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols)
    val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    y_train = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
    y_val = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")

    ct = build_scaled_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
    t0 = time.perf_counter()
    X_train = ct.fit_transform(train_df[all_cols])
    fit_transform_s = time.perf_counter() - t0
    X_val = ct.transform(val_df[all_cols])
    print(f"Pipeline fit+transform(train): {fit_transform_s:.2f}s, shape {X_train.shape}")

    from sklearn.linear_model import LogisticRegression

    # DIAGNOSED, NOT YET FIXED (Stage 7 diagnostic round -- see
    # docs/ML_WORKFLOW_STATE.md Decisions log and docs/HANDOFF.md): this
    # full 813-column fit is where the project's Logistic Regression
    # experiment stalled for 40+ minutes. Root cause, measured: 245 of the
    # 271 missingness-indicator columns are exact duplicates, and
    # liblinear's L1 solver is pathologically slow against that
    # multicollinearity specifically (1,692s for 100 non-converging
    # iterations on a 50K-row subsample, vs. 27s for solver="saga" on the
    # identical data). class_weight="balanced" and scaling the indicator
    # columns are contributing, not dominant, factors. The fix this points
    # to -- switch solver, use Stage 6's 26-pattern collapsed-indicator
    # representation -- requires separate implementation approval and has
    # NOT been applied here; this call is left exactly as it was when the
    # problem was diagnosed.
    results = {}
    for name, penalty, solver in [("logreg_l2", "l2", "liblinear"), ("logreg_l1", "l1", "liblinear")]:
        model = LogisticRegression(
            penalty=penalty, solver=solver, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        )
        model, fit_s = _timed_fit(model, X_train, y_train["isFraud"].to_numpy())
        scores, predict_s = _timed_predict_proba(model, X_val)
        metrics = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
        metrics["fit_seconds"] = round(fit_s, 3)
        metrics["predict_seconds"] = round(predict_s, 3)
        metrics["n_nonzero_coefficients"] = int(np.count_nonzero(model.coef_))
        metrics["n_total_coefficients"] = int(model.coef_.size)
        print(f"{name}: {metrics}")
        results[name] = metrics

    results["pipeline_fit_transform_seconds"] = round(fit_transform_s, 3)
    results["n_features"] = int(X_train.shape[1])
    return results


def run_tree_family() -> dict:
    section("Tree family: Stage 5 (unscaled) pipeline -- Random Forest + bounded Gradient Boosting")
    columns, indicator_split = _column_scope()
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    from src.preprocessing import build_preprocessing_pipeline

    train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols)
    val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    y_train = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
    y_val = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")

    ct = build_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
    X_train = ct.fit_transform(train_df[all_cols])
    X_val = ct.transform(val_df[all_cols])
    print(f"Unscaled pipeline output shape: {X_train.shape}")

    results = {}

    section("Random Forest (bounded)")
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    rf, fit_s = _timed_fit(rf, X_train, y_train["isFraud"].to_numpy())
    scores, predict_s = _timed_predict_proba(rf, X_val)
    rf_metrics = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
    rf_metrics["fit_seconds"] = round(fit_s, 3)
    rf_metrics["predict_seconds"] = round(predict_s, 3)
    rf_metrics["config"] = "n_estimators=50, max_depth=10, min_samples_leaf=50, class_weight=balanced"
    print("Random Forest:", rf_metrics)
    results["random_forest"] = rf_metrics

    section("Gradient Boosting -- small-scale timing probe BEFORE committing to a full bounded run")
    from sklearn.ensemble import GradientBoostingClassifier

    probe = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=RANDOM_STATE)
    probe, probe_fit_s = _timed_fit(probe, X_train, y_train["isFraud"].to_numpy())
    per_tree_seconds = probe_fit_s / 5
    projected_50_tree_seconds = per_tree_seconds * 50
    print(f"5-tree probe: {probe_fit_s:.2f}s total ({per_tree_seconds:.2f}s/tree) -> projected 50-tree: {projected_50_tree_seconds:.1f}s")
    results["gboost_probe"] = {
        "n_estimators_probed": 5,
        "probe_fit_seconds": round(probe_fit_s, 3),
        "projected_50_tree_seconds": round(projected_50_tree_seconds, 1),
    }

    gboost_feasible = projected_50_tree_seconds < 600  # 10-minute budget for this bounded candidate
    if gboost_feasible:
        section(f"Gradient Boosting (bounded, sparse-native -- no densification needed): n_estimators=50")
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=RANDOM_STATE)
        gb, gb_fit_s = _timed_fit(gb, X_train, y_train["isFraud"].to_numpy())
        scores, gb_predict_s = _timed_predict_proba(gb, X_val)
        gb_metrics = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
        gb_metrics["fit_seconds"] = round(gb_fit_s, 3)
        gb_metrics["predict_seconds"] = round(gb_predict_s, 3)
        gb_metrics["config"] = "n_estimators=50, max_depth=3 (sklearn GradientBoostingClassifier, native sparse support)"
        print("Gradient Boosting (bounded):", gb_metrics)
        results["gboost_bounded"] = gb_metrics
    else:
        print(f"SKIPPED full bounded run -- projected {projected_50_tree_seconds:.0f}s exceeds the 600s budget for this stage.")
        results["gboost_bounded"] = {"skipped": True, "reason": f"projected {projected_50_tree_seconds:.0f}s exceeds budget"}

    section("HistGradientBoostingClassifier -- memory assessment ONLY, per explicit instruction not to densify without one")
    n_rows, n_cols = X_train.shape
    dense_bytes = estimate_dense_memory_bytes(n_rows, n_cols, dtype_bytes=8)
    results["hgb_memory_assessment"] = {
        "requires_dense_input": True,
        "train_shape": [int(n_rows), int(n_cols)],
        "estimated_dense_bytes_float64": int(dense_bytes),
        "estimated_dense_mb_float64": round(dense_bytes / 1e6, 1),
        "consistent_with_stage5_measured_figure_mb": 2849.6,
        "decision": "NOT run this stage -- densifying costs ~2.85GB on top of HGB's own histogram-binning overhead, "
                    "and HGB is not required to answer Stage 7's model-family question (sklearn's sparse-native "
                    "GradientBoostingClassifier already covers that family). Deferred, not abandoned.",
    }
    print(results["hgb_memory_assessment"])

    return results


def run_feature_ablation_b() -> dict:
    section("Feature-engineering ablation, arm B: Stage 5 (scaled) + Stage 6 engineered features")
    columns, indicator_split = _column_scope()
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols)
    val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    y_train = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
    y_val = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")

    ct = build_scaled_preprocessing_pipeline(numeric_no_indicator, numeric_with_indicator, categorical)
    X_train_a = ct.fit_transform(train_df[all_cols])
    X_val_a = ct.transform(val_df[all_cols])

    section("Building Stage 6 engineered features (train+validation, never final_test)")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    ids = set(get_partition(assignment, ["train", "validation"])["TransactionID"])

    raw_cols = ["TransactionID", "TransactionDT", "card1", "P_emaildomain", "R_emaildomain", "ProductCD", "card4", "TransactionAmt"]
    raw_df = pd.read_csv(TRAIN_TXN, usecols=raw_cols)
    raw_df = raw_df.loc[raw_df["TransactionID"].isin(ids)]

    raw_df = compute_temporal_phase_features(raw_df)
    entity_hist = build_entity_history_train_and_validation(ASSIGNMENT_PATH, TRAIN_TXN, "card1")
    raw_df = raw_df.merge(entity_hist[["TransactionID", "card1_prior_txn_count", "card1_time_since_prior_txn"]], on="TransactionID")

    p_map = fit_frequency_encoder_on_train(ASSIGNMENT_PATH, TRAIN_TXN, "P_emaildomain")
    r_map = fit_frequency_encoder_on_train(ASSIGNMENT_PATH, TRAIN_TXN, "R_emaildomain")
    raw_df["p_emaildomain_freq"] = apply_frequency_encoder(raw_df["P_emaildomain"], p_map)
    raw_df["r_emaildomain_freq"] = apply_frequency_encoder(raw_df["R_emaildomain"], r_map)
    raw_df = compute_interaction_and_amount_features(raw_df)

    engineered_cols = [
        "TransactionID", "txn_hour_phase", "txn_dow_phase", "card1_prior_txn_count", "card1_time_since_prior_txn",
        "p_emaildomain_freq", "r_emaildomain_freq", "email_domain_match", "log1p_transaction_amt",
        "transaction_amt_is_round", "productcd_card4_interaction",
    ]
    engineered_df = raw_df[engineered_cols]

    train_ids = set(get_partition(assignment, ["train"])["TransactionID"])
    val_ids = set(get_partition(assignment, ["validation"])["TransactionID"])
    engineered_train = train_df[["TransactionID"]].merge(engineered_df, on="TransactionID", how="left")
    engineered_val = val_df[["TransactionID"]].merge(engineered_df, on="TransactionID", how="left")

    feat_block = build_engineered_feature_block()
    X_train_feat = feat_block.fit_transform(engineered_train)
    X_val_feat = feat_block.transform(engineered_val)

    X_train_b = hstack_sparse([X_train_a, X_train_feat])
    X_val_b = hstack_sparse([X_val_a, X_val_feat])
    print(f"Arm A shape: {X_train_a.shape}, Arm B shape: {X_train_b.shape}")

    from sklearn.linear_model import LogisticRegression

    # Same unfixed liblinear/class_weight="balanced" config as run_logreg_family()
    # above, carrying the same diagnosed-but-not-yet-fixed risk -- arm A alone
    # (Stage 5's 813 columns) is expected to hit the identical stall; this
    # ablation has not actually been run to completion.
    results = {}
    for name, X_train, X_val in [("arm_a_stage5_only", X_train_a, X_val_a), ("arm_b_stage5_plus_stage6", X_train_b, X_val_b)]:
        model = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
        model, fit_s = _timed_fit(model, X_train, y_train["isFraud"].to_numpy())
        scores, predict_s = _timed_predict_proba(model, X_val)
        metrics = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
        metrics["fit_seconds"] = round(fit_s, 3)
        metrics["predict_seconds"] = round(predict_s, 3)
        metrics["n_features"] = int(X_train.shape[1])
        print(f"{name}: {metrics}")
        results[name] = metrics

    results["ap_delta_b_minus_a"] = round(results["arm_b_stage5_plus_stage6"]["average_precision"] - results["arm_a_stage5_only"]["average_precision"], 5)
    return results


def run_indicator_ablation_collapsed() -> dict:
    section("Missingness-indicator ablation, collapsed arm: Stage 5 (scaled) pipeline with 26-pattern collapsed indicators")
    columns, indicator_split = _column_scope()
    numeric_no_indicator = indicator_split["without_indicator"]
    numeric_with_indicator = indicator_split["with_indicator"]
    categorical = columns["categorical"]
    all_cols = numeric_no_indicator + numeric_with_indicator + categorical

    train_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "train", all_cols)
    val_df = load_partition_df(ASSIGNMENT_PATH, TRAIN_TXN, "validation", all_cols)
    y_train = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "train")
    y_val = load_labels(ASSIGNMENT_PATH, TRAIN_TXN, "validation")

    groups = find_duplicate_indicator_groups(train_df, numeric_with_indicator)
    representatives = collapsed_indicator_representatives(groups)
    print(f"{len(numeric_with_indicator)} indicator columns -> {len(representatives)} representatives (recomputed fresh)")

    ct = build_scaled_pipeline_with_collapsed_indicators(numeric_no_indicator, numeric_with_indicator, categorical, representatives)
    t0 = time.perf_counter()
    X_train = ct.fit_transform(train_df[all_cols])
    fit_transform_s = time.perf_counter() - t0
    X_val = ct.transform(val_df[all_cols])
    print(f"Collapsed pipeline fit+transform(train): {fit_transform_s:.2f}s, shape {X_train.shape}")

    from sklearn.linear_model import LogisticRegression

    # Unlike the other three LogisticRegression calls in this file, this arm
    # already uses the 26-pattern collapsed-indicator representation, which
    # removes the 245 exact-duplicate columns identified as the dominant
    # cause of the Stage 7 slowdown (see docs/HANDOFF.md). It is expected to
    # be less affected, but this has NOT been run to completion or measured
    # -- still liblinear + class_weight="balanced", and other, milder
    # conditioning factors (wide per-column scale differences) were also
    # noted as contributing. Report its actual behavior; don't assume it's
    # fast just because the duplicates are gone.
    model = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    model, fit_s = _timed_fit(model, X_train, y_train["isFraud"].to_numpy())
    scores, predict_s = _timed_predict_proba(model, X_val)
    metrics = evaluate_predictions(y_val["isFraud"].to_numpy(), scores)
    metrics["fit_seconds"] = round(fit_s, 3)
    metrics["predict_seconds"] = round(predict_s, 3)
    metrics["n_features"] = int(X_train.shape[1])
    metrics["n_representatives"] = len(representatives)
    metrics["pipeline_fit_transform_seconds"] = round(fit_transform_s, 3)
    print("Collapsed-indicator arm:", metrics)
    return {"indicator_collapsed": metrics}


RUNS = {
    "baselines": run_baselines,
    "logreg_family": run_logreg_family,
    "tree_family": run_tree_family,
    "feature_ablation_b": run_feature_ablation_b,
    "indicator_ablation_collapsed": run_indicator_ablation_collapsed,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=list(RUNS.keys()))
    args = parser.parse_args()

    result = RUNS[args.run]()
    out_path = RESULTS_DIR / f"_stage7_{args.run}_raw.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWritten to {out_path}")
