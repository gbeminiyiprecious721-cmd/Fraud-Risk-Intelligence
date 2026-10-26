import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.explainability import (
    bounded_permutation_importance,
    build_feature_lineage,
    build_tree_explainer,
    compute_shap_values,
    correlated_group_permutation_importance,
    explanation_sample_fingerprint,
    global_importance_ranking,
    map_transformed_feature_to_original,
    perturb_numeric_features_and_recompute,
    select_explanation_sample,
    verify_shap_additivity,
)

MODULE_SOURCE_PATH = Path(__file__).resolve().parent.parent / "src" / "explainability.py"


def make_synthetic_classification(n=400, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42)
    model.fit(X, y)
    return X, y, model


# ---------------------------------------------------------------------------
# Deterministic, stratified, train-only sample selection
# ---------------------------------------------------------------------------

def make_synthetic_train_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": (rng.random(n) < 0.1).astype(int),
        "TransactionAmt": rng.uniform(1, 500, n),
    })


def test_select_explanation_sample_is_deterministic():
    df = make_synthetic_train_df()
    a = select_explanation_sample(df, n_fraud=10, n_non_fraud=10, random_state=42)
    b = select_explanation_sample(df, n_fraud=10, n_non_fraud=10, random_state=42)
    assert a["TransactionID"].tolist() == b["TransactionID"].tolist()


def test_select_explanation_sample_is_correctly_stratified():
    df = make_synthetic_train_df()
    sample = select_explanation_sample(df, n_fraud=15, n_non_fraud=25, random_state=42)
    assert (sample["isFraud"] == 1).sum() == 15
    assert (sample["isFraud"] == 0).sum() == 25


def test_select_explanation_sample_rejects_too_small_class_floor():
    df = make_synthetic_train_df()
    with pytest.raises(ValueError):
        select_explanation_sample(df, n_fraud=1, n_non_fraud=10)


def test_explanation_sample_fingerprint_order_independent():
    df = make_synthetic_train_df()
    sample = select_explanation_sample(df, n_fraud=10, n_non_fraud=10, random_state=42)
    shuffled = sample.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert explanation_sample_fingerprint(sample) == explanation_sample_fingerprint(shuffled)


def test_explanation_sample_fingerprint_changes_with_different_rows():
    df = make_synthetic_train_df()
    a = select_explanation_sample(df, n_fraud=10, n_non_fraud=10, random_state=42)
    b = select_explanation_sample(df, n_fraud=10, n_non_fraud=10, random_state=7)
    assert explanation_sample_fingerprint(a) != explanation_sample_fingerprint(b)


# ---------------------------------------------------------------------------
# Feature lineage
# ---------------------------------------------------------------------------

def test_map_numeric_plain_feature():
    result = map_transformed_feature_to_original("numeric_plain__card2", [])
    assert result == {"original_column": "card2", "feature_type": "numeric", "category": None}


def test_map_missingness_indicator_feature():
    result = map_transformed_feature_to_original("numeric_indicator__missingindicator_V128", [])
    assert result["feature_type"] == "missingness_indicator"
    assert result["original_column"] == "V128"


def test_map_numeric_indicator_value_feature():
    result = map_transformed_feature_to_original("numeric_indicator__V128", [])
    assert result["feature_type"] == "numeric"
    assert result["original_column"] == "V128"


def test_map_categorical_feature_with_underscore_in_category():
    result = map_transformed_feature_to_original("categorical__card4_american express", ["card4", "ProductCD"])
    assert result["feature_type"] == "categorical_onehot"
    assert result["original_column"] == "card4"
    assert result["category"] == "american express"


def test_map_categorical_feature_disambiguates_longest_match():
    # "card4" and "card4type" would both match a naive prefix check --
    # the longest-matching known column must win.
    result = map_transformed_feature_to_original("categorical__card4type_x", ["card4", "card4type"])
    assert result["original_column"] == "card4type"


def test_build_feature_lineage_covers_every_name():
    names = ["numeric_plain__card2", "numeric_indicator__missingindicator_V1", "categorical__ProductCD_W"]
    lineage = build_feature_lineage(names, ["ProductCD"])
    assert len(lineage) == 3
    assert set(lineage["transformed_feature"]) == set(names)


# ---------------------------------------------------------------------------
# SHAP -- output shape, additivity, global ranking
# ---------------------------------------------------------------------------

def test_build_tree_explainer_and_compute_shap_values_shape():
    X, y, model = make_synthetic_classification()
    explainer = build_tree_explainer(model)
    result = compute_shap_values(explainer, X[:20])
    assert result["shap_values"].shape == (20, X.shape[1])
    assert "output_space" in result and isinstance(result["output_space"], str)


def test_shap_additivity_passes_for_a_real_fitted_model():
    X, y, model = make_synthetic_classification()
    explainer = build_tree_explainer(model)
    subset = X[:30]
    result = compute_shap_values(explainer, subset)
    model_output = model.predict_proba(subset)[:, 1]
    additivity = verify_shap_additivity(result["shap_values"], result["expected_value"], model_output, tolerance=1e-3)
    assert additivity["passed"] is True
    assert additivity["max_abs_diff"] < 1e-3


def test_shap_additivity_fails_when_values_are_wrong():
    X, y, model = make_synthetic_classification()
    explainer = build_tree_explainer(model)
    subset = X[:10]
    result = compute_shap_values(explainer, subset)
    corrupted = result["shap_values"] + 10.0  # deliberately broken
    model_output = model.predict_proba(subset)[:, 1]
    additivity = verify_shap_additivity(corrupted, result["expected_value"], model_output, tolerance=1e-3)
    assert additivity["passed"] is False


def test_global_importance_ranking_is_sorted_descending():
    X, y, model = make_synthetic_classification()
    explainer = build_tree_explainer(model)
    result = compute_shap_values(explainer, X[:50])
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    ranking = global_importance_ranking(result["shap_values"], feature_names)
    assert list(ranking["mean_abs_shap"]) == sorted(ranking["mean_abs_shap"], reverse=True)
    # The feature actually used to construct y (f0) should rank highly.
    assert ranking.iloc[0]["feature"] in ("f0", "f1")


# ---------------------------------------------------------------------------
# Permutation importance -- bounded + correlated-group
# ---------------------------------------------------------------------------

def test_bounded_permutation_importance_only_touches_requested_features():
    X, y, model = make_synthetic_classification()
    result = bounded_permutation_importance(model, X, y, feature_indices=[0, 1], n_repeats=3)
    assert len(result["importances_mean"]) == 2
    assert result["feature_indices"] == [0, 1]


def test_correlated_group_permutation_importance_detects_a_real_duplicate_group():
    rng = np.random.default_rng(0)
    n = 500
    base = rng.normal(size=n)
    X = np.column_stack([base, base + rng.normal(scale=0.01, size=n), rng.normal(size=n)])  # cols 0,1 near-duplicate
    y = (base > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42).fit(X, y)
    groups = {"duplicate_pair": [0, 1], "unrelated": [2]}
    result = correlated_group_permutation_importance(model, X, y, groups, n_repeats=5)
    assert result["duplicate_pair"]["mean_score_drop"] > result["unrelated"]["mean_score_drop"]


# ---------------------------------------------------------------------------
# Stability under small, realistic perturbation
# ---------------------------------------------------------------------------

def test_perturb_numeric_features_stability_runs_and_reports_both_metrics():
    X, y, model = make_synthetic_classification()
    explainer = build_tree_explainer(model)
    row = X[0]
    result = perturb_numeric_features_and_recompute(model, explainer, row, numeric_col_indices=[0, 1, 2], relative_noise_scale=0.01, n_trials=5)
    assert 0.0 <= result["mean_abs_proba_delta"] <= 1.0
    assert -1.0 <= result["mean_shap_cosine_similarity"] <= 1.0 or np.isnan(result["mean_shap_cosine_similarity"])


def test_perturb_numeric_features_small_noise_gives_high_cosine_similarity():
    """A sanity check on the stability metric itself: TINY noise on a
    well-behaved model should leave attributions nearly unchanged."""
    X, y, model = make_synthetic_classification(n=800)
    explainer = build_tree_explainer(model)
    row = X[0]
    result = perturb_numeric_features_and_recompute(model, explainer, row, numeric_col_indices=list(range(X.shape[1])), relative_noise_scale=0.001, n_trials=5)
    assert result["mean_shap_cosine_similarity"] > 0.8


# ---------------------------------------------------------------------------
# final_test isolation
# ---------------------------------------------------------------------------

def test_module_imports_nothing_that_reads_datasets():
    tree = ast.parse(MODULE_SOURCE_PATH.read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    forbidden = {"src.preprocessing", "src.leakage_boundaries", "src.model_selection", "pandas.read_csv"}
    assert not (imported_names & forbidden)


def test_module_never_calls_with_allow_final_test_true():
    tree = ast.parse(MODULE_SOURCE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                assert not (kw.arg == "allow_final_test" and isinstance(kw.value, ast.Constant) and kw.value.value is True)
