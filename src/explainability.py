"""Stage 11 — Explainability.

Explains the FROZEN Stage 9/10 Random Forest configuration
(`n_estimators=100, max_depth=15, min_samples_leaf=5, max_features="sqrt",
random_state=42, class_weight=None`) using SHAP's `TreeExplainer` (global
+ local attributions) and scikit-learn's `permutation_importance` (a
cross-check using a genuinely different mechanism). Operates ONLY on
`train` -- this module never imports anything capable of reading
`validation` or `final_test`, and contains no call site with
`allow_final_test=True` (verified by a static test, the same discipline
already applied to `src/experiment_tracking.py` and `src/model_evaluation.py`).

Explicit, load-bearing design choice: `feature_perturbation="tree_path_dependent"`
(SHAP's default background-free mode for tree ensembles), NOT
`"interventional"`. `"interventional"` SHAP values are computed by
independently resampling each feature from a background dataset while
holding others fixed -- which explicitly BREAKS real-world feature
correlation structure to isolate each feature's "pure" marginal effect.
This project has extensively documented (Stage 7) that its 271
missingness-indicator columns collapse to 26 truly-independent patterns,
and that ~64-66 of its numeric `V`-columns form tight correlated clusters
(>0.99 correlation). Deliberately breaking that structure (which is how
the real transactions actually co-vary) to compute an "independent"
attribution would produce LESS faithful, not more faithful, local
explanations for this specific dataset -- `"tree_path_dependent"` instead
respects the training data's actual joint distribution via the tree's own
path-weighting. This choice is itself a documented limitation, not a free
win: `"tree_path_dependent"` values can still assign credit ambiguously
*among* a correlated group (see `assess_correlated_group_stability`).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

MIN_LOCAL_EXAMPLES_PER_CLASS = 2  # floor for a meaningful stratified sample


# ---------------------------------------------------------------------------
# 1. Deterministic, train-only explanation sample
# ---------------------------------------------------------------------------

def select_explanation_sample(
    train_df: pd.DataFrame, label_col: str = "isFraud", id_col: str = "TransactionID",
    n_fraud: int = 100, n_non_fraud: int = 100, random_state: int = 42,
) -> pd.DataFrame:
    """A fixed, deterministic, STRATIFIED sample from `train` ONLY --
    `n_fraud` positive rows and `n_non_fraud` negative rows, selected by
    sorting on `id_col` (never file/row order, which is not a meaningful
    selection criterion) and sampling with a fixed seed. Stratified, not
    cherry-picked: this guarantees the local-explanation examples include
    genuine fraud cases without looking at any model prediction, score, or
    "how interesting the plot looks" -- the selection criterion is fixed
    BEFORE any explanation is computed.
    """
    if n_fraud < MIN_LOCAL_EXAMPLES_PER_CLASS or n_non_fraud < MIN_LOCAL_EXAMPLES_PER_CLASS:
        raise ValueError(f"n_fraud/n_non_fraud must each be >= {MIN_LOCAL_EXAMPLES_PER_CLASS}")
    sorted_df = train_df.sort_values(id_col, kind="stable").reset_index(drop=True)
    fraud_pool = sorted_df.loc[sorted_df[label_col] == 1]
    non_fraud_pool = sorted_df.loc[sorted_df[label_col] == 0]
    fraud_sample = fraud_pool.sample(n=min(n_fraud, len(fraud_pool)), random_state=random_state)
    non_fraud_sample = non_fraud_pool.sample(n=min(n_non_fraud, len(non_fraud_pool)), random_state=random_state)
    return pd.concat([fraud_sample, non_fraud_sample]).sort_values(id_col, kind="stable").reset_index(drop=True)


def explanation_sample_fingerprint(sample_df: pd.DataFrame, id_col: str = "TransactionID") -> str:
    joined = "\x1f".join(str(i) for i in sorted(sample_df[id_col].tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 2. Feature lineage -- transformed (ColumnTransformer output) -> original
# ---------------------------------------------------------------------------

def map_transformed_feature_to_original(feature_name: str, categorical_columns: list[str]) -> dict:
    """Parses one `ColumnTransformer.get_feature_names_out()` name back to
    its original raw column and transform type, given the REAL naming
    convention of `src.preprocessing.build_preprocessing_pipeline`
    (verified empirically, not assumed): `numeric_plain__<col>`,
    `numeric_indicator__<col>` (imputed value) or
    `numeric_indicator__missingindicator_<col>` (the added 0/1 flag),
    `categorical__<col>_<category>` (one-hot).
    """
    if feature_name.startswith("numeric_plain__"):
        return {"original_column": feature_name[len("numeric_plain__"):], "feature_type": "numeric", "category": None}
    if feature_name.startswith("numeric_indicator__missingindicator_"):
        return {"original_column": feature_name[len("numeric_indicator__missingindicator_"):], "feature_type": "missingness_indicator", "category": None}
    if feature_name.startswith("numeric_indicator__"):
        return {"original_column": feature_name[len("numeric_indicator__"):], "feature_type": "numeric", "category": None}
    if feature_name.startswith("categorical__"):
        rest = feature_name[len("categorical__"):]
        # Disambiguate "<col>_<category>" by the LONGEST matching known
        # categorical column prefix -- category values themselves may
        # contain underscores (e.g. "american express"), so this is not a
        # simple single-split.
        matches = [c for c in categorical_columns if rest == c or rest.startswith(c + "_")]
        if not matches:
            return {"original_column": None, "feature_type": "categorical_onehot", "category": rest}
        best = max(matches, key=len)
        category = rest[len(best) + 1:] if rest != best else None
        return {"original_column": best, "feature_type": "categorical_onehot", "category": category}
    return {"original_column": None, "feature_type": "unknown", "category": None}


def build_feature_lineage(feature_names: list[str], categorical_columns: list[str]) -> pd.DataFrame:
    rows = [{"transformed_feature": f, **map_transformed_feature_to_original(f, categorical_columns)} for f in feature_names]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. SHAP -- feasibility probe, explainer construction, additivity check
# ---------------------------------------------------------------------------

def build_tree_explainer(model):
    """`feature_perturbation="tree_path_dependent"` -- see the module
    docstring for why this, not `"interventional"`, was chosen for this
    dataset's documented correlation structure. No explicit background
    dataset is needed or passed in this mode (SHAP uses the fitted trees'
    own internal path-weighting instead)."""
    import shap

    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


def compute_shap_values(explainer, X) -> dict:
    """Runs the explainer and returns a dict documenting EXACTLY which
    model output the returned SHAP values correspond to -- never assumed
    silently. For a binary `RandomForestClassifier` with
    `tree_path_dependent` TreeExplainer, SHAP's modern API returns an
    `Explanation` object whose `.values` has shape
    `(n_samples, n_features, n_classes)`; this function extracts class 1
    (the fraud-probability output, matching every metric this project has
    ever reported) and states that choice explicitly in the returned dict.
    """
    explanation = explainer(X)
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        # (n_samples, n_features, n_classes) -- take class index 1 (fraud).
        class_index = 1
        shap_values = values[:, :, class_index]
        expected_value = base_values[:, class_index] if base_values.ndim == 2 else base_values[class_index]
        output_space = "class_1_probability (fraud), extracted from a multi-class Explanation object at class index 1"
    else:
        shap_values = values
        expected_value = base_values
        output_space = "single-output (class 1 / fraud probability assumed, binary classifier default)"
    return {"shap_values": shap_values, "expected_value": expected_value, "output_space": output_space}


def verify_shap_additivity(shap_values: np.ndarray, expected_value, model_output: np.ndarray, tolerance: float = 1e-3) -> dict:
    """The standard SHAP sanity check: for each row, `sum(shap_values) +
    expected_value` should equal that row's actual model output, within
    numerical tolerance. A real failure here would mean the SHAP values
    are not explaining the stated output space correctly."""
    row_sums = shap_values.sum(axis=1)
    expected_value_arr = np.broadcast_to(np.asarray(expected_value), row_sums.shape)
    reconstructed = row_sums + expected_value_arr
    diffs = np.abs(reconstructed - np.asarray(model_output))
    return {
        "max_abs_diff": float(diffs.max()), "mean_abs_diff": float(diffs.mean()),
        "tolerance": tolerance, "passed": bool(diffs.max() <= tolerance),
    }


def global_importance_ranking(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Permutation importance cross-check -- sklearn-native, bounded
# ---------------------------------------------------------------------------

def bounded_permutation_importance(model, X, y, feature_indices: list[int], scoring: str = "average_precision", n_repeats: int = 5, random_state: int = 42) -> dict:
    """Permutation importance computed with sklearn's own `get_scorer`,
    but manually BOUNDED to `feature_indices` only (e.g. the top-K SHAP
    features, not all 728+ columns -- scoring every column would multiply
    n_repeats x n_features expensive re-predictions on the already-fitted
    model). sklearn's own `permutation_importance` has no "only these
    columns" option and always permutes every column internally, which
    would defeat the bounding -- this reimplements its per-feature logic
    directly, still using only already-provided scikit-learn primitives
    (`get_scorer`), no new dependency.
    """
    from sklearn.metrics import get_scorer

    rng = np.random.default_rng(random_state)
    scorer = get_scorer(scoring)
    X = X if not hasattr(X, "toarray") else X.toarray()
    baseline_score = scorer(model, X, y)

    means, stds = [], []
    for idx in feature_indices:
        drops = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, idx] = rng.permutation(X_permuted[:, idx])
            drops.append(baseline_score - scorer(model, X_permuted, y))
        means.append(float(np.mean(drops)))
        stds.append(float(np.std(drops)))

    return {
        "feature_indices": feature_indices, "baseline_score": float(baseline_score),
        "importances_mean": means, "importances_std": stds,
        "n_repeats": n_repeats, "scoring": scoring,
    }


def correlated_group_permutation_importance(model, X, y, column_groups: dict[str, list[int]], scoring: str = "average_precision", n_repeats: int = 5, random_state: int = 42) -> dict:
    """Permutes an ENTIRE correlated group of columns together (the SAME
    row-permutation index applied to every column in the group
    simultaneously), rather than one column at a time -- a group of
    near-duplicate columns permuted individually will each show near-zero
    importance (any one of them is redundant given the others), which is
    a known, misleading artifact of naive permutation importance under
    correlation. Permuting the whole group together measures what the
    GROUP collectively contributes.
    """
    from sklearn.metrics import get_scorer

    rng = np.random.default_rng(random_state)
    scorer = get_scorer(scoring)
    baseline_score = scorer(model, X, y)
    results = {}
    for group_name, col_indices in column_groups.items():
        drops = []
        for _ in range(n_repeats):
            X_permuted = X.copy() if not hasattr(X, "toarray") else X.toarray()
            perm = rng.permutation(X_permuted.shape[0])
            X_permuted[:, col_indices] = X_permuted[perm][:, col_indices]
            score = scorer(model, X_permuted, y)
            drops.append(baseline_score - score)
        results[group_name] = {
            "n_columns": len(col_indices), "baseline_score": float(baseline_score),
            "mean_score_drop": float(np.mean(drops)), "std_score_drop": float(np.std(drops)),
        }
    return results


# ---------------------------------------------------------------------------
# 5. Explanation stability -- small, predefined, realistic perturbations
# ---------------------------------------------------------------------------

def perturb_numeric_features_and_recompute(
    model, explainer, X_row: np.ndarray, numeric_col_indices: list[int],
    relative_noise_scale: float = 0.01, n_trials: int = 10, random_state: int = 42,
) -> dict:
    """Adds small, REALISTIC Gaussian noise (a fixed fraction of each
    feature's own value, not an arbitrary absolute jump) to ONLY the
    continuous numeric columns of one row (never to one-hot/indicator
    columns, which only take legitimate values of exactly 0 or 1 -- adding
    noise there would be an unrealistic perturbation, not a realistic one).
    Measures (a) how much the predicted probability moves, and (b) how
    much the SHAP attribution vector itself moves (cosine similarity to
    the original) -- two DIFFERENT things, reported separately, never
    conflated.
    """
    rng = np.random.default_rng(random_state)
    X_row = np.asarray(X_row, dtype=float).reshape(1, -1)
    original_proba = float(model.predict_proba(X_row)[0, 1])
    original_shap = compute_shap_values(explainer, X_row)["shap_values"][0]

    proba_deltas, cosine_sims = [], []
    for _ in range(n_trials):
        perturbed = X_row.copy()
        for idx in numeric_col_indices:
            noise = rng.normal(0, abs(perturbed[0, idx]) * relative_noise_scale + 1e-9)
            perturbed[0, idx] = perturbed[0, idx] + noise
        new_proba = float(model.predict_proba(perturbed)[0, 1])
        new_shap = compute_shap_values(explainer, perturbed)["shap_values"][0]
        proba_deltas.append(abs(new_proba - original_proba))
        denom = (np.linalg.norm(original_shap) * np.linalg.norm(new_shap))
        cosine_sims.append(float(np.dot(original_shap, new_shap) / denom) if denom > 0 else float("nan"))

    return {
        "original_proba": original_proba, "n_trials": n_trials, "relative_noise_scale": relative_noise_scale,
        "mean_abs_proba_delta": float(np.mean(proba_deltas)), "max_abs_proba_delta": float(np.max(proba_deltas)),
        "mean_shap_cosine_similarity": float(np.nanmean(cosine_sims)), "min_shap_cosine_similarity": float(np.nanmin(cosine_sims)),
    }
