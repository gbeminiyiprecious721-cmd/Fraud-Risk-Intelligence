import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.responsible_ai import (
    build_edge_case_rows,
    malformed_dtype_case,
    run_edge_case_predictions,
    schema_mismatch_case,
    segment_report_with_lift,
)
from src.model_evaluation import segment_report

MODULE_SOURCE_PATH = Path(__file__).resolve().parent.parent / "src" / "responsible_ai.py"

NUMERIC_COLUMNS = ["TransactionAmt", "C1"]
CATEGORICAL_COLUMNS = ["ProductCD", "card6"]
ALL_COLS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def make_synthetic_frame(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "isFraud": (rng.random(n) < 0.1).astype(int),
            "TransactionAmt": rng.uniform(1, 500, n),
            "C1": rng.uniform(0, 20, n),
            "ProductCD": rng.choice(["W", "C", "R"], n),
            "card6": rng.choice(["debit", "credit"], n),
        }
    )


def make_fitted_pipeline_and_model(df: pd.DataFrame):
    ct = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    X = ct.fit_transform(df[ALL_COLS])
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_dense, df["isFraud"].to_numpy())
    return ct, model


# ---------------------------------------------------------------------------
# Subgroup lift
# ---------------------------------------------------------------------------

def test_segment_report_with_lift_computes_ap_over_prevalence():
    df = make_synthetic_frame()
    ct, model = make_fitted_pipeline_and_model(df)
    X = ct.transform(df[ALL_COLS])
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    proba = model.predict_proba(X_dense)[:, 1]
    report = segment_report(df["isFraud"].to_numpy(), proba, df["ProductCD"].to_numpy(), min_samples=10, min_positives=2)
    flattened = segment_report_with_lift(report)
    for segment, row in flattened.iterrows():
        if row["sufficient_sample"]:
            assert row["ap_to_prevalence_lift"] == pytest.approx(row["average_precision"] / row["prevalence"])


def test_segment_report_with_lift_leaves_insufficient_segments_as_nan():
    report = {
        "tiny": {"n_samples": 2, "n_positive": 0, "sufficient_sample": False},
    }
    flattened = segment_report_with_lift(report)
    assert np.isnan(flattened.loc["tiny", "average_precision"])
    assert np.isnan(flattened.loc["tiny", "ap_to_prevalence_lift"])


# ---------------------------------------------------------------------------
# Edge-case construction and execution
# ---------------------------------------------------------------------------

def test_build_edge_case_rows_changes_exactly_one_field_each():
    df = make_synthetic_frame()
    template = df.iloc[0]
    cases = build_edge_case_rows(template, numeric_columns=NUMERIC_COLUMNS)
    assert cases["amount_zero"]["TransactionAmt"].iloc[0] == 0.0
    assert cases["amount_zero"]["C1"].iloc[0] == template["C1"]  # unchanged
    assert cases["unseen_productcd"]["ProductCD"].iloc[0] == "ZZ"
    assert cases["all_numeric_missing"][NUMERIC_COLUMNS].isna().all(axis=None)


def test_run_edge_case_predictions_amount_extremes_do_not_crash():
    df = make_synthetic_frame()
    ct, model = make_fitted_pipeline_and_model(df)
    cases = build_edge_case_rows(df.iloc[0], numeric_columns=NUMERIC_COLUMNS)
    results = run_edge_case_predictions(model, ct, cases, ALL_COLS)
    for name in ["amount_zero", "amount_tiny_positive", "amount_negative", "amount_tail_p999_seen_in_data", "amount_far_out_of_distribution"]:
        assert results[name]["raised"] is False, name
        assert results[name]["is_finite"] is True, name
        assert results[name]["in_valid_range"] is True, name


def test_run_edge_case_predictions_unseen_categorical_does_not_crash():
    df = make_synthetic_frame()
    ct, model = make_fitted_pipeline_and_model(df)
    cases = build_edge_case_rows(df.iloc[0], numeric_columns=NUMERIC_COLUMNS)
    results = run_edge_case_predictions(model, ct, cases, ALL_COLS)
    assert results["unseen_productcd"]["raised"] is False
    assert results["unseen_card6"]["raised"] is False
    assert results["unseen_productcd"]["in_valid_range"] is True


def test_run_edge_case_predictions_all_numeric_missing_imputed_not_crashed():
    df = make_synthetic_frame()
    ct, model = make_fitted_pipeline_and_model(df)
    cases = build_edge_case_rows(df.iloc[0], numeric_columns=NUMERIC_COLUMNS)
    results = run_edge_case_predictions(model, ct, cases, ALL_COLS)
    assert results["all_numeric_missing"]["raised"] is False
    assert results["all_numeric_missing"]["in_valid_range"] is True


def test_run_edge_case_predictions_reports_real_exception_when_one_occurs():
    df = make_synthetic_frame()
    ct, model = make_fitted_pipeline_and_model(df)
    # Deliberately malformed input: a column the transformer was never fit on.
    bad_case = {"bogus": df.iloc[[0]].drop(columns=["TransactionAmt"]).assign(TransactionAmt="not_numeric")}
    results = run_edge_case_predictions(model, ct, bad_case, ALL_COLS)
    assert results["bogus"]["raised"] is True
    assert results["bogus"]["exception_type"] is not None
    assert results["bogus"]["predicted_proba"] is None


# ---------------------------------------------------------------------------
# Schema mismatch and malformed dtype -- real observed behaviour, not assumed
# ---------------------------------------------------------------------------

def test_schema_mismatch_case_raises_a_clear_error():
    df = make_synthetic_frame()
    ct, _ = make_fitted_pipeline_and_model(df)
    row_df = df.iloc[[0]]
    result = schema_mismatch_case(ct, row_df, ALL_COLS, drop_column="TransactionAmt")
    assert result["raised"] is True
    assert result["exception_type"] is not None


def test_malformed_dtype_case_is_observed_not_assumed():
    df = make_synthetic_frame()
    ct, model = make_fitted_pipeline_and_model(df)
    row_df = malformed_dtype_case(df.iloc[0], "TransactionAmt")
    results = run_edge_case_predictions(model, ct, {"malformed": row_df}, ALL_COLS)
    # Whatever sklearn actually does (raise, or coerce-and-impute), the
    # result must be ONE of these two well-defined outcomes -- never a
    # silent NaN/garbage probability with no exception and no valid range.
    entry = results["malformed"]
    assert entry["raised"] is True or (entry["is_finite"] and entry["in_valid_range"])


# ---------------------------------------------------------------------------
# final_test isolation (same discipline as src/explainability.py)
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
