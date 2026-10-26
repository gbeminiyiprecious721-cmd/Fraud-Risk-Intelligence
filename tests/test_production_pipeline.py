import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.production_pipeline import (
    ArtifactIntegrityError,
    SchemaValidationError,
    align_raw_columns,
    compute_ordered_schema_fingerprint,
    fit_production_pipeline,
    load_artifact,
    predict_fraud_probability,
    predict_from_transformed,
    save_artifact,
    validate_raw_input,
)

MODULE_SOURCE_PATH = Path(__file__).resolve().parent.parent / "src" / "production_pipeline.py"

NUMERIC_WITHOUT_INDICATOR = ["TransactionAmt"]
NUMERIC_WITH_INDICATOR = ["C1"]  # a column that would carry a missingness indicator in the real pipeline
CATEGORICAL_COLUMNS = ["ProductCD", "card6"]


def make_synthetic_frame(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c1 = rng.uniform(0, 20, n)
    c1[rng.random(n) < 0.1] = np.nan  # permitted missing values
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "isFraud": (rng.random(n) < 0.15).astype(int),
            "TransactionAmt": rng.uniform(1, 500, n),
            "C1": c1,
            "ProductCD": rng.choice(["W", "C", "R"], n),
            "card6": rng.choice(["debit", "credit"], n),
        }
    )


def fit_synthetic_artifact(seed=0):
    df = make_synthetic_frame(seed=seed)
    artifact = fit_production_pipeline(df, NUMERIC_WITHOUT_INDICATOR, NUMERIC_WITH_INDICATOR, CATEGORICAL_COLUMNS)
    return df, artifact


# ---------------------------------------------------------------------------
# Fitting and schema
# ---------------------------------------------------------------------------

def test_fit_production_pipeline_builds_a_schema_with_the_right_raw_columns():
    _, artifact = fit_synthetic_artifact()
    assert list(artifact.schema.raw_columns) == NUMERIC_WITHOUT_INDICATOR + NUMERIC_WITH_INDICATOR + CATEGORICAL_COLUMNS
    assert set(artifact.schema.numeric_columns) == {"TransactionAmt", "C1"}
    assert len(artifact.schema.transformed_feature_names) > 0


def test_predict_fraud_probability_matches_manual_transform_and_predict():
    df, artifact = fit_synthetic_artifact()
    proba_via_api = predict_fraud_probability(artifact, df)
    X = artifact.preprocessor.transform(df[list(artifact.schema.raw_columns)])
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    proba_manual = artifact.model.predict_proba(X_dense)[:, 1]
    assert np.allclose(proba_via_api, proba_manual)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_validate_raw_input_raises_on_missing_column():
    _, artifact = fit_synthetic_artifact()
    bad_df = make_synthetic_frame().drop(columns=["TransactionAmt"])
    with pytest.raises(SchemaValidationError, match="Missing required"):
        validate_raw_input(artifact.schema, bad_df)


def test_validate_raw_input_raises_on_duplicate_column_name():
    _, artifact = fit_synthetic_artifact()
    bad_df = make_synthetic_frame()
    bad_df = pd.concat([bad_df, bad_df[["TransactionAmt"]]], axis=1)  # duplicate "TransactionAmt"
    with pytest.raises(SchemaValidationError, match="Duplicate column"):
        validate_raw_input(artifact.schema, bad_df)


def test_validate_raw_input_raises_on_non_numeric_value_in_numeric_column():
    _, artifact = fit_synthetic_artifact()
    bad_df = make_synthetic_frame().copy()
    bad_df["TransactionAmt"] = bad_df["TransactionAmt"].astype(object)
    bad_df.loc[0, "TransactionAmt"] = "not_a_number"
    with pytest.raises(SchemaValidationError, match="non-numeric"):
        validate_raw_input(artifact.schema, bad_df)


def test_validate_raw_input_allows_permitted_missing_numeric_values():
    _, artifact = fit_synthetic_artifact()
    df_with_nans = make_synthetic_frame()
    assert df_with_nans["C1"].isna().any()  # the fixture already has real NaNs
    validate_raw_input(artifact.schema, df_with_nans)  # must not raise


def test_validate_raw_input_ignores_extra_columns():
    _, artifact = fit_synthetic_artifact()
    df = make_synthetic_frame()
    df["some_extra_metadata_column"] = "ignored"
    validate_raw_input(artifact.schema, df)  # must not raise


# ---------------------------------------------------------------------------
# Column-order robustness (direct response to the Stage 12 finding)
# ---------------------------------------------------------------------------

def test_predict_fraud_probability_is_deterministic_under_raw_column_reordering():
    df, artifact = fit_synthetic_artifact()
    proba_canonical = predict_fraud_probability(artifact, df)
    scrambled = df[list(df.columns)[::-1]]  # same columns, reversed order
    proba_scrambled = predict_fraud_probability(artifact, scrambled)
    assert np.allclose(proba_canonical, proba_scrambled)


def test_align_raw_columns_reorders_by_name_not_position():
    _, artifact = fit_synthetic_artifact()
    df = make_synthetic_frame()
    scrambled = df[list(df.columns)[::-1]]
    aligned = align_raw_columns(artifact.schema, scrambled)
    assert list(aligned.columns) == list(artifact.schema.raw_columns)


def test_predict_from_transformed_rejects_mismatched_feature_order():
    _, artifact = fit_synthetic_artifact()
    X = artifact.preprocessor.transform(make_synthetic_frame()[list(artifact.schema.raw_columns)])
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    scrambled_names = list(artifact.schema.transformed_feature_names)[::-1]
    with pytest.raises(SchemaValidationError, match="do not match"):
        predict_from_transformed(artifact, X_dense, scrambled_names)


def test_predict_from_transformed_succeeds_with_correct_feature_order():
    _, artifact = fit_synthetic_artifact()
    X = artifact.preprocessor.transform(make_synthetic_frame()[list(artifact.schema.raw_columns)])
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    proba = predict_from_transformed(artifact, X_dense, artifact.schema.transformed_feature_names)
    assert proba.shape[0] == X_dense.shape[0]
    assert np.all((proba >= 0) & (proba <= 1))


def test_compute_ordered_schema_fingerprint_is_order_sensitive():
    fp_ab = compute_ordered_schema_fingerprint(["a", "b"])
    fp_ba = compute_ordered_schema_fingerprint(["b", "a"])
    fp_ab_again = compute_ordered_schema_fingerprint(["a", "b"])
    assert fp_ab != fp_ba
    assert fp_ab == fp_ab_again


# ---------------------------------------------------------------------------
# Unseen categorical values / no-refit-during-inference
# ---------------------------------------------------------------------------

def test_predict_fraud_probability_handles_unseen_categorical_value():
    df, artifact = fit_synthetic_artifact()
    unseen_row = df.iloc[[0]].copy()
    unseen_row["ProductCD"] = "ZZ_never_seen"
    proba = predict_fraud_probability(artifact, unseen_row)
    assert proba.shape == (1,)
    assert 0.0 <= proba[0] <= 1.0


def test_predict_does_not_refit_preprocessor_or_model():
    df, artifact = fit_synthetic_artifact()
    imputer_stats_before = artifact.preprocessor.named_transformers_["numeric_plain"].named_steps["impute"].statistics_.copy()
    n_estimators_before = len(artifact.model.estimators_)
    predict_fraud_probability(artifact, df)
    predict_fraud_probability(artifact, df)
    imputer_stats_after = artifact.preprocessor.named_transformers_["numeric_plain"].named_steps["impute"].statistics_
    assert np.array_equal(imputer_stats_before, imputer_stats_after)
    assert len(artifact.model.estimators_) == n_estimators_before


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_artifact_round_trip_predictions_match(tmp_path):
    df, artifact = fit_synthetic_artifact()
    proba_before = predict_fraud_probability(artifact, df)
    manifest = save_artifact(artifact, tmp_path / "artifact.joblib", metadata={"artifact_id": "test", "version": "0.0.1"})
    assert manifest["artifact_sha256"] is not None
    assert manifest["artifact_size_bytes"] > 0
    loaded_artifact, loaded_manifest = load_artifact(tmp_path / "artifact.joblib")
    proba_after = predict_fraud_probability(loaded_artifact, df)
    assert np.allclose(proba_before, proba_after)
    assert loaded_manifest["artifact_id"] == "test"
    assert list(loaded_artifact.schema.transformed_feature_names) == list(artifact.schema.transformed_feature_names)


def test_load_artifact_detects_checksum_tamper(tmp_path):
    _, artifact = fit_synthetic_artifact()
    artifact_path = tmp_path / "artifact.joblib"
    save_artifact(artifact, artifact_path, metadata={"artifact_id": "tamper-test"})
    with open(artifact_path, "ab") as f:
        f.write(b"\x00\x01\x02tampered-bytes")
    with pytest.raises(ArtifactIntegrityError):
        load_artifact(artifact_path)


# ---------------------------------------------------------------------------
# final_test isolation (same discipline as src/explainability.py, src/responsible_ai.py)
# ---------------------------------------------------------------------------

def test_module_imports_nothing_that_reads_datasets():
    tree = ast.parse(MODULE_SOURCE_PATH.read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    forbidden = {"src.leakage_boundaries", "src.model_selection", "pandas.read_csv"}
    assert not (imported_names & forbidden)


def test_module_never_calls_with_allow_final_test_true():
    tree = ast.parse(MODULE_SOURCE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                assert not (kw.arg == "allow_final_test" and isinstance(kw.value, ast.Constant) and kw.value.value is True)


# ---------------------------------------------------------------------------
# Heavyweight, real-data regression test -- the stage's actual completion
# evidence. Fits TWICE on the complete `train` partition (438,125 rows):
# once via an INDEPENDENT reference procedure that calls
# `build_preprocessing_pipeline`/`build_frozen_model` directly (never
# `fit_production_pipeline`), and once via the new packaged
# `fit_production_pipeline`. Deliberately NOT mocked/synthetic -- this
# is the one real-data check this stage requires before acceptance.
# Expensive (~1-2 minutes); kept in its own clearly-named function,
# separate from the fast unit tests above, per explicit instruction.
# ---------------------------------------------------------------------------

def test_full_training_regression_against_independent_reference_procedure():
    from src.leakage_boundaries import FinalTestAccessError, get_partition
    from src.model_evaluation import FROZEN_RF_CONFIG, build_frozen_model
    from src.model_selection import align_labels_to_partition_df, load_labels
    from src.preprocessing import (
        DEFAULT_INDICATOR_THRESHOLD,
        build_preprocessing_pipeline,
        load_included_columns,
        load_partition_df,
        measure_train_missingness,
        select_indicator_columns,
    )

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "raw" / "ieee-fraud-detection"
    train_txn = data_dir / "train_transaction.csv"
    docs_dir = root / "docs"
    register_path = docs_dir / "feature_availability_register.csv"
    assignment_path = docs_dir / "_stage4_split_assignment.csv"

    assignment = pd.read_csv(assignment_path)
    with pytest.raises(FinalTestAccessError):
        get_partition(assignment, ["train", "final_test"])

    columns = load_included_columns(register_path, train_txn)
    numeric_columns, categorical_columns = columns["numeric"], columns["categorical"]
    missingness = measure_train_missingness(assignment_path, train_txn, numeric_columns)
    indicator_split = select_indicator_columns(missingness, threshold_pct=DEFAULT_INDICATOR_THRESHOLD * 100)
    all_cols = indicator_split["without_indicator"] + indicator_split["with_indicator"] + categorical_columns

    train_df = load_partition_df(assignment_path, train_txn, "train", all_cols)
    labels = load_labels(assignment_path, train_txn, "train")
    train_df = align_labels_to_partition_df(train_df, labels)

    # Independent reference path -- calls the approved building blocks directly,
    # NEVER `fit_production_pipeline`, so this test does not compare the new
    # wrapper against itself.
    t0 = time.perf_counter()
    ct_reference = build_preprocessing_pipeline(indicator_split["without_indicator"], indicator_split["with_indicator"], categorical_columns)
    X_reference = ct_reference.fit_transform(train_df[all_cols])
    model_reference = build_frozen_model()
    model_reference.fit(X_reference, train_df["isFraud"].to_numpy())
    reference_seconds = time.perf_counter() - t0
    proba_reference = model_reference.predict_proba(X_reference)[:, 1]

    # New packaged path.
    t0 = time.perf_counter()
    artifact = fit_production_pipeline(train_df, indicator_split["without_indicator"], indicator_split["with_indicator"], categorical_columns)
    new_path_seconds = time.perf_counter() - t0
    proba_new = predict_fraud_probability(artifact, train_df)

    print(f"\nFull-training regression: reference {reference_seconds:.1f}s, new path {new_path_seconds:.1f}s")

    # 1. Model configuration identity.
    assert model_reference.get_params() == artifact.model.get_params()
    for key, value in FROZEN_RF_CONFIG.items():
        assert artifact.model.get_params()[key] == value
    # 2. Training-data identity -- both fits used the identical `train_df`.
    # 3. Ordered feature-schema identity.
    assert list(artifact.schema.transformed_feature_names) == list(ct_reference.get_feature_names_out())
    # 4. Equivalent probability predictions, within a tolerance justified by this
    # project's own documented macOS n_jobs=-1 parallel-float-summation artifact
    # (Stage 7/10 precedent: differences at the ~1e-15 level between otherwise-
    # identical fits/reloads).
    assert np.allclose(proba_reference, proba_new, rtol=1e-9, atol=1e-12)
