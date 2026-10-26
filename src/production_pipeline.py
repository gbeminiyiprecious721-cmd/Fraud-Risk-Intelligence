"""Stage 13 -- Production pipeline.

Packages the APPROVED Stage 5 preprocessing + the FROZEN Stage 9/10
Random Forest into one importable, versioned, testable unit, decoupled
from any driver script. Never reimplements preprocessing or model-
fitting logic -- every fit here delegates to
`src.preprocessing.build_preprocessing_pipeline` and
`src.model_evaluation.build_frozen_model`, unchanged.

Feature-schema ORDER matters, not just feature-name membership. Stage
12's own audit proved that two preprocessing code paths producing the
IDENTICAL transformed-feature SET in a DIFFERENT ORDER yield two
different fitted `RandomForestClassifier` ensembles -- because
`max_features="sqrt"` subsamples features by POSITION at every split.
This module therefore treats transformed-feature order as part of a
fitted artifact's identity (`FeatureSchema.transformed_feature_names`,
and `compute_ordered_schema_fingerprint` below), deliberately distinct
from `experiment_tracking.compute_feature_schema_fingerprint`, which is
intentionally order-INSENSITIVE (sorted before hashing) -- correct for
its own purpose (confirming two fits used the same logical column SET)
but exactly the wrong tool for catching the Stage 12 failure mode.

This module never reads a dataset file itself and never requests
`final_test` -- the driver script owns all data access, same
separation as `src/explainability.py` and `src/responsible_ai.py`.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.experiment_tracking import compute_file_checksum, get_environment_versions, get_git_commit_and_dirty


class SchemaValidationError(ValueError):
    """Raised when raw input to `predict_fraud_probability` (or the
    lower-level `predict_from_transformed`) does not match the fitted
    artifact's documented contract."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when a loaded artifact file's sha256 does not match its
    sidecar manifest's recorded checksum -- the file has changed (or
    been corrupted/tampered with) since `save_artifact` wrote it."""


@dataclass(frozen=True)
class FeatureSchema:
    raw_columns: tuple[str, ...]  # exact ordered raw input columns the fitted ColumnTransformer expects
    numeric_columns: tuple[str, ...]  # subset of raw_columns expected to be numeric-coercible
    transformed_feature_names: tuple[str, ...]  # ct.get_feature_names_out(), EXACT order -- part of the model's identity


def compute_ordered_schema_fingerprint(ordered_names) -> str:
    """sha256 of the EXACT ordered feature-name sequence -- deliberately
    position-sensitive (NOT sorted), the opposite of
    `experiment_tracking.compute_feature_schema_fingerprint`. Two
    fingerprints matching is evidence the feature-name SEQUENCE (not
    just the set) is identical."""
    joined = "\x1f".join(ordered_names)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class PipelineArtifact:
    preprocessor: object  # fitted sklearn ColumnTransformer
    model: object  # fitted RandomForestClassifier
    schema: FeatureSchema


def fit_production_pipeline(
    train_df: pd.DataFrame,
    numeric_without_indicator: list[str],
    numeric_with_indicator: list[str],
    categorical_columns: list[str],
    label_col: str = "isFraud",
) -> PipelineArtifact:
    """Fit the approved Stage 5 preprocessing pipeline + the frozen
    Stage 9/10 Random Forest, in a fixed column order, on `train_df`.
    Delegates entirely to `build_preprocessing_pipeline`/
    `build_frozen_model` -- no preprocessing or model logic is
    reimplemented here."""
    from src.model_evaluation import build_frozen_model
    from src.preprocessing import build_preprocessing_pipeline

    raw_columns = list(numeric_without_indicator) + list(numeric_with_indicator) + list(categorical_columns)
    numeric_columns = list(numeric_without_indicator) + list(numeric_with_indicator)
    ct = build_preprocessing_pipeline(numeric_without_indicator, numeric_with_indicator, categorical_columns)
    X = ct.fit_transform(train_df[raw_columns])
    model = build_frozen_model()
    model.fit(X, train_df[label_col].to_numpy())
    schema = FeatureSchema(
        raw_columns=tuple(raw_columns),
        numeric_columns=tuple(numeric_columns),
        transformed_feature_names=tuple(ct.get_feature_names_out()),
    )
    return PipelineArtifact(preprocessor=ct, model=model, schema=schema)


def validate_raw_input(schema: FeatureSchema, raw_df: pd.DataFrame) -> None:
    """Raises `SchemaValidationError` with a precise reason if `raw_df`
    cannot be safely used for inference:
      - a column name appears more than once
      - a required raw column is missing entirely
      - a numeric-expected column contains a value that cannot be
        coerced to float (the same failure mode Stage 12's
        malformed-dtype check surfaced -- caught EARLY here with a
        clear message instead of deep inside `SimpleImputer`).

    Extra columns beyond `schema.raw_columns` are NOT an error -- they
    are ignored by `align_raw_columns` -- a documented, deliberate part
    of this contract, not an oversight. A permitted missing (NaN) value
    in a numeric column is likewise NOT an error (the approved pipeline
    imputes it) -- only a value that is present but non-numeric is."""
    columns = list(raw_df.columns)
    duplicate_names = sorted({c for c in columns if columns.count(c) > 1})
    if duplicate_names:
        raise SchemaValidationError(f"Duplicate column name(s) in input: {duplicate_names}")

    missing = [c for c in schema.raw_columns if c not in raw_df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required raw column(s): {missing}")

    for col in schema.numeric_columns:
        series = raw_df[col]
        coerced = pd.to_numeric(series, errors="coerce")
        bad_mask = coerced.isna() & series.notna()
        if bad_mask.any():
            bad_values = series[bad_mask].unique().tolist()
            raise SchemaValidationError(
                f"Column '{col}' contains non-numeric value(s) that cannot be used for inference: {bad_values[:5]}"
            )


def align_raw_columns(schema: FeatureSchema, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Selects and REORDERS `raw_df`'s columns to match
    `schema.raw_columns` BY NAME -- safe under any input column order,
    since pandas' `df[list_of_names]` indexing is itself name-based,
    never positional. Must only be called after `validate_raw_input`
    has confirmed every required column is present exactly once."""
    return raw_df[list(schema.raw_columns)]


def predict_fraud_probability(artifact: PipelineArtifact, raw_df: pd.DataFrame) -> np.ndarray:
    """The inference entry point for raw (named-column) input. Validates,
    aligns by name, transforms with the artifact's ALREADY-FITTED
    preprocessor (`.transform`, never `.fit`/`.fit_transform` -- no
    refitting ever happens during inference), and returns
    `model.predict_proba(...)[:, 1]`."""
    validate_raw_input(artifact.schema, raw_df)
    aligned = align_raw_columns(artifact.schema, raw_df)
    X = artifact.preprocessor.transform(aligned)
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    return artifact.model.predict_proba(X_dense)[:, 1]


def predict_from_transformed(artifact: PipelineArtifact, X_transformed, feature_names) -> np.ndarray:
    """A lower-level inference entry point for a caller that already
    holds a transformed numeric matrix (e.g. a cached feature store).
    Requires `feature_names` -- the caller's own record of that matrix's
    column order -- and raises if it does not match the artifact's
    `schema.transformed_feature_names` EXACTLY (same names, same order).
    This function never reorders an already-transformed matrix itself:
    Stage 12 showed why column order is not safe to silently infer or
    "fix" for a matrix that is already numeric with no column names of
    its own attached."""
    if list(feature_names) != list(artifact.schema.transformed_feature_names):
        raise SchemaValidationError(
            "Transformed feature names/order do not match this artifact's fitted schema -- "
            "refusing to guess a reordering. Pass a matrix whose columns exactly match "
            "`artifact.schema.transformed_feature_names`, in that order."
        )
    X_dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed
    return artifact.model.predict_proba(X_dense)[:, 1]


def save_artifact(artifact: PipelineArtifact, path, metadata: dict) -> dict:
    """Persists `{preprocessor, model, schema}` as one joblib file at
    `path`, then writes a sidecar `<path>.manifest.json` containing
    `metadata` plus the artifact file's sha256 checksum, size, and
    ordered-schema fingerprint. Returns the written manifest dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": artifact.preprocessor, "model": artifact.model, "schema": artifact.schema}, path)
    checksum = compute_file_checksum(path)
    manifest = dict(metadata)
    manifest["artifact_path"] = str(path)
    manifest["artifact_sha256"] = checksum
    manifest["artifact_size_bytes"] = path.stat().st_size
    manifest["ordered_schema_fingerprint"] = compute_ordered_schema_fingerprint(artifact.schema.transformed_feature_names)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def load_artifact(path) -> tuple[PipelineArtifact, dict]:
    """Loads a joblib artifact written by `save_artifact`, after
    verifying its current sha256 matches the sidecar manifest's
    recorded checksum (catching corruption or an unexpected
    post-write modification).

    **This checksum check only verifies the file has not changed since
    THIS PROJECT wrote it. It does NOT establish that the artifact's
    original content was trustworthy.** `joblib`/pickle deserialization
    executes arbitrary code paths for an arbitrary pickled object,
    exactly like raw `pickle.load`. Never call this on a `path` outside
    this project's own trusted `artifacts/` directory, and never on a
    file obtained from an untrusted source -- a matching checksum on a
    malicious file only proves the malicious file itself was not
    altered after the fact, not that it is safe."""
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    actual_checksum = compute_file_checksum(path)
    if actual_checksum != manifest.get("artifact_sha256"):
        raise ArtifactIntegrityError(
            f"Checksum mismatch for {path}: manifest says {manifest.get('artifact_sha256')}, "
            f"actual file is {actual_checksum}. Refusing to load a possibly-corrupted or "
            f"-modified artifact."
        )
    payload = joblib.load(path)
    artifact = PipelineArtifact(preprocessor=payload["preprocessor"], model=payload["model"], schema=payload["schema"])
    return artifact, manifest


def environment_compatibility_info() -> dict:
    """Python + dependency versions relevant to whether a saved artifact
    can be safely reloaded later -- reuses the project's existing
    provenance helpers rather than reimplementing version introspection."""
    info = get_environment_versions()
    info["python_version"] = platform.python_version()
    info["joblib"] = joblib.__version__
    return info


def git_provenance_info() -> dict:
    """Reuses the project's existing git-provenance helper (the same one
    `experiment_tracking.track_run` already relies on) rather than
    reimplementing `git rev-parse`/`git status` calls here."""
    commit, dirty = get_git_commit_and_dirty()
    return {"git_commit": commit, "git_dirty": dirty}
