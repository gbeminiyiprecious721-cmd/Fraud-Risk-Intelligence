"""Stage 14 -- local model-serving API.

A small, importable FastAPI application that serves predictions from
the EXISTING, trusted Stage 13 artifact only. It never trains,
retrains, or re-evaluates anything; never accesses `final_test`; and
never exposes a way to upload or load an arbitrary joblib file (the
artifact path is a fixed, project-local constant, never a user-
supplied value).

All feature-schema validation and inference logic is DELEGATED to
`src.production_pipeline` (`load_artifact`, `predict_fraud_probability`,
which itself calls `validate_raw_input`/`align_raw_columns`) -- this
module adds only the HTTP contract (request/response shape, status
codes, logging) around that existing, already-tested logic. It never
reimplements preprocessing, schema validation, or feature engineering.

Run locally (never bind beyond 127.0.0.1 -- see `run_local_demo.py`):
    .venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from src.production_pipeline import (
    ArtifactIntegrityError,
    SchemaValidationError,
    load_artifact,
    predict_fraud_probability,
)

API_CONTRACT_VERSION = "v1"  # the HTTP contract's own version -- deliberately independent of
# the artifact's `artifact_version` (1.0.0) and the manifest's `manifest_schema_version` (1.1.0).
# Bumping this means the REQUEST/RESPONSE SHAPE changed; bumping the artifact/manifest versions
# means the MODEL or its METADATA changed. The three must never be conflated.

EXPECTED_ARTIFACT_ID = "frozen-rf-fraud-risk"
DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "frozen_rf_fraud_risk_v1.joblib"

# Stage 15 -- narrowly scoped CORS, explicitly approved, for the local static UI
# ONLY. Both this API and the UI's own local static server are documented to
# bind to 127.0.0.1 -- never 0.0.0.0, never a public interface. No wildcard
# origin, no credentials (no cookies/auth exist to protect here), and only the
# two HTTP methods the UI actually needs. `localhost` and `127.0.0.1` are
# listed as SEPARATE origins deliberately -- browsers treat them as different
# origins even though they resolve to the same host.
ALLOWED_UI_ORIGINS = [
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

logger = logging.getLogger("fraud_risk_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    fraud_probability: float
    api_contract_version: str
    artifact_version: str | None
    manifest_schema_version: str | None
    artifact_sha256: str | None


class _DuplicateKeyError(Exception):
    """Raised when the raw JSON request body contains the same top-level
    field name more than once -- standard `json.loads` silently keeps
    only the LAST value for a repeated key, which would hide the
    problem from Stage 13's own (DataFrame-column-based) duplicate
    check entirely. Caught here, at the JSON-parsing boundary, which is
    the only place this specific failure mode is even visible."""


def _parse_json_detecting_duplicate_keys(raw_body: bytes):
    def hook(pairs):
        seen: set[str] = set()
        duplicates: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        if duplicates:
            raise _DuplicateKeyError(f"Duplicate field name(s) in request body: {sorted(duplicates)}")
        return dict(pairs)

    return json.loads(raw_body, object_pairs_hook=hook)


def _log_event(**fields) -> None:
    """Structured, single-line JSON log. Deliberately NEVER passed a
    full transaction payload, an exception message, or any raw field
    value -- only request metadata (id, status, timing, a short error
    CATEGORY name) that is safe to persist."""
    logger.info(json.dumps(fields, default=str))


def create_app(artifact_path: Path | str = DEFAULT_ARTIFACT_PATH) -> FastAPI:
    """Factory, not a bare module global -- lets tests point a fresh app
    instance at a different (including deliberately broken) artifact
    path without any shared mutable state between app instances."""
    artifact_path = Path(artifact_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        try:
            artifact, manifest = load_artifact(artifact_path)
            if manifest.get("artifact_id") != EXPECTED_ARTIFACT_ID:
                raise ArtifactIntegrityError(
                    f"Loaded artifact_id {manifest.get('artifact_id')!r} does not match the "
                    f"expected {EXPECTED_ARTIFACT_ID!r} -- refusing to serve an unexpected artifact."
                )
            app.state.artifact = artifact
            app.state.manifest = manifest
            _log_event(event="startup", artifact_loaded=True, startup_seconds=round(time.perf_counter() - t0, 4))
        except Exception as exc:  # noqa: BLE001 -- a startup failure must surface via /health, never crash the process
            app.state.artifact = None
            app.state.manifest = None
            _log_event(event="startup", artifact_loaded=False, error_category=type(exc).__name__)
        yield
        app.state.artifact = None
        app.state.manifest = None

    app = FastAPI(
        title="Fraud Risk Scoring API (local demonstration only -- not deployed)",
        version=API_CONTRACT_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_UI_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict:
        loaded = getattr(app.state, "artifact", None) is not None
        return {"status": "ok" if loaded else "artifact_unavailable", "artifact_loaded": loaded}

    @app.get("/version")
    def version():
        manifest = getattr(app.state, "manifest", None)
        if manifest is None:
            return JSONResponse(status_code=503, content={"detail": "Artifact not loaded.", "error_category": "ArtifactUnavailable"})
        return {
            "api_contract_version": API_CONTRACT_VERSION,
            "artifact_id": manifest.get("artifact_id"),
            "artifact_version": manifest.get("artifact_version"),
            "manifest_schema_version": manifest.get("manifest_schema_version"),
            "artifact_sha256": manifest.get("artifact_sha256"),
        }

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(request: Request):
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        def error_response(status: int, error_category: str, detail: str):
            _log_event(
                request_id=request_id, api_contract_version=API_CONTRACT_VERSION,
                status=status, error_category=error_category, duration_seconds=round(time.perf_counter() - t0, 4),
            )
            return JSONResponse(status_code=status, content={"request_id": request_id, "detail": detail, "error_category": error_category})

        artifact = getattr(app.state, "artifact", None)
        manifest = getattr(app.state, "manifest", None)
        if artifact is None:
            return error_response(503, "ArtifactUnavailable", "Model artifact is not available.")

        raw_body = await request.body()
        try:
            payload = _parse_json_detecting_duplicate_keys(raw_body)
        except _DuplicateKeyError as exc:
            return error_response(422, "DuplicateField", str(exc))
        except json.JSONDecodeError:
            return error_response(422, "MalformedJSON", "Request body is not valid JSON.")

        if not isinstance(payload, dict) or not payload:
            return error_response(422, "EmptyOrInvalidRequest", "Request body must be a non-empty JSON object.")

        try:
            row_df = pd.DataFrame([payload])
            proba = predict_fraud_probability(artifact, row_df)
        except SchemaValidationError as exc:
            # The client sent this data themselves -- echoing back what was
            # wrong with it is normal API usability, not a privacy leak.
            # This detail is deliberately NOT included in the persisted log
            # event above/below, only in this one HTTP response.
            return error_response(422, "SchemaValidationError", str(exc))
        except Exception as exc:  # noqa: BLE001 -- any other failure becomes a controlled 500, never a leaked traceback
            _log_event(
                request_id=request_id, api_contract_version=API_CONTRACT_VERSION,
                status=500, error_category=type(exc).__name__, duration_seconds=round(time.perf_counter() - t0, 4),
            )
            return JSONResponse(status_code=500, content={"request_id": request_id, "detail": "Internal error while scoring the request."})

        duration = time.perf_counter() - t0
        _log_event(
            request_id=request_id, api_contract_version=API_CONTRACT_VERSION,
            artifact_version=manifest.get("artifact_version"), artifact_sha256=manifest.get("artifact_sha256"),
            duration_seconds=round(duration, 4), status=200,
        )
        return PredictResponse(
            request_id=request_id,
            fraud_probability=float(proba[0]),
            api_contract_version=API_CONTRACT_VERSION,
            artifact_version=manifest.get("artifact_version"),
            manifest_schema_version=manifest.get("manifest_schema_version"),
            artifact_sha256=manifest.get("artifact_sha256"),
        )

    return app


app = create_app()
