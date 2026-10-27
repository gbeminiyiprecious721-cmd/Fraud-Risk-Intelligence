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
import random
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------------------
# Stage 15+ -- portfolio investigation dashboard (ui/dashboard.html)
#
# Deliberately attached to the module-level `app` object AFTER create_app()
# returns, NOT inside the factory above. `src/render_demo.py` builds its own,
# separately rate-limited/size-limited public app by calling create_app()
# itself; a route added inside the factory would silently also appear,
# unrestricted, on that public service. Attaching it here keeps this
# endpoint local-only, exactly mirroring how render_demo.py layers its own
# extra routes onto its own app instance after calling create_app().
# ---------------------------------------------------------------------------

# Stage 11's own global permutation-importance ranking (top 15, by original
# column), copied verbatim from the already-approved
# docs/_stage11_explainability_raw.json ("global_ranking_by_original_column_top15").
# Never recomputed here. This is a GLOBAL, model-level ranking -- the
# dashboard must present it as such, never as a per-transaction explanation.
STAGE11_GLOBAL_FEATURE_IMPORTANCE: list[tuple[str, float]] = [
    ("C13", 0.008137720972397648),
    ("C1", 0.007033049475817606),
    ("C14", 0.006093365643972535),
    ("C8", 0.005720210235720594),
    ("V258", 0.004878912008691257),
    ("C4", 0.004308088125071169),
    ("card6", 0.004149345447419782),
    ("C7", 0.003959770886404638),
    ("V257", 0.003828390879751714),
    ("V45", 0.0037167247845072467),
    ("C10", 0.003571967985725224),
    ("V274", 0.00341574607246177),
    ("C12", 0.0033148095393316646),
    ("V200", 0.003237809942552511),
    ("V242", 0.0031373676029885847),
]

# Numeric columns varied when generating a synthetic dashboard row: the
# numeric features from the ranking above (card6 is categorical, handled
# separately below). Kept short and deliberate -- not every one of the 392
# raw columns needs to move for a batch to show a believable score spread.
_DASHBOARD_VARY_NUMERIC = [
    "C1", "C13", "C14", "C8", "C4", "C7", "C10", "C12",
    "V258", "V257", "V45", "V274", "V200", "V242",
]
_DASHBOARD_PRODUCT_CODES = ["W", "C", "R", "H", "S"]
_DASHBOARD_CARD4 = ["visa", "mastercard", "american express", "discover"]
_DASHBOARD_CARD6 = ["debit", "credit"]
_DASHBOARD_EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "aol.com", "unknown"]

# Synthetic investigation-workflow status, purely a display/UX construct --
# carries no scientific content and is never derived from the risk score.
# Weighted so the queue looks "live" (mostly New/In Review) rather than
# uniformly random.
_DASHBOARD_STATUSES = ["New", "In Review", "Escalated", "Resolved"]
_DASHBOARD_STATUS_WEIGHTS = [45, 25, 10, 20]

# Quantile cut points separating Low | Medium | High | Critical, computed
# from each batch's OWN score distribution -- never a fixed/absolute
# threshold. `fraud_probability` is a relative ranking signal, not a
# calibrated probability, so there is no principled absolute cutoff to use
# instead; bands are only meaningful relative to the batch they're drawn from.
RISK_BAND_QUANTILES = (0.60, 0.85, 0.97)
RISK_BAND_LABELS = ("Low", "Medium", "High", "Critical")


def _synthetic_dashboard_row(schema, rng: random.Random) -> dict:
    """One fabricated, schema-valid transaction. Every value is generated
    here by this function -- never read from any real, training, or test
    dataset -- consistent with this dashboard's synthetic-only scope."""
    row: dict = {}
    for col in schema.raw_columns:
        row[col] = 100.0 if col in schema.numeric_columns else "unknown"
    for col in _DASHBOARD_VARY_NUMERIC:
        if col in schema.raw_columns:
            # Deliberately skewed (occasional high outlier) so the batch
            # produces a believable spread of relative risk scores -- an
            # illustrative distribution shape, not one measured from real
            # transaction data.
            row[col] = round(max(0.0, rng.expovariate(1 / 8.0) * rng.choice([1, 1, 1, 4, 9])), 2)
    row["TransactionAmt"] = round(rng.uniform(5.0, 5000.0), 2)
    if "ProductCD" in schema.raw_columns:
        row["ProductCD"] = rng.choice(_DASHBOARD_PRODUCT_CODES)
    if "card4" in schema.raw_columns:
        row["card4"] = rng.choice(_DASHBOARD_CARD4)
    if "card6" in schema.raw_columns:
        row["card6"] = rng.choice(_DASHBOARD_CARD6)
    if "P_emaildomain" in schema.raw_columns:
        row["P_emaildomain"] = rng.choice(_DASHBOARD_EMAIL_DOMAINS)
    return row


def _risk_bands_for(scores: list[float]) -> list[str]:
    if not scores:
        return []
    cutoffs = [float(np.quantile(scores, q)) for q in RISK_BAND_QUANTILES]
    return [RISK_BAND_LABELS[sum(1 for c in cutoffs if s > c)] for s in scores]


MAX_DASHBOARD_BATCH_SIZE = 200  # hard ceiling regardless of caller -- see build_dashboard_queue_payload


def build_dashboard_queue_payload(artifact, manifest: dict | None, n: int, seed: int) -> dict:
    """Shared batch-generation/scoring/banding logic for the investigation
    dashboard. Used by BOTH this module's own (locally-unrestricted)
    `/v1/dashboard/queue` route below AND `src/render_demo.py`'s separate,
    deliberately bounded public route -- a single source of truth so the
    two surfaces can never silently diverge in how a score or risk band is
    computed. Callers are responsible for checking `artifact is not None`
    first; this function assumes a loaded artifact."""
    n = max(1, min(n, MAX_DASHBOARD_BATCH_SIZE))
    rng = random.Random(seed)
    rows = [_synthetic_dashboard_row(artifact.schema, rng) for _ in range(n)]
    statuses = [rng.choices(_DASHBOARD_STATUSES, weights=_DASHBOARD_STATUS_WEIGHTS)[0] for _ in range(n)]
    row_df = pd.DataFrame(rows)
    scores = [float(p) for p in predict_fraud_probability(artifact, row_df)]
    bands = _risk_bands_for(scores)

    transactions = [
        {
            "transaction_id": f"TXN-DEMO-{seed:04d}-{i:04d}",
            "product_cd": row.get("ProductCD"),
            "card4": row.get("card4"),
            "card6": row.get("card6"),
            "amount": row.get("TransactionAmt"),
            "risk_score": round(score, 6),
            "risk_band": band,
            "status": status,
            "sequence": i,
        }
        for i, (row, score, band, status) in enumerate(zip(rows, scores, bands, statuses))
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "count": n,
        "synthetic": True,
        "model": {
            "artifact_version": manifest.get("artifact_version") if manifest else None,
            "artifact_sha256": manifest.get("artifact_sha256") if manifest else None,
            "api_contract_version": API_CONTRACT_VERSION,
        },
        "risk_band_quantiles": list(RISK_BAND_QUANTILES),
        "global_feature_importance": [
            {"feature": feature, "importance": round(importance, 6)}
            for feature, importance in STAGE11_GLOBAL_FEATURE_IMPORTANCE
        ],
        "transactions": transactions,
    }


@app.get("/v1/dashboard/queue")
def dashboard_queue(n: int = 48, seed: int = 2024):
    """Synthetic, schema-valid transactions scored through the real frozen
    artifact (via the same `predict_fraud_probability` the single-
    transaction endpoint uses -- no separate/duplicated inference logic),
    for the portfolio investigation-queue dashboard. Local-only: `n` and
    `seed` are caller-controlled here (bounded only by
    MAX_DASHBOARD_BATCH_SIZE). `src/render_demo.py` exposes its OWN,
    separately bounded version of this path for the public deployment --
    see that module's `dashboard_queue` for why."""
    artifact = getattr(app.state, "artifact", None)
    manifest = getattr(app.state, "manifest", None)
    if artifact is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Model artifact is not available.", "error_category": "ArtifactUnavailable"},
        )
    return build_dashboard_queue_payload(artifact, manifest, n, seed)
