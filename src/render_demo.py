"""Render-only portfolio adapter: same model/API, one public static demo page.

This module is additive. It does not change the approved local API or artifact.
Deployment is blocked until a trusted exact-checksum model artifact is provisioned
and external security/resource checks pass.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.api import DEFAULT_ARTIFACT_PATH, create_app

_ROOT = Path(__file__).resolve().parent.parent
_UI = _ROOT / "ui" / "index.html"
_MARKER = re.compile(r"const SYNTHETIC_TRANSACTION\s*=\s*(\{.*?\})\s*;", re.S)

# The real fixture is ~5.8KB of JSON; 16KB leaves generous margin for a
# legitimate browser request while still bounding memory/CPU spent on an
# oversized public POST before it ever reaches json.loads or the model.
MAX_PREDICT_BODY_BYTES = 16 * 1024

# Lightweight, free-tier-compatible rate limiting: an in-memory sliding
# window, no external store (Render's free plan has no Redis/paid add-ons
# and runs a single instance, so process-local state is adequate here).
# These numbers are conservative defaults for a portfolio demo, not
# measured load-test thresholds.
_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_PER_CLIENT = 20   # generous for a human manually re-clicking the demo
_RATE_LIMIT_MAX_GLOBAL = 120      # backstop regardless of per-client attribution (see _client_key)
_rate_limit_by_client: dict[str, list[float]] = {}
_rate_limit_global: list[float] = []


def _client_key(request: Request) -> str:
    """Best-effort client attribution behind a reverse proxy.

    Render's exact X-Forwarded-For semantics are not officially
    documented, and community reports conflict on whether Render replaces
    or appends to this header -- this has NOT been verified against a
    live Render deployment. We take the right-most entry (the hop closest
    to our own infrastructure), the generally-recommended safe default
    when a proxy's exact behaviour is unverified, and fall back to the
    direct TCP peer when the header is absent. Because this attribution
    is unverified, the GLOBAL limit above protects the service even if
    every request is ever attributed to the same key.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


def _prune_expired(timestamps: list[float], now: float) -> None:
    cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
    while timestamps and timestamps[0] < cutoff:
        timestamps.pop(0)


def _is_rate_limited(request: Request) -> bool:
    now = time.monotonic()
    _prune_expired(_rate_limit_global, now)
    if len(_rate_limit_global) >= _RATE_LIMIT_MAX_GLOBAL:
        return True
    key = _client_key(request)
    bucket = _rate_limit_by_client.setdefault(key, [])
    _prune_expired(bucket, now)
    if len(bucket) >= _RATE_LIMIT_MAX_PER_CLIENT:
        return True
    bucket.append(now)
    _rate_limit_global.append(now)
    if len(_rate_limit_by_client) > 10_000:  # bound worst-case memory growth
        _rate_limit_by_client.clear()
    return False


def _demo_payload() -> dict:
    page = _UI.read_text(encoding="utf-8")
    match = _MARKER.search(page)
    if not match:
        raise RuntimeError("Public demo fixture missing from approved UI")
    return json.loads(match.group(1))


DEMO_TRANSACTION = _demo_payload()

# Reuses the same FRAUD_API_ARTIFACT_PATH override Stage 17's
# deploy/asgi_launcher.py already established, so the hosted artifact's
# location (delivered out-of-band -- see docs/RENDER_DEPLOYMENT_PREPARATION.md)
# can be configured without a second prediction pathway or code change.
_artifact_override = os.environ.get("FRAUD_API_ARTIFACT_PATH")
app = create_app(Path(_artifact_override) if _artifact_override else DEFAULT_ARTIFACT_PATH)


@app.middleware("http")
async def synthetic_only_public_demo(request: Request, call_next):
    """Forbid arbitrary public scoring: only the fixed UI synthetic fixture.

    Also enforces (public-demo-only) rate limiting and a request-body size
    cap. Body size is checked both via the declared Content-Length (fast
    path, rejects before any read) and against the actually-buffered body
    (catches a Content-Length that understates the real size). This
    deliberately still uses `request.body()` -- not `request.stream()` --
    so Starlette's body cache is populated exactly as before and the real
    `/v1/predict` handler downstream can still read it; switching to a
    raw stream read here would exhaust the ASGI receive channel before
    the real handler gets a chance to read it. A residual gap this does
    NOT close: a chunked-transfer body with no Content-Length is still
    fully buffered by `request.body()` before our size check runs, so an
    adversarial client bypassing Content-Length can still force buffering
    up to whatever `request.body()` itself is willing to read. Declared
    as a known, disclosed limitation rather than solved with unverified
    stream-interception code that risks breaking real browser requests.
    """
    if request.url.path == "/v1/predict" and request.method == "POST":
        if _is_rate_limited(request):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests to this public demo. Please slow down."},
            )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_PREDICT_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large for this demo."},
                    )
            except ValueError:
                pass
        raw = await request.body()
        if len(raw) > MAX_PREDICT_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large for this demo."},
            )
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return JSONResponse(status_code=422, content={"detail": "Invalid demo JSON."})
        if payload != DEMO_TRANSACTION:
            return JSONResponse(
                status_code=403,
                content={"detail": "Public portfolio demo accepts its fixed synthetic transaction only."},
            )
    return await call_next(request)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    html = _UI.read_text(encoding="utf-8")
    html = html.replace(
        'const API_BASE = "http://127.0.0.1:8000";',
        "const API_BASE = window.location.origin;",
    )
    html = html.replace(
        "Local demonstration only — not a production system",
        "Public portfolio demonstration — synthetic data only",
    ).replace(
        "running locally on this machine (Stage 14 of this project)",
        "running on this demonstration service",
    ).replace(
        "not a deployed service and not a commercial product",
        "a hosted portfolio demonstration, not a commercial production system",
    )
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})
