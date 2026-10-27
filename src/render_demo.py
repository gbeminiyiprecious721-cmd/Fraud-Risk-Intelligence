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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from src.api import DEFAULT_ARTIFACT_PATH, build_dashboard_queue_payload, create_app

_ROOT = Path(__file__).resolve().parent.parent
_UI_DIR = _ROOT / "ui"
_UI = _UI_DIR / "index.html"
_DASHBOARD_HTML = _UI_DIR / "dashboard.html"
_DASHBOARD_CSS = _UI_DIR / "dashboard.css"
_DASHBOARD_JS = _UI_DIR / "dashboard.js"
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

# Separate, independent rate-limit state for the dashboard's batch-scoring
# endpoint (GET /v1/dashboard/queue). Kept apart from the /v1/predict
# buckets above so loading the dashboard page can never starve, or be
# starved by, the single-transaction demo's own budget. The per-client
# limit is more generous than /v1/predict's because a single dashboard
# page load legitimately fires one fetch on load plus one per Refresh
# click, not one deliberate user action per request.
_DASHBOARD_RATE_LIMIT_MAX_PER_CLIENT = 30
_DASHBOARD_RATE_LIMIT_MAX_GLOBAL = 180
_dashboard_rate_limit_by_client: dict[str, list[float]] = {}
_dashboard_rate_limit_global: list[float] = []

# The public dashboard batch is deliberately BOUNDED and DETERMINISTIC --
# unlike the local-only /v1/dashboard/queue in src/api.py (which accepts
# arbitrary caller-supplied `n`/`seed`), the public version never accepts
# either. `n` is fixed to a small constant, well under
# api.MAX_DASHBOARD_BATCH_SIZE, to bound the inference cost of every
# request on Render's free tier; the seed is chosen from a small, fixed,
# enumerable allow-list via a `variant` index (not a raw seed), so the
# full set of possible public responses is finite and reviewable -- never
# an attacker-chosen seed driving arbitrary-looking "model output".
PUBLIC_DASHBOARD_BATCH_SIZE = 24
PUBLIC_DASHBOARD_SEED_ALLOWLIST = (101, 202, 303, 404, 505)


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


def _is_dashboard_rate_limited(request: Request) -> bool:
    """Same sliding-window algorithm as `_is_rate_limited`, against the
    dashboard's own, separate counters/thresholds (see their definitions
    above for why they're kept apart from the /v1/predict ones)."""
    now = time.monotonic()
    _prune_expired(_dashboard_rate_limit_global, now)
    if len(_dashboard_rate_limit_global) >= _DASHBOARD_RATE_LIMIT_MAX_GLOBAL:
        return True
    key = _client_key(request)
    bucket = _dashboard_rate_limit_by_client.setdefault(key, [])
    _prune_expired(bucket, now)
    if len(bucket) >= _DASHBOARD_RATE_LIMIT_MAX_PER_CLIENT:
        return True
    bucket.append(now)
    _dashboard_rate_limit_global.append(now)
    if len(_dashboard_rate_limit_by_client) > 10_000:  # bound worst-case memory growth
        _dashboard_rate_limit_by_client.clear()
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

    Also rate-limits GET /v1/dashboard/queue (its own, separate counters --
    see `_is_dashboard_rate_limited`); that route itself is what bounds
    `n`/the seed, so no body/payload check is needed here for it.

    For /v1/predict: enforces (public-demo-only) rate limiting and a
    request-body size cap. Body size is checked both via the declared
    Content-Length (fast
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
    if request.url.path == "/v1/dashboard/queue" and request.method == "GET":
        if _is_dashboard_rate_limited(request):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many dashboard requests. Please slow down."},
            )
        return await call_next(request)

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


@app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
def home():
    """The single-transaction demo -- a separate, secondary route as of
    Stage 5. `/` and `/dashboard.html` now serve the investigation
    dashboard (see `dashboard_page()` below), which is the primary public
    demo as of this stage; this page remains reachable at this exact path
    and is still linked from the dashboard's nav and footer."""
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard.html", response_class=HTMLResponse, include_in_schema=False)
def dashboard_page():
    """The investigation dashboard -- the homepage as of Stage 5, also
    reachable at its own `/dashboard.html` path (kept as an alias so
    existing links/bookmarks/tests referencing that exact path continue
    to work). No local-vs-hosted text rewriting needed here (unlike
    `home()` above): ui/dashboard.html's own copy never claims to be a
    local-only demonstration, and ui/dashboard.js resolves its own API
    base from `window.location.origin` directly (see that file) rather
    than needing a build-time/serve-time string replacement."""
    html = _DASHBOARD_HTML.read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/dashboard.css", include_in_schema=False)
def dashboard_css():
    return FileResponse(_DASHBOARD_CSS, media_type="text/css", headers={"Cache-Control": "no-store"})


@app.get("/dashboard.js", include_in_schema=False)
def dashboard_js():
    return FileResponse(_DASHBOARD_JS, media_type="application/javascript", headers={"Cache-Control": "no-store"})


@app.get("/v1/dashboard/queue")
def public_dashboard_queue(n: int = 48, seed: int = 2024):
    """Bounded, deterministic public counterpart to src/api.py's
    locally-unrestricted `/v1/dashboard/queue`. Accepts the SAME query
    parameter names (`n`, `seed`) so ui/dashboard.js needs no awareness of
    which backend it's talking to, but neither is used the way the local
    route uses them: `n` is ignored outright (always
    PUBLIC_DASHBOARD_BATCH_SIZE). `seed` is used directly ONLY when it is
    already one of the fixed PUBLIC_DASHBOARD_SEED_ALLOWLIST values --
    exactly the 5 values ui/dashboard.js's "Demo batch" dropdown sends --
    so that control produces the SAME batch on this public service as it
    would against the local, unrestricted route. Any other value (a
    caller bypassing the dropdown) is mapped via modulo into that same
    allowlist rather than used as the actual RNG seed. Either way, the
    set of possible public responses is small, fixed, and reviewable,
    regardless of what a caller sends.
    """
    artifact = getattr(app.state, "artifact", None)
    manifest = getattr(app.state, "manifest", None)
    if artifact is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Model artifact is not available.", "error_category": "ArtifactUnavailable"},
        )
    if seed in PUBLIC_DASHBOARD_SEED_ALLOWLIST:
        effective_seed = seed
    else:
        effective_seed = PUBLIC_DASHBOARD_SEED_ALLOWLIST[abs(seed) % len(PUBLIC_DASHBOARD_SEED_ALLOWLIST)]
    return build_dashboard_queue_payload(artifact, manifest, PUBLIC_DASHBOARD_BATCH_SIZE, effective_seed)
