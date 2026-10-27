"""Render-only portfolio adapter: same model/API, one public static demo page.

This module is additive. It does not change the approved local API or artifact.
Deployment is blocked until a trusted exact-checksum model artifact is provisioned
and external security/resource checks pass.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.api import DEFAULT_ARTIFACT_PATH, create_app

_ROOT = Path(__file__).resolve().parent.parent
_UI = _ROOT / "ui" / "index.html"
_MARKER = re.compile(r"const SYNTHETIC_TRANSACTION\s*=\s*(\{.*?\})\s*;", re.S)


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
    """Forbid arbitrary public scoring: only the fixed UI synthetic fixture."""
    if request.url.path == "/v1/predict" and request.method == "POST":
        raw = await request.body()
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
