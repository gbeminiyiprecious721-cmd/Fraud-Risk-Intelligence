"""Stage 17 -- thin local-deployment ASGI entrypoint.

This file exists ONLY so a release's artifact path can be selected at
process-start time (via the `FRAUD_API_ARTIFACT_PATH` environment
variable), which is what lets the Stage 17 rollback exercise point a
disposable "Version B" release at a deliberately broken artifact copy
without touching `src/api.py` or introducing a second prediction
pathway. It reuses Stage 14's existing, unmodified `create_app()`
factory verbatim -- the HTTP contract (routes, request/response shape,
status codes) is completely unchanged. When the environment variable
is unset, this is byte-for-byte equivalent to importing `src.api:app`
directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.api import DEFAULT_ARTIFACT_PATH, create_app

_override = os.environ.get("FRAUD_API_ARTIFACT_PATH")
app = create_app(Path(_override)) if _override else create_app(DEFAULT_ARTIFACT_PATH)
