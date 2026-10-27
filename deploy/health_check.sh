#!/usr/bin/env bash
# Stage 17 -- bounded release-acceptance / readiness check.
#
# IMPORTANT: `GET /health` always returns HTTP 200 by design (see
# `src/api.py` -- it reports readiness in the JSON BODY, not the HTTP
# status code), so this script treats a release as READY only when the
# body's own "status"=="ok" AND "artifact_loaded"==true. A process
# that is merely running (the OS scheduled it) is not the same thing
# as an application that is ready to serve -- see Stage 17's report,
# section "Failure visibility" for why that distinction matters.
#
# Usage: health_check.sh <port> [timeout_seconds]
# Exit 0 and print "READY: <body>" if ready within the timeout.
# Exit 1 and print "NOT READY ...: <last body>" otherwise.
set -euo pipefail

PORT="$1"
TIMEOUT="${2:-15}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python3"

DEADLINE=$(( $(date +%s) + TIMEOUT ))
LAST_BODY=""

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    LAST_BODY="$(curl -s -m 2 "http://127.0.0.1:${PORT}/health" || true)"
    READY="$(printf '%s' "$LAST_BODY" | "$PY" -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print("yes" if d.get("status") == "ok" and d.get("artifact_loaded") is True else "no")
except Exception:
    print("no")
' 2>/dev/null || echo "no")"
    if [ "$READY" = "yes" ]; then
        echo "READY: $LAST_BODY"
        exit 0
    fi
    sleep 0.3
done

echo "NOT READY after ${TIMEOUT}s: last body = ${LAST_BODY:-<no response>}" >&2
exit 1
