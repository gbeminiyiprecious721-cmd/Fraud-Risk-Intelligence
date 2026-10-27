#!/usr/bin/env bash
# Stage 17 -- start a named local release of the fraud-risk API.
#
# Uses only tools already present on macOS (bash, nohup, the project's
# own existing .venv). Binds strictly to 127.0.0.1 -- never a public
# interface. Runs `deploy/asgi_launcher.py`'s `app` (Stage 14's
# unmodified `create_app()` factory underneath).
#
# Usage: start_release.sh <release_id> <port> [artifact_path_override]
set -euo pipefail

RELEASE_ID="$1"
PORT="$2"
ARTIFACT_PATH="${3:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/deploy/run"
mkdir -p "$RUN_DIR"
PIDFILE="$RUN_DIR/${RELEASE_ID}.pid"
LOGFILE="$RUN_DIR/${RELEASE_ID}.log"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Release '$RELEASE_ID' is already running (pid $(cat "$PIDFILE"))." >&2
    exit 1
fi

cd "$ROOT_DIR"

if [ -n "$ARTIFACT_PATH" ]; then
    export FRAUD_API_ARTIFACT_PATH="$ARTIFACT_PATH"
fi

nohup "$ROOT_DIR/.venv/bin/python3" -m uvicorn deploy.asgi_launcher:app \
    --host 127.0.0.1 --port "$PORT" --workers 1 \
    > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

echo "Started release '$RELEASE_ID' (pid $(cat "$PIDFILE")) on 127.0.0.1:${PORT} -- log: $LOGFILE"
