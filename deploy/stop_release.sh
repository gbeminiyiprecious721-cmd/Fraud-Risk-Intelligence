#!/usr/bin/env bash
# Stage 17 -- stop a named local release started by start_release.sh.
#
# Controlled shutdown: SIGTERM first (uvicorn's graceful-shutdown
# signal), falls back to SIGKILL only if the process has not exited
# after a bounded wait. Always removes the pidfile so a later `status`
# or `start` check cannot be misled by a stale one.
#
# Usage: stop_release.sh <release_id>
set -euo pipefail

RELEASE_ID="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$ROOT_DIR/deploy/run/${RELEASE_ID}.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "No pidfile for release '$RELEASE_ID' -- nothing to stop."
    exit 0
fi

PID="$(cat "$PIDFILE")"

if kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID"
    for _ in $(seq 1 50); do
        kill -0 "$PID" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Release '$RELEASE_ID' (pid $PID) did not exit after SIGTERM within 5s; sending SIGKILL." >&2
        kill -KILL "$PID" || true
    fi
else
    echo "Release '$RELEASE_ID' (pid $PID) was not running."
fi

rm -f "$PIDFILE"
echo "Stopped release '$RELEASE_ID'."
