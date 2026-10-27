#!/usr/bin/env bash
# Render build-time artifact delivery for the synthetic-only portfolio demo
# (docs/RENDER_DEPLOYMENT_PREPARATION.md).
#
# Problem this solves: artifacts/frozen_rf_fraud_risk_v1.joblib (~16MB) is
# gitignored by design and must never be committed to this PUBLIC repo, yet
# the Render service needs those exact bytes at build time. Render's own
# "Secret Files" feature was investigated and rejected for this: Render
# caps combined secret-file size at 1MB per service, ~16x too small for
# this artifact (and that feature pastes plaintext into a dashboard field,
# not suited to a binary joblib).
#
# This script instead fetches the artifact from a SEPARATE, PRIVATE
# (not this repo) GitHub repository that only the operator controls, using
# a fine-grained, read-only, single-repo-scoped Personal Access Token
# supplied as a Render environment variable (never written to any file in
# this repo or to git history). It then verifies the downloaded bytes
# against the manifest this public repo already commits
# (artifacts/*.manifest.json has never contained the model itself -- only
# its checksum/config/schema) before anything is allowed to start serving.
#
# Required Render environment variables (set in the Render dashboard only):
#   ARTIFACT_SOURCE_REPO  e.g. "your-github-username/your-private-artifact-repo"
#   ARTIFACT_SOURCE_PATH  path to the joblib inside that private repo,
#                         e.g. "frozen_rf_fraud_risk_v1.joblib"
#   ARTIFACT_SOURCE_PAT   a fine-grained GitHub PAT with READ-ONLY "Contents"
#                         access to ONLY that one private repo -- never a
#                         classic/all-repo token.
#
# This script never prints ARTIFACT_SOURCE_PAT and never passes it on the
# command line (curl reads it from an env-populated header file instead,
# so it cannot appear in `ps` output or shell history).
#
# A 403 from this endpoint is NOT necessarily a misconfigured token.
# GitHub's own documented behavior: a 403 is also the exact status GitHub
# returns for primary AND secondary rate limiting (the response carries a
# `Retry-After` header, or `X-RateLimit-Remaining: 0` with an
# `X-RateLimit-Reset` epoch time, and a JSON body `message` explaining
# which case applies) -- this is plausible on a CI/build host like
# Render, whose outbound IP is shared across many customers' builds. This
# script detects that specific signal from the real HTTP response headers
# and retries with GitHub's own recommended backoff before concluding
# anything about the token itself; a non-rate-limit 403 (e.g. genuinely
# insufficient "Contents: Read" permission or the repo not being in the
# token's selected-repositories list) is NOT retried, since GitHub warns
# that retrying a non-rate-limit failure risks the integration being
# banned, and is reported immediately with GitHub's own sanitized error
# message (never a token, header, or signed URL).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$ROOT_DIR/artifacts"
ARTIFACT_NAME="frozen_rf_fraud_risk_v1.joblib"
DEST="$ARTIFACT_DIR/$ARTIFACT_NAME"
MANIFEST="$DEST.manifest.json"

: "${ARTIFACT_SOURCE_REPO:?ARTIFACT_SOURCE_REPO must be set (private repo, owner/name)}"
: "${ARTIFACT_SOURCE_PATH:?ARTIFACT_SOURCE_PATH must be set (path inside that private repo)}"
: "${ARTIFACT_SOURCE_PAT:?ARTIFACT_SOURCE_PAT must be set (fine-grained, read-only, single-repo PAT)}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "FATAL: $MANIFEST not found. This repo's committed manifest is required to verify the fetched artifact; refusing to proceed without it." >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR"

HEADER_FILE="$(mktemp)"
RESPONSE_HEADERS_FILE="$(mktemp)"
RESPONSE_BODY_FILE="$(mktemp)"
trap 'rm -f "$HEADER_FILE" "$RESPONSE_HEADERS_FILE" "$RESPONSE_BODY_FILE"' EXIT
printf 'Authorization: token %s\n' "$ARTIFACT_SOURCE_PAT" > "$HEADER_FILE"

# `_extract_sanitized_github_message FILE`: prints GitHub's own JSON error
# `message` field (public, non-secret -- GitHub never echoes a submitted
# token/header back in this body), truncated and defensively redacted of
# anything token-shaped, for safe inclusion in build logs.
_extract_sanitized_github_message() {
  python3 -c "
import json, re, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    msg = str(data.get('message', '(no message field)'))[:300]
except Exception:
    msg = '(non-JSON or unreadable error body)'
msg = re.sub(r'(ghp_|github_pat_|gho_|ghs_|ghu_)[A-Za-z0-9_]+', '[REDACTED_TOKEN]', msg)
print(msg)
" "$1" 2>/dev/null || echo "(error body unavailable)"
}

MAX_ATTEMPTS=3
ATTEMPT=1
FETCHED=0
while [[ "$ATTEMPT" -le "$MAX_ATTEMPTS" ]]; do
  echo "Fetching trusted artifact from private repo, attempt ${ATTEMPT}/${MAX_ATTEMPTS} (token not shown)..."
  HTTP_STATUS=$(curl -sS -w "%{http_code}" -o "$RESPONSE_BODY_FILE" -D "$RESPONSE_HEADERS_FILE" \
    -H @"$HEADER_FILE" \
    -H "Accept: application/vnd.github.raw+json" \
    "https://api.github.com/repos/${ARTIFACT_SOURCE_REPO}/contents/${ARTIFACT_SOURCE_PATH}")

  if [[ "$HTTP_STATUS" == "200" ]]; then
    mv "$RESPONSE_BODY_FILE" "$DEST"
    FETCHED=1
    break
  fi

  SAFE_MESSAGE="$(_extract_sanitized_github_message "$RESPONSE_BODY_FILE")"
  RETRY_AFTER="$(grep -i '^retry-after:' "$RESPONSE_HEADERS_FILE" 2>/dev/null | awk '{print $2}' | tr -d '\r' || true)"
  RATE_REMAINING="$(grep -i '^x-ratelimit-remaining:' "$RESPONSE_HEADERS_FILE" 2>/dev/null | awk '{print $2}' | tr -d '\r' || true)"
  echo "GitHub API error on attempt ${ATTEMPT}/${MAX_ATTEMPTS}: HTTP ${HTTP_STATUS} -- ${SAFE_MESSAGE}" >&2

  if [[ "$HTTP_STATUS" == "403" ]] && { [[ -n "$RETRY_AFTER" ]] || [[ "$RATE_REMAINING" == "0" ]]; }; then
    WAIT_SECONDS="${RETRY_AFTER:-60}"
    if ! [[ "$WAIT_SECONDS" =~ ^[0-9]+$ ]]; then
      WAIT_SECONDS=60
    fi
    if [[ "$WAIT_SECONDS" -gt 90 ]]; then
      # Capped deliberately: a Render build has its own time budget, and
      # blindly sleeping for whatever GitHub asks (sometimes much longer)
      # could exhaust that budget on a failure this script can't fix by
      # waiting longer anyway. A capped, clearly-logged short wait plus a
      # fast, honest failure beats a build that times out silently.
      echo "GitHub asked us to wait ${WAIT_SECONDS}s; capping the in-build wait to 90s to avoid exhausting Render's own build-time budget. If this keeps failing, redeploy again in a few minutes." >&2
      WAIT_SECONDS=90
    fi
    echo "Detected GitHub's rate-limit signal (Retry-After or X-RateLimit-Remaining: 0) -- this is NOT necessarily a token problem. Waiting ${WAIT_SECONDS}s per GitHub's documented guidance before retrying." >&2
    sleep "$WAIT_SECONDS"
    ATTEMPT=$((ATTEMPT + 1))
    continue
  fi

  # Any other failure (including a non-rate-limited 403) is NOT retried:
  # GitHub documents that repeated requests while not rate-limited risk
  # the integration being banned, and retrying would not fix a genuine
  # permission/not-found condition.
  echo "FATAL: artifact fetch failed (HTTP ${HTTP_STATUS}), not a recognized rate-limit signal -- see the sanitized message above. Refusing to start with a missing/partial artifact." >&2
  exit 1
done

if [[ "$FETCHED" != "1" ]]; then
  echo "FATAL: artifact fetch still rate-limited after ${MAX_ATTEMPTS} attempts. Refusing to start with a missing/partial artifact." >&2
  exit 1
fi

EXPECTED_SHA256=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['artifact_sha256'])" "$MANIFEST")
# python3's hashlib (not `shasum`/`sha256sum`) is used here deliberately:
# `shasum` is a macOS/Perl tool not guaranteed present on Render's Linux
# build image, and this project already requires python3 as the runtime.
ACTUAL_SHA256=$(python3 -c "
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
" "$DEST")

if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "FATAL: checksum mismatch. Manifest expects $EXPECTED_SHA256, fetched file is $ACTUAL_SHA256. Refusing to serve an unverified artifact." >&2
  rm -f "$DEST"
  exit 1
fi

echo "Artifact fetched and verified: sha256 matches the committed manifest ($ACTUAL_SHA256)."
