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
trap 'rm -f "$HEADER_FILE"' EXIT
printf 'Authorization: token %s\n' "$ARTIFACT_SOURCE_PAT" > "$HEADER_FILE"

echo "Fetching trusted artifact from private repo (token not shown)..."
HTTP_STATUS=$(curl -sS -w "%{http_code}" -o "$DEST" \
  -H @"$HEADER_FILE" \
  -H "Accept: application/vnd.github.raw+json" \
  "https://api.github.com/repos/${ARTIFACT_SOURCE_REPO}/contents/${ARTIFACT_SOURCE_PATH}")

if [[ "$HTTP_STATUS" != "200" ]]; then
  echo "FATAL: artifact fetch failed (HTTP $HTTP_STATUS). Refusing to start with a missing/partial artifact." >&2
  rm -f "$DEST"
  exit 1
fi

EXPECTED_SHA256=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['artifact_sha256'])" "$MANIFEST")
ACTUAL_SHA256=$(shasum -a 256 "$DEST" | awk '{print $1}')

if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "FATAL: checksum mismatch. Manifest expects $EXPECTED_SHA256, fetched file is $ACTUAL_SHA256. Refusing to serve an unverified artifact." >&2
  rm -f "$DEST"
  exit 1
fi

echo "Artifact fetched and verified: sha256 matches the committed manifest ($ACTUAL_SHA256)."
