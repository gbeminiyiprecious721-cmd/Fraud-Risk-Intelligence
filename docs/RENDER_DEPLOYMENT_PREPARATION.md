# Optional Render deployment preparation (not deployed)

This optional work is distinct from the approved Stages 0–19 and does not modify ML training, testing, metrics, the model, or final_test.

## What is prepared
- `src/render_demo.py`: isolated single-origin FastAPI adapter serving the existing `ui/index.html`, with API_BASE rewritten to the deployed origin.
- Public POST /v1/predict is restricted to the single fixed, fully synthetic demonstration transaction embedded in the reviewed UI; it must not be used for arbitrary, personal or banking records.
- Existing `src/api.py` and `src/production_pipeline.py` remain unchanged. GET /health and /version remain available.
- `requirements-render.txt`: deliberately small runtime dependency list matching the original serialized artifact's libraries.

## Blocking prerequisite: trusted model file

The pretrained `artifacts/frozen_rf_fraud_risk_v1.joblib` is gitignored (by
design), ~16MB, and is **not on GitHub**. It is not present on Render.

Do NOT:
- upload this joblib to the public repo,
- download an arbitrary joblib from the internet,
- retrain during Render build,
- deploy and claim a working predictor before verifying the artifact.

### Artifact-delivery design (investigated and verified this round)

**Render's built-in "Secret Files" feature was investigated first and
rejected.** Per Render's own documentation, secret files are capped at a
combined 1MB per service and are uploaded by pasting plaintext into a
dashboard field -- about 16x too small for this binary artifact, and not
designed for binary content at all.

**Chosen mechanism: fetch at build time from a separate, private GitHub
repository the operator controls, verified against this public repo's
already-committed manifest before anything is allowed to start.**

- `deploy/render_fetch_artifact.sh` (new, this round) downloads the exact
  trusted bytes via the GitHub Contents API using a fine-grained,
  read-only, single-repo-scoped Personal Access Token, then computes the
  downloaded file's SHA-256 and compares it to
  `artifacts/frozen_rf_fraud_risk_v1.joblib.manifest.json`'s
  `artifact_sha256` field -- the same manifest this public repo already
  commits (it has never contained the model itself, only its
  checksum/config/schema). The build **fails closed**: any HTTP failure
  or checksum mismatch deletes the partial file and exits non-zero rather
  than let a bad or missing artifact silently start serving.
- Verified this round (in an isolated scratch copy, never against the
  real local artifact): missing-env-var guard exits 1; missing-manifest
  guard exits 1; a real GitHub API call against a nonexistent/unauthorized
  private repo returns a real HTTP error and the script deletes the
  partial file and exits 1 -- all three fail-closed paths exercised
  against real behavior, not simulated.
- The PAT is read only from a Render environment variable, is never
  written to any file in this repo, is passed to `curl` via a temp header
  file (not a command-line argument, so it cannot appear in `ps` output),
  and the script never echoes it.
- `src/render_demo.py` now reuses the exact same `FRAUD_API_ARTIFACT_PATH`
  override Stage 17's `deploy/asgi_launcher.py` already established --
  zero new prediction pathway, zero change to `src/api.py`.

### What the operator must still do (cannot be done from this session)

1. Create a **separate private** GitHub repository containing only the
   trusted `frozen_rf_fraud_risk_v1.joblib` (e.g.
   `gbeminiyiprecious721-cmd/fraud-risk-artifact-private`). This is a new
   piece of infrastructure under the operator's own account and requires
   their own GitHub action (and 2FA) -- it is not something this agent
   session creates or has credentials for.
2. Generate a **fine-grained** GitHub PAT scoped to **read-only Contents
   access on that one private repo only** (never a classic/all-repo
   token).
3. In the Render dashboard, set three environment variables on the
   service: `ARTIFACT_SOURCE_REPO`, `ARTIFACT_SOURCE_PATH`,
   `ARTIFACT_SOURCE_PAT` (values never committed anywhere).
4. Set the Render **Build Command** to run the fetch script before
   installing dependencies (see below).

## Pending deployment checks
1. ~~Run local tests for the new adapter~~ **Done this round** -- see
   `tests/test_render_demo.py` (10 tests: fixed fixture accepted and
   scored, arbitrary/empty payload rejected with 403, malformed JSON
   rejected with 422 not 500, `/health`+`/version` unaffected, UI
   same-origin rewrite, demo fixture matches the real model's raw
   schema). All pass locally against the real artifact.
2. Validate Python 3.12.15 or compatible Python and frozen scikit-learn/pandas/numpy/scipy/joblib versions using a disposable deployment environment. **Still pending** -- requires an actual Render build, not reproducible locally.
3. Verify trusted artifact delivery and checksum on Render itself, then test `/health` says `artifact_loaded: true`, `/version`, and a real browser-triggered POST against the live hosted URL. **Still pending** -- requires the operator steps above plus an actual deploy.
4. Add public rate limiting, request-body size restrictions, service resource tests, and appropriate HTTP caching/security policies before exposing the inference endpoint. **Still pending.**
5. Do not auto-deploy unreviewed changes to the public portfolio demo.

## Proposed Render free service (only after blockers)
- Source: `gbeminiyiprecious721-cmd/Fraud-Risk-Intelligence`
- Runtime: Python; region: Frankfurt; plan: free
- Python runtime requested: 3.12.15
- Build: `bash deploy/render_fetch_artifact.sh && pip install -r requirements-render.txt`
- Start: `uvicorn src.render_demo:app --host 0.0.0.0 --port $PORT`
- Required environment variables: `ARTIFACT_SOURCE_REPO`, `ARTIFACT_SOURCE_PATH`, `ARTIFACT_SOURCE_PAT` (see "What the operator must still do" above).
- Public base URL serves frontend and API on one origin.

Render service not yet created. No public demo URL exists. No hosted
prediction has succeeded -- this document describes a verified-locally
design, not a deployment.
