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
- **Done, verified this round:** the private artifact repository
  (`gbeminiyiprecious721-cmd/fraud-risk-model-artifact`, confirmed
  `PRIVATE`) now holds the exact trusted binary -- local, committed, and
  re-downloaded-and-rehashed copies all matched sha256
  `2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.
- **Done, verified this round:** ran `deploy/render_fetch_artifact.sh`
  for real against that real private repo (in an isolated scratch copy,
  never against the real local artifact), authenticated with this
  session's own already-authorized `gh` OAuth token (not a fine-grained
  PAT -- none exists yet; see below). It fetched the full 16,799,089-byte
  file via `curl` + the GitHub raw-content Accept header and verified its
  checksum, exit 0. This proves the curl-based raw fetch works correctly
  for a file this size; an earlier attempt at the same thing via `gh api`
  (not this script, not `curl`) failed with a client-side buffering error
  on a response this large -- the script deliberately uses plain `curl`,
  which does not have that problem.
- **Fixed this round:** the script's own checksum step used `shasum`,
  a macOS/Perl tool not guaranteed present on Render's Linux build image;
  replaced with a `python3`/`hashlib` computation (python3 is already a
  hard requirement of this service, so this adds no new dependency).
- **Added this round:** a 16KB request-body size cap and lightweight
  per-client + global in-memory rate limiting on `POST /v1/predict` (see
  "Public-demo safeguards" below) -- free-plan compatible, no external
  store.

### Investigated this round: HTTP 403 on an actual Render build

A real Render build of this branch hit a 403 fetching the artifact. Per
GitHub's own REST API documentation (verified directly against
docs.github.com, not from memory), **403 from the Contents API is not
unique to a misconfigured or insufficiently-scoped token** -- it is also
the exact status GitHub returns for primary *and secondary* rate
limiting, distinguishable only by response headers/body: a `Retry-After`
header, or `X-RateLimit-Remaining: 0` with `X-RateLimit-Reset`, and a
JSON `message` field stating which case applies. This is plausible on a
CI/build host like Render, whose outbound IP is shared across many
customers' builds.

**Fix implemented in `deploy/render_fetch_artifact.sh`:** the fetch now
inspects the real response headers and body on any non-200 result and
distinguishes the two cases:
- A detected rate-limit signal (`Retry-After` present, or
  `X-RateLimit-Remaining: 0`) retries up to 3 attempts total, waiting
  GitHub's own suggested duration (capped at 90s per attempt so a single
  wait can't exhaust Render's build-time budget).
- Any other failure (a genuine permission/scope/not-found problem) is
  **not** retried -- GitHub's own docs warn that retrying while *not*
  actually rate-limited risks the integration being banned -- and fails
  immediately with GitHub's own sanitized `message` field logged (e.g.
  `"Bad credentials"`, `"Resource not accessible by personal access
  token"`), defensively redacted of anything token-shaped, never a
  token/header/signed URL.

**Verified this round, not merely written:**
- Re-ran the real successful fetch against the real private repo
  (`gh` OAuth token, isolated scratch copy) -- unaffected by the
  refactor, exit 0, correct checksum, same as before.
- Re-ran the real failure case against a nonexistent/unauthorized repo
  -- single attempt (correctly not retried, since GitHub's real response
  there was `401 Bad credentials`, not a rate-limit signal), that exact
  sanitized message logged, completed in 0.24s.
- Built a local mock server reproducing GitHub's documented secondary-
  rate-limit response shape (403 + `Retry-After: 1` +
  `X-RateLimit-Remaining: 0` + the real documented message text) to
  verify the retry path itself, since deliberately triggering a real
  GitHub rate limit against a live account for a test would be
  irresponsible: confirmed 2 retries with real ~1s waits before a
  3rd-attempt success, and separately confirmed exhausting all 3
  attempts produces the correct "still rate-limited" failure. No token
  ever appeared in any output in any of these tests.

**Not yet confirmed:** this session has not reproduced the *original*
403 directly (no fine-grained PAT available here), so which of the two
documented causes actually applies has not been proven from a live
Render build log. **Safe diagnostic procedure for the operator:**
redeploy on Render with this fix in place and read the build log. It
now prints, per attempt: the HTTP status and GitHub's own sanitized
`message` text (never the token), and explicitly states whether a
rate-limit signal was detected. If the log shows retries with the
rate-limit message, the 403 was transient load-based and this fix
should resolve it on its own. If it shows an immediate, non-retried
failure with a message like `"Resource not accessible by personal
access token"` or `"Not Found"`, that specific sanitized text (safe to
paste anywhere -- it is GitHub's own public error text) identifies the
real cause precisely and points at the fine-grained PAT's granted scope
or selected-repository list, not at this script.

1. ~~Create a separate private GitHub repository~~ **Done** --
   `gbeminiyiprecious721-cmd/fraud-risk-model-artifact` exists, is
   private, and holds the verified artifact.
2. Generate a **fine-grained** GitHub PAT scoped to **read-only Contents
   access on that one private repo only** (never a classic/all-repo
   token). **Still pending** -- this session verified the fetch mechanism
   using its own `gh` session token, not a fine-grained PAT; the exact
   token type used in production has not been separately tested.
3. In the Render dashboard, set three environment variables on the
   service: `ARTIFACT_SOURCE_REPO=gbeminiyiprecious721-cmd/fraud-risk-model-artifact`,
   `ARTIFACT_SOURCE_PATH=frozen_rf_fraud_risk_v1.joblib`,
   `ARTIFACT_SOURCE_PAT` (values never committed anywhere).
4. Set the Render **Build Command** to run the fetch script before
   installing dependencies (see below).

## Public-demo safeguards (added this round)

Scoped only to `POST /v1/predict`; `GET /health`, `GET /version`, and
`GET /` are never rate-limited or body-size-checked, so Render's own
health checks (and casual browsing) can't be caught by this.

- **Body size cap:** 16KB (`MAX_PREDICT_BODY_BYTES` in
  `src/render_demo.py`). The real fixture is ~5.8KB, so there's generous
  margin. Checked twice: the declared `Content-Length` header (rejects
  before any read) and the actually-buffered body (catches an
  understated `Content-Length`). **Disclosed limitation:** a
  chunked-transfer body with no `Content-Length` is still fully buffered
  by Starlette's `request.body()` before our second check runs --
  switching to a raw stream read to close this gap was deliberately
  avoided because it would exhaust the ASGI receive channel before the
  real `/v1/predict` handler downstream gets to read it, which would
  break legitimate browser predictions. Verified live against a real
  running server: the real fixture POST succeeds (200); a real oversized
  POST is rejected (413) without affecting the next legitimate request.
- **Rate limiting:** in-memory sliding window, 20 requests/60s per
  attributed client plus a 120 requests/60s global backstop. No
  external store (Render free has none). **Disclosed limitation:**
  per-client attribution behind a reverse proxy uses the right-most
  `X-Forwarded-For` entry if present, falling back to the direct TCP
  peer -- Render's exact `X-Forwarded-For` semantics are not officially
  documented and community reports conflict on whether Render replaces
  or appends to it, so this attribution is **not verified against a
  live Render deployment**. The global backstop exists specifically so
  the service is still protected even if every request is ever
  attributed to the same key. Verified live against a real running
  server with 22 real sequential requests: the 21st (first to exceed the
  20/60s limit, after 2 already consumed earlier in the same test
  session) returned 429, `/health`/`/version` remained unaffected even
  after the limit was exhausted.

## Pending deployment checks
1. ~~Run local tests for the new adapter~~ **Done** -- see
   `tests/test_render_demo.py` (19 tests: fixed fixture accepted and
   scored, arbitrary/empty payload rejected with 403, malformed JSON
   rejected with 422 not 500, oversized body rejected with 413,
   per-client and global rate-limit 429s, `/health`+`/version` never
   rate-limited, UI same-origin rewrite, demo fixture matches the real
   model's raw schema, artifact-path override honored/fails closed). All
   19 pass locally against the real artifact; also verified live against
   a real running `uvicorn` process with real `curl` requests, not only
   via the ASGI TestClient.
2. Validate Python 3.12.15 or compatible Python and frozen scikit-learn/pandas/numpy/scipy/joblib versions using a disposable deployment environment. **Still pending** -- requires an actual Render build, not reproducible locally.
3. Verify trusted artifact delivery and checksum on Render itself, then test `/health` says `artifact_loaded: true`, `/version`, and a real browser-triggered POST against the live hosted URL. **Still pending** -- requires a fine-grained PAT (see above) plus an actual deploy.
4. ~~Add public rate limiting, request-body size restrictions~~ **Done** -- see "Public-demo safeguards" above. Service resource tests and HTTP caching/security policies beyond this are **still pending** and require an actual Render deployment to measure against.
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
