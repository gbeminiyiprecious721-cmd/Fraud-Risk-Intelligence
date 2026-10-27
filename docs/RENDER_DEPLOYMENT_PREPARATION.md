# Optional Render deployment preparation (not deployed)

This optional work is distinct from the approved Stages 0–19 and does not modify ML training, testing, metrics, the model, or final_test.

## What is prepared
- `src/render_demo.py`: isolated single-origin FastAPI adapter serving the existing `ui/index.html`, with API_BASE rewritten to the deployed origin.
- Public POST /v1/predict is restricted to the single fixed, fully synthetic demonstration transaction embedded in the reviewed UI; it must not be used for arbitrary, personal or banking records.
- Existing `src/api.py` and `src/production_pipeline.py` remain unchanged. GET /health and /version remain available.
- `requirements-render.txt`: deliberately small runtime dependency list matching the original serialized artifact's libraries.

## Blocking prerequisite: trusted model file
The pretrained `artifacts/frozen_rf_fraud_risk_v1.joblib` is gitignored (by design) and is **not on GitHub**. It is not present on Render.

Do NOT:
- upload this joblib to the public repo,
- download an arbitrary joblib from the internet,
- retrain during Render build,
- deploy and claim a working predictor before verifying the artifact.

The operator must provision the *exact locally trusted* model via an access-controlled artifact delivery mechanism, compare SHA-256 to the committed sidecar manifest, and ensure that the bytes are accessible to the running service at the expected `artifacts/` path. The sidecar checksum detects changes but does **not** establish that an untrusted joblib is safe to deserialize.

## Pending deployment checks
1. Run local tests for the new adapter (fixed synthetic payload accepted, different input blocked, no data leakage, UI same-origin request).
2. Validate Python 3.12.15 or compatible Python and frozen scikit-learn/pandas/numpy/scipy/joblib versions using a disposable deployment environment.
3. Verify trusted artifact delivery and checksum, then test `/health` says `artifact_loaded: true`, `/version`, and a real browser-triggered POST.
4. Add public rate limiting, request-body size restrictions, service resource tests, and appropriate HTTP caching/security policies before exposing the inference endpoint.
5. Do not auto-deploy unreviewed changes to the public portfolio demo.

## Proposed Render free service (only after blockers)
- Source: `gbeminiyiprecious721-cmd/Fraud-Risk-Intelligence`
- Runtime: Python; region: Frankfurt; plan: free
- Python runtime requested: 3.12.15
- Build: `pip install -r requirements-render.txt`
- Start: `uvicorn src.render_demo:app --host 0.0.0.0 --port $PORT`
- Public base URL serves frontend and API on one origin.

Render service not yet created. No public demo URL exists.
