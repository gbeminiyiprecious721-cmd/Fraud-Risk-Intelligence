# Stage 14 — Local Model-Serving API

**Status: implementation complete, not yet marked done.** A small, importable FastAPI application that serves predictions from the existing, trusted Stage 13 artifact — local only, never deployed, never a dashboard, never a commercial product. `final_test` was not accessed; the frozen Random Forest configuration and the Stage 13 artifact were not changed.

## 1. Dependency management

**Dry-run first**: `pip install --dry-run fastapi uvicorn` confirmed zero version change to any already-installed package. Installed the minimal `fastapi`+`uvicorn` **without** the `[standard]` extra — this project has no websockets, no hot-reload, no `.env` files, so `uvloop`/`httptools`/`watchfiles`/`websockets`/`python-dotenv` were unnecessary footprint and were not installed.

| Package | Version | Role |
|---|---|---|
| fastapi | 0.162.0 | the API framework |
| uvicorn | 0.54.0 | the ASGI server |
| starlette | 1.10.0 | fastapi's transitive dependency |
| pydantic / pydantic_core | 2.14.1 / 2.50.1 | request/response models |
| click, annotated-doc, annotated-types, typing-inspection, opentelemetry-api | — | transitive dependencies of the above |

**Zero version change** to numpy/pandas/scikit-learn/scipy/joblib/shap (verified before and after). `httpx` (0.28.1) was already present and is used by FastAPI's `TestClient`. `requirements.txt` regenerated via `pip freeze` and diffed against the prior file — confirmed the only changes are these 10 new lines, nothing else moved or changed.

**One honest environment observation, not acted on**: the newly-installed `starlette` emits a `StarletteDeprecationWarning` recommending `httpx2` in place of `httpx` for its `TestClient`. All 24 new tests pass cleanly against the existing `httpx` regardless — installing `httpx2` would be an additional library beyond what was approved this round, so it was not installed; noted here for whoever maintains this project's dependencies next.

## 2. API contract and versioning

Three endpoints, in `src/api.py`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | `{"status": "ok"\|"artifact_unavailable", "artifact_loaded": bool}` |
| `/version` | GET | API contract + artifact/manifest version identifiers |
| `/v1/predict` | POST | Score one transaction |

**Three independent version numbers, deliberately never conflated**: `API_CONTRACT_VERSION = "v1"` (this HTTP shape), `artifact_version = "1.0.0"` (the fitted binary, Stage 13), `manifest_schema_version = "1.1.0"` (the sidecar description, Stage 13's audit). Bumping one says nothing about the other two.

**`POST /v1/predict` request** — a flat JSON object of raw transaction fields (the same 392 columns `src/production_pipeline.py`'s `FeatureSchema.raw_columns` already requires); extra fields are accepted and ignored (Stage 13's documented contract), consistent with Pydantic's `extra="allow"` on the inbound model.

**`POST /v1/predict` response** (`PredictResponse`, `extra="forbid"` — the OUTBOUND shape is intentionally strict):
```json
{
  "request_id": "8c4b45b6-184b-4e40-bf79-24113dc9bdb0",
  "fraud_probability": 0.5201984944879683,
  "api_contract_version": "v1",
  "artifact_version": "1.0.0",
  "manifest_schema_version": "1.1.0",
  "artifact_sha256": "2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04"
}
```

**No decision threshold anywhere.** The response is a raw probability (`fraud_probability`, float in [0,1]). Stage 1 never approved a specific cost-justified threshold (Stage 10's cost-sweep stayed explicitly conditional); this API does not invent one. There is no `is_fraud` field, no "confirmed"/"flagged" label — only the number, exactly as every prior stage has reported it.

## 3. Model loading and inference

`create_app()` uses FastAPI's `lifespan` context manager to call `load_artifact(DEFAULT_ARTIFACT_PATH)` **once** at process startup, storing the result on `app.state`. Every request reuses that same already-fitted artifact — **nothing is ever re-fit, re-loaded, or re-validated per request**.

**Reused, not reimplemented**: `load_artifact`, `predict_fraud_probability` (which itself calls `validate_raw_input`/`align_raw_columns`) — all imported directly from `src/production_pipeline.py`. `src/api.py` adds exactly one new identity check beyond what `load_artifact` already does: at startup, it confirms the loaded manifest's `artifact_id` matches `EXPECTED_ARTIFACT_ID = "frozen-rf-fraud-risk"`, refusing to serve an unexpected artifact even if one somehow passed the checksum check (e.g., a different, validly-checksummed artifact placed at the same path by mistake).

**No upload endpoint exists.** `DEFAULT_ARTIFACT_PATH` is a fixed, project-local `Path` constant computed from `__file__` — never derived from a request, a query parameter, or any user input. Verified by a dedicated test (`test_module_has_no_artifact_upload_endpoint`) that the module source contains no `UploadFile`/multipart handling at all.

## 4. Input and error handling

| Failure | HTTP status | `error_category` |
|---|---|---|
| Malformed JSON | 422 | `MalformedJSON` |
| Empty body / empty object | 422 | `EmptyOrInvalidRequest` |
| Duplicate top-level field name | 422 | `DuplicateField` |
| Missing required column (Stage 13) | 422 | `SchemaValidationError` |
| Non-numeric value in a numeric column (Stage 13) | 422 | `SchemaValidationError` |
| Extra/unexpected field | 200 (ignored, per Stage 13's documented contract) | — |
| Unseen categorical value | 200 (handled by `OneHotEncoder(ignore)`, per Stage 5/12/13) | — |
| Permitted missing (`null`) numeric value | 200 (imputed, per Stage 5/12/13) | — |
| Artifact unavailable | 503 | `ArtifactUnavailable` |
| Any other unexpected failure | 500 | the exception's type name only — no message, no traceback |

**Duplicate-key detection is a genuine API-layer concern, not a Stage 13 one.** Standard `json.loads` silently keeps only the *last* value for a repeated top-level key — Stage 13's `validate_raw_input` operates on an already-parsed DataFrame and can never see that a duplicate existed. `src/api.py` therefore parses the raw request body itself with a custom `object_pairs_hook` that detects a repeated key *before* the standard library's default behavior erases the evidence, and returns a `DuplicateField` 422 — a failure mode this project would otherwise have silently missed end-to-end.

### The negative-`TransactionAmt` policy — investigated, not invented

Per instruction, this was investigated using **existing evidence only**: Stage 2's own dataset validation found the real `TransactionAmt` column's observed minimum is **0.251** (always positive, 590,540 real rows) — no negative or zero value has ever been observed in this dataset. Neither Stage 12's nor Stage 13's edge-case checks ever added a sign check to the approved pipeline; both found that a negative amount passes through without error and produces a real (if physically-implausible) prediction. **This API does not add a new rejection either** — doing so would be inventing a prohibition that was never part of the approved Stage 13 contract, which the approval explicitly forbids. The gap is instead documented here, again, as a concrete, named candidate for a future, separately-approved input-validation rule — exactly the limitation Stage 12 and Stage 13 already carried forward, now carried forward a third time rather than silently fixed mid-stage.

**Server errors never leak internals**: the generic 500 response body is `{"detail": "Internal error while scoring the request."}` — no stack trace, no exception message, no file path. The persisted log event for a 500 records only the exception's *type name* (e.g. `"ValueError"`), never its message.

## 5. Privacy and logging

Every request logs one structured, single-line JSON event via a dedicated `fraud_risk_api` logger: `request_id`, `api_contract_version`, `status`, `duration_seconds`, and on success `artifact_version`/`artifact_sha256`; on failure, `error_category` only. **The full transaction payload, any individual field value, and any exception message are never written to this log** — verified directly by two tests that plant a unique marker string in a request field (one a value that triggers a 200, one that triggers a `SchemaValidationError`) and assert the marker never appears anywhere in the captured log output, while confirming the *category* name does appear. The HTTP response to a validation error DOES include a human-readable detail message (the client already knows what it sent — that's ordinary API usability, not a leak) — the log and the response are deliberately held to different standards.

The service binds to `127.0.0.1` only, in both the API's own design and the real demonstration run (§8) — never `0.0.0.0`, never a public interface. No `--reload` and a single worker were used for the benchmark measurement, so autoreload file-watching and multi-worker scheduling never confound the latency numbers.

## 6. Testing

**24 new tests in `tests/test_api.py` — all passed on first run.** No bug was found this round, stated plainly rather than forcing a "bug found" narrative (unlike several prior stages, this round's fast feedback loop — small, focused functions delegating almost everything to already-tested Stage 13 code — left very little surface for a new defect). Coverage: valid request + full response schema, `/version` matching the real manifest exactly, malformed/empty/duplicate-key JSON, missing/non-numeric fields, extra fields ignored, numeric extremes, unseen categories, permitted missingness, the negative-amount policy, artifact-loading failure (a deliberately broken path), no-refitting-across-requests, repeated-request determinism, artifact-identity consistency across `/health`/`/version`/`/predict`, privacy-conscious logging (2 tests), and the project's standing `final_test`-isolation static checks.

`fastapi.testclient.TestClient` was used **as a context manager** (`with TestClient(app) as client:`) in every test — confirmed directly necessary, not assumed: `test_testclient_context_manager_triggers_lifespan_and_loads_artifact` asserts `/health` reports `artifact_loaded: True` only once the context manager has actually entered and lifespan has run.

**Full project suite**: **291 passed** (290 fast tests + the 1 heavyweight Stage 13 full-training regression test, which this round's instruction asked to be run in full, unlike Stage 13's own audit round) — 24 new, zero regressions. The Stage 13 artifact's sha256 was independently recomputed and confirmed **identical** (`2fb0cbc4...`) both before and after this round's work — nothing in Stage 14 touched it.

## 7. Local runtime demonstration (real, not simulated)

`scripts/run_stage14_api_demo.py` started the actual `src.api:app` as a genuine, separate Uvicorn process (`uvicorn src.api:app --host 127.0.0.1 --port <free-port> --workers 1`, no `--reload`), made real HTTP requests over localhost with `httpx`, and shut the process down at the end.

| Step | Result |
|---|---|
| Server startup + artifact load | **1.166s** (process spin-up + `load_artifact`, measured together — see the latency note below for why these are kept separate from per-request timing) |
| `GET /health` | `200 {"status": "ok", "artifact_loaded": true}` |
| `GET /version` | `200` with the exact artifact/manifest identifiers from §2 |
| `POST /v1/predict` (valid) | `200`, `fraud_probability=0.5202` |
| `POST /v1/predict` (missing field) | `422`, `{"detail": "Missing required raw column(s): ['TransactionAmt']", "error_category": "SchemaValidationError"}` |
| `GET /openapi.json` | `200`, `paths = ["/health", "/v1/predict", "/version"]` — the autogenerated OpenAPI documentation correctly lists all three endpoints |
| Server shutdown | Clean `terminate()`, confirmed no orphaned process remained (`ps aux | grep uvicorn` empty afterward) |

**Warm-request latency (50 requests, server already started — startup time excluded, per explicit instruction to separate the two):**

| Statistic | Value |
|---|---|
| min | 0.0424s |
| median | 0.0442s |
| p95 | 0.0462s |
| max | 0.0474s |

**This is a local, single-machine, single-worker, cold-cache-excluded measurement of one process on one laptop — not a production throughput number and not a latency SLA.** Stage 1 never defined a latency requirement for this system (it is an offline, human-investigation-prioritization framing, not a real-time one), and this report does not retroactively invent one. The ~44ms median is consistent with Stage 13's own measured ~0.06s/1000-row in-process inference time plus real HTTP/ASGI round-trip overhead for a single-row request.

## 8. Documentation and learning

**An ML model and an inference service are not the same artifact.** The `RandomForestClassifier` + `ColumnTransformer` pair (Stage 13's `.joblib`) is a pure function of its inputs — call it directly in a Python process and it has no concept of "a request," "a client," or "a bad actor." The service wrapped around it in `src/api.py` is what turns that pure function into something a network client can safely call: it adds a contract (what shape of input is accepted), a failure mode (what happens when that contract is violated), and an identity (which exact model version answered). Confusing the two is how a model that was carefully evaluated (Stages 10-12) ends up serving nonsense once it's "just" wrapped in a few lines of glue code that nobody treated as real engineering.

**API version and model version answer different questions.** "Is this still the same HTTP contract my client code was written against?" (`api_contract_version`) is a completely different question from "is this still the same fitted model that was evaluated?" (`artifact_version`/`artifact_sha256`). A future Stage 14.1 could add a new optional request field without touching the model at all (API version bump, same artifact); Stage 9 could in principle select a new configuration without changing a single line of the HTTP contract (new artifact, same API version). Keeping them independent is what makes either change safe to reason about in isolation.

**Input validation and predictive reliability are different guarantees.** Every test in §6 proves the API *behaves predictably* for a given input shape — it crashes on nothing, it returns a number in [0,1] for anything schema-valid, it rejects anything schema-invalid with a clear reason. None of that is a claim that the *returned number* is a *good* fraud-risk estimate for that particular input — that claim lives entirely in Stages 10-12's own evidence (AP 0.42784 on `final_test`, the segment caveats, the lack of demographic fairness evidence), unchanged and unextended by this stage.

**A local demonstration is not a production deployment.** §7's measurements describe one process, started and stopped by a script, on the machine this project has always run on, with no concurrent load, no authentication, no rate limiting, no TLS, no process supervisor restarting it if it crashes, and no monitoring watching it afterward. Every one of those gaps is real and is Stage 17's (Deployment) job, not something this stage's passing tests paper over.

**Notion connections**: *Python software engineering* — the `create_app()` factory pattern (rather than a bare module-level `app`) is a direct, concrete instance of "don't use global mutable state when a parameter will do," made necessary here specifically so tests could point a fresh instance at a deliberately broken artifact path without any cross-test leakage. *Modular design* — `src/api.py` imports Stage 13's functions and adds nothing to feature handling; the API layer's only real job was the HTTP-specific concerns (status codes, the duplicate-JSON-key edge case, logging) that genuinely don't belong in `production_pipeline.py`. *Testing* — the `test_artifact_loading_failure_is_reported_via_health_and_predict_not_a_crash` test is this stage's version of Stage 12/13's "what happens on a bad input" discipline, applied to the service's own startup path, not just its request path. *Reusable ML pipelines* — this stage is the concrete payoff of Stage 13's own investment: because `predict_fraud_probability` already existed as one clean, well-tested function, wiring it to HTTP took one `try`/`except` block, not a new implementation.

## 9. Reproduction commands

```bash
# Run the full test suite (incl. the real local Uvicorn demonstration's unit tests):
.venv/bin/python3 -m pytest tests/test_api.py -v

# Start the API by hand, locally, bound to 127.0.0.1 only:
.venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8000

# Re-run the full, real local demonstration (starts and stops its own server):
.venv/bin/python3 scripts/run_stage14_api_demo.py
```

## Three interview-defense questions

1. **"What happens if a client sends a field with the wrong type?"** — It depends on *which* wrong type, and the API answers precisely, not vaguely: a non-numeric value in a numeric field returns a `422 SchemaValidationError` with the exact column named (caught by Stage 13's `validate_raw_input`, reused unchanged); a genuinely malformed JSON body returns a `422 MalformedJSON` before the request ever reaches the model; neither case can silently become a nonsense prediction, because the schema check runs unconditionally, on every request, before `predict_proba` is ever called.
2. **"How would you roll out a new model version without breaking existing clients?"** — Build a new Stage 13 artifact (new `artifact_version`, new checksum), point a new `create_app(artifact_path=...)` instance at it, and keep `api_contract_version` at `"v1"` as long as the request/response *shape* hasn't changed — existing clients see an unchanged contract and a new `artifact_version` in every response, which is exactly the signal a monitoring system (Stage 18) would need to notice the swap happened.
3. **"Why doesn't the API just return `is_fraud: true/false`?"** — Because that would require a decision threshold, and no cost-justified threshold has ever been approved for this project (Stage 10's cost-sensitivity sweep stayed deliberately conditional across four labelled dollar-value assumptions, never collapsed into one number). Returning a raw probability keeps the API honest about what Stages 1-12 actually established, instead of silently promoting an unapproved default threshold into the one piece of code most likely to be trusted uncritically.

**Stage 14 is not marked complete.** Stage 15 has not begun, no dashboard was built, nothing was deployed, `final_test` was not accessed. Awaiting your scientific and engineering review.
