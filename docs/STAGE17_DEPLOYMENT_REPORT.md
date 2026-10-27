# Stage 17 — Local Deployment and Rollback

Status: **implementation complete, awaiting scientific and engineering review.**
This document reports what was actually built and actually observed. A
separate, step-by-step **tested rollback runbook** lives at
[`docs/STAGE17_ROLLBACK_RUNBOOK.md`](STAGE17_ROLLBACK_RUNBOOK.md).

Scope reminder: this is a free/local ML engineering **portfolio**
deployment. Nothing here is exposed outside this machine, nothing costs
money, and no production SLA, on-call rotation, or external user traffic
exists or is claimed.

---

## 1. Deployment target and selection rationale

Section 2 of the approved instruction required inspecting this Mac for an
existing, functional Docker installation *before* choosing a target, and
explicitly forbade installing Docker (or anything else) without a separate
approval.

Real, read-only checks performed first:

```
$ docker --version        -> "command not found: docker"
$ docker info              -> "command not found: docker"
$ which docker             -> "docker not found"
$ ls /Applications/Docker.app -> "No such file or directory"
```

**Finding: Docker is not installed on this machine.** Per the explicit
instruction, Docker was not installed. The macOS local **process-management
approach** (instruction Section 4) was selected instead — no Docker image,
no container runtime, nothing new installed. Everything used already
existed on this Mac before Stage 17: `bash`, `curl`, `lsof`, `ps`, `kill`,
`nohup`, and the project's own `.venv` (Python 3.12.15, with `fastapi
0.162.0` / `uvicorn 0.54.0` already pinned since Stage 14 — unchanged).

**Resource implications:** zero new software, zero disk image, zero
container runtime daemon running in the background. The only persistent
footprint is the API's own process while it is deliberately running, which
Section 4 also requires *not* to be an unnecessary standing background
service — accordingly, every release in this stage's demonstration is
started, exercised, and then explicitly stopped; nothing is left running
at the end (verified in §9 below).

---

## 2. Files created or modified

**New (deployment layer only — no Stage 1–16 file touched):**

| File | Purpose |
|---|---|
| `deploy/asgi_launcher.py` | Thin ASGI entrypoint. Imports Stage 14's unmodified `src.api.create_app()` and lets a release choose its artifact path via the `FRAUD_API_ARTIFACT_PATH` environment variable — the *only* reason this file exists is to let "Version B" point at a different (broken) artifact copy without editing `src/api.py`. With the variable unset it is byte-identical in behavior to `src.api:app`. |
| `deploy/start_release.sh` | Starts a named release (`uvicorn deploy.asgi_launcher:app`) in the background via `nohup`, bound to `127.0.0.1` only, writes a pidfile + logfile under `deploy/run/`. |
| `deploy/stop_release.sh` | Reads the pidfile, sends `SIGTERM`, waits up to 5s, escalates to `SIGKILL` only if needed, removes the pidfile. |
| `deploy/health_check.sh` | Bounded, polling readiness probe. Reads the **JSON body** of `GET /health` (`status`, `artifact_loaded`), never the bare HTTP status code — see §7. |
| `scripts/run_stage17_deployment_demo.py` | The real orchestration script that ran the entire Version A → Version B failure → rollback sequence end-to-end and wrote the raw evidence file below. |
| `docs/_stage17_deployment_demo_raw.json` | Full structured, timestamped evidence from the real run (gitignored, like every other Stage's `_stageN_*_raw.json`). |
| `docs/STAGE17_DEPLOYMENT_REPORT.md` | This report. |
| `docs/STAGE17_ROLLBACK_RUNBOOK.md` | The separate, tested rollback runbook. |

**Modified:**

| File | Change |
|---|---|
| `.gitignore` | Added `docs/_stage17_*_raw.json`, `deploy/run/`, `deploy/release_candidates/` — all regenerated, ephemeral, or disposable; none hand-authored. |

**Not touched:** `src/api.py`, `src/production_pipeline.py`, `artifacts/frozen_rf_fraud_risk_v1.joblib`, its manifest, `ui/`, every test file, every prior stage's docs/notebooks. Confirmed by `git status`/`git diff` review before committing (§9).

---

## 3. Deployment architecture

```
operator (you)
  │
  ├─ deploy/start_release.sh <release_id> <port> [artifact_path]
  │     └─ nohup .venv/bin/python3 -m uvicorn deploy.asgi_launcher:app
  │           --host 127.0.0.1 --port <port> --workers 1
  │              │
  │              └─ deploy/asgi_launcher.py
  │                    └─ src.api.create_app(artifact_path)      <- Stage 14, UNCHANGED
  │                          └─ src.production_pipeline.load_artifact(...)   <- Stage 13, UNCHANGED
  │                                (verifies sha256 against the sidecar .manifest.json
  │                                 BEFORE joblib.load -- this is what catches Version B)
  │
  ├─ deploy/health_check.sh <port> <timeout>   (reads /health JSON body)
  └─ deploy/stop_release.sh <release_id>        (SIGTERM -> wait -> SIGKILL fallback)
```

One release occupies one port at a time; a "deployment" in this
demonstration means *replacing* the running release, exactly as a real
rollback does — Version A and Version B are never run concurrently.

**Why this target was chosen over Docker:** Docker was simply not
available, and installing it required separate approval this instruction
explicitly withheld. The process-based approach still demonstrates every
concept Stage 17 is meant to teach — a named release, a release-acceptance
check distinct from "the process started," a deliberate failure, and a
real, executed rollback — without introducing any new moving part (a
daemon, an image registry, a build step) that this portfolio project does
not otherwise need. Nothing about the chosen target weakens the exercise;
if Docker becomes available later, `deploy/asgi_launcher.py`'s environment-
variable artifact override and the same release-acceptance logic in
`deploy/health_check.sh` could be reused almost unchanged inside a
container's entrypoint.

---

## 4. Local start / stop commands

```bash
# Start a release named "version_a" on port 8000, using the default
# (real, approved) artifact:
./deploy/start_release.sh version_a 8000

# Check readiness (bounded 20s timeout):
./deploy/health_check.sh 8000 20

# Exercise it:
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/version
curl -s -X POST http://127.0.0.1:8000/v1/predict -H 'Content-Type: application/json' -d '{...}'

# Stop it:
./deploy/stop_release.sh version_a

# Start a release pointed at a DIFFERENT artifact file (used only for the
# Version B exercise -- never the real artifact path):
./deploy/start_release.sh version_b 8000 /absolute/path/to/other_artifact.joblib
```

All three scripts resolve the repository root from their own location, so
they can be invoked from any working directory. Logs and pidfiles live
under `deploy/run/<release_id>.{log,pid}` (gitignored).

---

## 5. Artifact provisioning and checksum verification

The Stage 13 artifact (`artifacts/frozen_rf_fraud_risk_v1.joblib`) is
intentionally excluded from git (too large, generated, not hand-authored —
decision made in Stage 13). Because the process-based approach runs
directly on this machine's existing filesystem, "provisioning" the
artifact to a release is simply: *it is already there, at its one fixed,
trusted, local path* — there is no copy, upload, or network transfer step
to secure, which is one real advantage of this target over a container
that would need an explicit mount or bake-in step.

Checksum verification happens automatically, for **every** release, A or
B, via the existing, unmodified `src/production_pipeline.load_artifact()`:
it recomputes the file's sha256 and compares it against the sidecar
`*.manifest.json`'s recorded `artifact_sha256` **before** calling
`joblib.load` — refusing to deserialize anything that does not match. This
Stage 13 control, built for a different reason (catching accidental
corruption or modification), is exactly the mechanism that makes Section
3/6's "verify checksum before loading" requirement already satisfied, and
is also the real mechanism that caught the deliberately broken Version B
below — nothing new had to be written to get this property.

---

## 6. Version A — known-good deployment (first run)

| Item | Value |
|---|---|
| Release identifier | `version_a` |
| Startup command | `./deploy/start_release.sh version_a 53338` (a free port chosen automatically by the demo script; any port works identically) |
| Issued at | `2026-10-27T08:12:52.386Z` |
| Confirmed ready at | `2026-10-27T08:12:53.850Z` |
| Startup/readiness time | **1.464 s** (process launch + model artifact load, cold start) |
| `GET /health` | `{"status": "ok", "artifact_loaded": true}` |
| `GET /version` | `{"api_contract_version": "v1", "artifact_id": "frozen-rf-fraud-risk", "artifact_version": "1.0.0", "manifest_schema_version": "1.1.0", "artifact_sha256": "2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04"}` |
| `POST /v1/predict` (valid synthetic transaction, built from the real schema) | `200 OK`, `fraud_probability = 0.5201984944879683` |
| Model fitting during startup/prediction? | **No.** `load_artifact`/`predict_fraud_probability` call only `joblib.load`, `.transform`, and `.predict_proba` — never `.fit`. Already independently demonstrated against this exact artifact by Stage 16's `test_lightweight_integration_real_artifact_full_path_is_sane` (imputer statistics and `n_estimators` proven unchanged across repeated predictions); not re-derived here, only re-cited. |
| Bind address (real `lsof` evidence) | `127.0.0.1:53338` — loopback only. No `0.0.0.0:*` or `*:53338` entry for this process anywhere in the full `lsof -nP -iTCP -sTCP:LISTEN` output. |
| Process resource usage at ready | RSS ≈ 213,440 KB (~208 MB); a transient 45.6% CPU snapshot taken while serving the measurement requests, not an idle baseline. |

All of the above is real output captured by `scripts/run_stage17_deployment_demo.py` and persisted verbatim in `docs/_stage17_deployment_demo_raw.json` (`version_a_first_run`).

---

## 7. Version B — deliberate failure and rollback exercise

**The one deliberate defect:** a disposable copy of the artifact file with
exactly one byte flipped (`artifact_bytes[len//2] ^= 0xFF`), placed at
`deploy/release_candidates/version_b/frozen_rf_fraud_risk_v1.joblib`,
accompanied by an **unmodified copy of the original manifest** (so the
manifest still records the *original, correct* checksum). The original
frozen artifact at `artifacts/frozen_rf_fraud_risk_v1.joblib` and every
approved source file were never opened for writing. This specific defect
was chosen because it exercises the project's own existing integrity
control end-to-end, and because it is a realistic failure mode (an
artifact corrupted or truncated during copy/provisioning to a new
release), not a toy failure invented only to make a test pass.

Real sequence, in order (also captured in `docs/_stage17_deployment_demo_raw.json`):

1. **Version A is healthy.** Confirmed in §6, then stopped (`2026-10-27T08:12:54.432Z`) as if a new release were about to be deployed in its place.
2. **Version B is prepared** (`0.0177 s` to copy + corrupt + write; trivial, since this is a byte-level file operation, not a build). Resulting checksum `1fd1478272c870d5de79034058d95b6dcf06034688eb53fc36532c280b70b867` ≠ the manifest's expected `2fb0cbc...`. Confirmed the **original** artifact's checksum was unaffected by this step.
3. **Version B is started** (`./deploy/start_release.sh version_b 53338 <path-to-corrupt-copy>`, exit code 0 — the OS process itself starts successfully).
4. **Version B fails to become ready.** `src/api.py`'s `lifespan` calls `load_artifact`, which raises `ArtifactIntegrityError` (checksum mismatch); the lifespan's existing `except Exception` catches it, logs `{"event": "startup", "artifact_loaded": false, "error_category": "ArtifactIntegrityError"}`, and leaves `app.state.artifact = None`. **The process is alive; the application is not ready.**
5. **The failure is detected through an actual command, not assumed:** `./deploy/health_check.sh 53338 8` exits **1** (`NOT READY after 8s`); independently, `GET /health` returns HTTP 200 with body `{"status": "artifact_unavailable", "artifact_loaded": false}`, `GET /version` returns **503**, and `POST /v1/predict` with the same valid synthetic transaction also returns **503**.
6. **Operator stops Version B** (`./deploy/stop_release.sh version_b`) and **deletes the disposable broken artifact copy** — confirmed by the demo script asserting the file no longer exists. The broken release is not left active at any point after this.
7. **Operator restores Version A** — the *same* command as in §6 (`./deploy/start_release.sh version_a 53338`), no special "rollback mode," because a rollback here is simply redeploying the last known-good release.
8. **Version A becomes healthy again** — confirmed ready at `2026-10-27T08:13:05.053Z`, `1.501 s` after the restart command (same cold-start cost as the first run, as expected — identical artifact, identical code path).
9. **A real synthetic prediction succeeds after rollback:** `POST /v1/predict` → `200 OK`, `fraud_probability = 0.5201984944879683` — the **exact same value** as before the incident (expected: identical input, identical model, identical code; see Stage 7/10/13/16's documented macOS floating-point note — not triggered here since this was a single process, no cross-run comparison was needed beyond this one).
10. **The restored release's model version and checksum match the original approved release exactly:** post-rollback `/version` response is byte-for-byte equal to the pre-incident `/version` response from §6, and `artifact_sha256` equals the original approved `2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04`.
11. **Version A is stopped** at the very end of the whole demonstration (§9) — no persistent background service is left running once the exercise concludes.

This was not simulated through description: every one of the 11 steps above corresponds to a real subprocess call, a real HTTP request, or a real file operation captured in `docs/_stage17_deployment_demo_raw.json`, independently re-verifiable by re-running `scripts/run_stage17_deployment_demo.py`.

---

## 8. Failure visibility and security

**Process running ≠ application ready ≠ model inference functioning ≠ high availability.** This stage's own Version B exercise is the concrete proof: the Version B *process* started successfully (`start_command_exit_code: 0`, the PID was alive after 1 second) while the *application* was never ready (`artifact_loaded: false`) and *inference* never functioned (`/v1/predict` → 503 for every request). None of this says anything about *availability* in the production sense — this is one process, on one laptop, with no replica, no load balancer, and no automatic restart-on-crash. **A single local instance does not eliminate a single point of failure; it has exactly one point of failure, and this stage never claims otherwise.** If this process is killed, or this Mac is asleep or offline, the service is simply down until a human restarts it — there is no monitoring stage wired to detect that yet (that is Stage 18's job, not claimed here).

**A real, non-obvious gap this exercise surfaced:** `GET /health` always returns HTTP **200**, even when the artifact failed to load (see `src/api.py`, written in Stage 14) — readiness lives in the JSON body, not the status code. `deploy/health_check.sh` was written to check the body deliberately for this reason; a naive `curl -f` (which only checks for a non-2xx status) would have reported Version B as "healthy." This is documented here rather than silently worked around, and is listed again under Known Limitations.

**Bounded timeouts:** `deploy/health_check.sh` always takes an explicit timeout argument and always returns within it (never an unbounded wait). The deployment demo used 20s for expected-healthy transitions and 8s for the expected-failure case.

**Logs free of payloads/secrets:** inspected the real `deploy/run/version_a.log` and `deploy/run/version_b.log` produced by this exercise (reproduced in full in the runbook) — every log line is either a Uvicorn access-log line (method/path/status only) or this project's existing Stage 14 structured `_log_event` output (`request_id`, `api_contract_version`, `status`, `duration_seconds`, and for Version B, `error_category: "ArtifactIntegrityError"`). No transaction field value, no exception message, and no file path beyond the already-non-secret artifact path appears anywhere in either log.

**No administrative endpoints, no public binding:** the only application routes are the three Stage 14 routes (`/health`, `/version`, `/v1/predict`) plus FastAPI's standard, already-existing, read-only `/docs` / `/openapi.json` / `/redoc` introspection routes (unchanged since Stage 14 — not administrative, not mutating). Every release in this stage bound to `127.0.0.1` only, confirmed by real `lsof` output in §6; no `--host 0.0.0.0` was ever used.

**Privileged containers / Linux capabilities:** not applicable — no container runtime is in use (§1). The process ran under this machine's ordinary login user (confirmed `whoami` → the local account, never `root`), with no elevated privileges requested or required.

---

## 9. Resource and cost verification

| Measurement | Value |
|---|---|
| Build/setup time | **N/A / zero incremental.** No Docker image to build. The `.venv` and pinned dependencies (`fastapi==0.162.0`, `uvicorn==0.54.0`) already existed, unchanged, since Stage 14. The only new "setup" artifacts are the four small deployment scripts above (one-time, not per-release). |
| Startup/readiness time (Version A, cold start) | **1.464 s** (§6) |
| Startup/readiness time (rollback restart of Version A) | **1.501 s** (§7, step 8) — consistent with the first run, as expected |
| Version B release-candidate preparation time | **0.0177 s** (a byte-level file copy, not a build) |
| Health-check / prediction outcomes | All recorded exactly in §6/§7 (200/200/200 for A; 200(body=unavailable)/503/503 for B) |
| Rollback recovery time (failure confirmed → Version A confirmed healthy again) | **1.745 s** (`08:13:03.308Z` → `08:13:05.053Z`) |
| Container/image size | **N/A** — no Docker used |
| Approximate resource usage | Version A at steady state: **RSS ≈ 208 MB**, transient CPU ≈ 46% while actively serving the measurement requests (not an idle baseline; Stage 14's own `run_stage14_api_demo.py` has the fuller idle/warm-latency numbers: median 0.0442s/p95 0.0462s over 50 warm requests, unchanged and not re-measured here). Version B's resource usage while briefly alive was not separately measured — a real gap, listed below. |
| Cost | **$0.** No paid service, no cloud compute, no container registry, no remote storage was used anywhere in this stage. |

**One honest caveat on "detection time":** `deploy/health_check.sh`'s probe for Version B was configured with an 8-second timeout, and the script reports "NOT READY after 8s" — but this reflects the **configured timeout**, not the true failure latency. The real evidence (`version_b.log`) shows the artifact-load failure and the resulting `startup_complete` log line appear within roughly one second of process start; a single health probe cannot distinguish "still starting" from "will never become ready" without an observation window, so this script conservatively waits out its full timeout before declaring failure. A tighter timeout (e.g. 3s) would have worked identically for this specific model (sub-second cold start) and is noted as a tuning opportunity, not implemented here since no evidence suggested the chosen 8s caused any problem.

Automated tests run for this stage: the existing fast suite, **`pytest tests/ -m "not heavy and not data_contract_full"` → 304 passed, 6 deselected, 6.53 s** — identical pass count to Stage 16's closure, confirming this stage introduced no regression (Stage 17 added no new `tests/` files; its own correctness evidence is the real, assertion-checked run of `scripts/run_stage17_deployment_demo.py` itself, which raises `AssertionError` on any unexpected outcome and did not).

The original fitted artifact's checksum was verified **unchanged** before this stage began and again after every step completed: `2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04` throughout (`original_artifact_checksum_unchanged_end_to_end: true` in the raw evidence file).

---

## 10. Reproduction steps

```bash
cd /path/to/Fraud-Risk-Intelligence
.venv/bin/python3 scripts/run_stage17_deployment_demo.py
```

This re-runs the entire Version A → Version B → rollback sequence for
real, on a freshly chosen free port, and rewrites
`docs/_stage17_deployment_demo_raw.json`. It cleans up `deploy/run/*.pid`
and the disposable `deploy/release_candidates/` directory itself; `*.log`
files are left in place for inspection and are gitignored.

For a manual, step-by-step walkthrough instead (recommended the first time, to see each step individually), use [`docs/STAGE17_ROLLBACK_RUNBOOK.md`](STAGE17_ROLLBACK_RUNBOOK.md).

---

## 11. Known limitations

1. **No Docker path exists yet.** If Docker becomes available and is separately approved, Section 3's containerized path (non-root user, read-only artifact mount, loopback-only port publishing) has not been built or exercised — only designed in principle (§3).
2. **`GET /health` always returns HTTP 200**, even when not ready (by design, from Stage 14) — any future automation calling this endpoint must read the JSON body, not the status code, exactly as `deploy/health_check.sh` does. A naive `curl -f` or a load balancer configured to key off HTTP status alone would misreport Version B as healthy.
3. **No automatic restart-on-crash, no supervisor, no monitoring.** If the Uvicorn process dies unexpectedly (not via the documented stop script), nothing restarts it or alerts anyone. This is explicitly out of scope for Stage 17 and is Stage 18's concern.
4. **Single instance, single point of failure, by construction** — stated plainly in §8, not minimized.
5. **Version B's resource usage while briefly running was not measured** (only its readiness/HTTP outcomes were) — a real, disclosed gap, not fabricated.
6. **The 8-second Version B detection timeout is a configuration choice, not a measured minimum failure-detection latency** — see the caveat in §9.
7. **This exercise used one specific defect class** (artifact checksum/corruption). Other deployment defect classes (e.g. a missing dependency, a bad port binding, an OOM during model load) were not exercised and may behave differently; the general principle demonstrated — process running vs. application ready, detected via an actual check, with a real rollback — should generalize, but that generalization itself has not been tested.
8. **No TLS, no authentication, no rate limiting** — unnecessary for a loopback-only, single-operator local demonstration, but would be required before this could ever be exposed beyond `127.0.0.1`, which this project does not do.
9. **Latency/throughput numbers are Stage 14's, not re-measured here** — this stage's script optimizes for correctness/rollback evidence, not performance benchmarking; no new SLA claim is made.

---

## 12. Interview-defense questions

**Q1: Your "deployment" here is a shell script and a background process, not Kubernetes or even Docker. Isn't that a toy?**
No — the mechanism is intentionally minimal because the *lesson* being demonstrated (process-running vs. application-ready vs. inference-functioning vs. genuine HA, and a real, executed rollback with release-acceptance checks) is orthogonal to the orchestration technology. The same release-acceptance logic in `deploy/health_check.sh` — read the body, not the status code; use a bounded timeout; detect, don't assume — is exactly what a Kubernetes readiness probe or an ECS health check does under the hood. Building it by hand once, and watching it actually catch a real failure, is more instructive than wiring up an orchestrator that would have hidden the same logic behind a YAML field.

**Q2: Why did Version B fail, and how would you have found out if you hadn't designed the failure yourself?**
It failed because the artifact's sha256 no longer matched its manifest's recorded checksum — `load_artifact` (an existing Stage 13 control, built for a different reason: catching accidental corruption) refused to deserialize it, and the FastAPI lifespan caught that exception and left the app unready rather than crashing. In a scenario I hadn't engineered myself, I would find out the same way this exercise did: `GET /health`'s body says `artifact_unavailable`, and the structured log line (`{"event": "startup", "artifact_loaded": false, "error_category": "ArtifactIntegrityError"}`) names the exact exception type without leaking any payload data — enough to diagnose "integrity check failed" immediately, without needing a stack trace in the logs.

**Q3: What would actually break if ten people used this "deployed" service at once, or if this Mac went to sleep?**
Nothing in this stage prevents concurrent requests — Uvicorn's single worker can interleave several requests, and nothing here is stateful across requests, so moderate concurrent load would mostly just queue. But the moment this Mac sleeps, loses power, or the process is killed outside the documented stop script, the service is simply gone with no failover, no second instance, and no alert — exactly the single-point-of-failure limitation stated in §8. This is a deliberate, disclosed property of a free/local portfolio deployment, not an oversight; a real production deployment would need at minimum a process supervisor with restart-on-crash and more than one instance behind a load balancer, neither of which this stage claims to provide.
