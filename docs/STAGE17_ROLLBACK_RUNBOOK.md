# Stage 17 — Rollback Runbook (local, free/local portfolio deployment)

This is the operator-facing runbook: the exact commands to run, in order,
to deploy, detect a bad release, and roll back. It was **tested end-to-end
for real** on 2026-10-27 — the commands below are the same ones
`scripts/run_stage17_deployment_demo.py` executed as subprocesses, not a
separate, untested description. Full narrative context, security notes,
and resource measurements are in
[`docs/STAGE17_DEPLOYMENT_REPORT.md`](STAGE17_DEPLOYMENT_REPORT.md); this
file is deliberately short and command-first.

Scope: local machine only, `127.0.0.1`-bound, no Docker (not installed on
this machine — see the report, §1). All commands assume the repository
root as the current directory.

---

## 0. Prerequisites

- `.venv/` already set up with the pinned dependencies (unchanged since Stage 14).
- `artifacts/frozen_rf_fraud_risk_v1.joblib` and its `.manifest.json` present (Stage 13).
- Nothing currently listening on the port you choose below.

```bash
shasum -a 256 artifacts/frozen_rf_fraud_risk_v1.joblib
# Expect: 2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04
```

If this checksum ever differs, **stop** — do not proceed with a deploy
until that is understood; something has changed the approved artifact.

---

## 1. Deploy the known-good release ("Version A")

```bash
./deploy/start_release.sh version_a 8000
./deploy/health_check.sh 8000 20
```

Expected: `health_check.sh` prints `READY: {"status":"ok","artifact_loaded":true}` and exits 0.

Verify the release is actually the one you think it is:

```bash
curl -s http://127.0.0.1:8000/version
```

Expected (exact values will match the artifact currently installed):
```json
{"api_contract_version":"v1","artifact_id":"frozen-rf-fraud-risk","artifact_version":"1.0.0","manifest_schema_version":"1.1.0","artifact_sha256":"2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04"}
```

Sanity-check inference:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/predict -H 'Content-Type: application/json' -d '{"...": "..."}'
# (build a full, valid payload from the schema -- see scripts/run_stage14_api_demo.py
# or scripts/run_stage17_deployment_demo.py for a working example)
```

Expected: `200 OK` with a `fraud_probability` between 0 and 1.

**Observed on the tested run (2026-10-27):** issued `08:12:52.386Z`, ready `08:12:53.850Z` (1.464s), `fraud_probability = 0.5201984944879683`. Full detail in the report, §6.

---

## 2. A new release is ready to deploy — stop the current one

```bash
./deploy/stop_release.sh version_a
```

Expected: `Stopped release 'version_a'.` No lingering process (`ps -p <pid>` after this returns nothing for that pid).

---

## 3. Deploy the new release ("Version B")

```bash
./deploy/start_release.sh version_b 8000 /path/to/new_or_candidate_artifact.joblib
./deploy/health_check.sh 8000 15
```

**If `health_check.sh` exits 0** → the new release passed its
release-acceptance check. Proceed normally (nothing to roll back).

**If `health_check.sh` exits non-zero** (as it did on 2026-10-27's tested
run, with the output `NOT READY after 8s: last body =
{"status":"artifact_unavailable","artifact_loaded":false}`) → **this is a
failed release. Go to Step 4.** Do not assume it will "warm up" — confirm
with one more direct check first:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/version
curl -s -X POST http://127.0.0.1:8000/v1/predict -H 'Content-Type: application/json' -d '{"...": "..."}'
```

On the tested run, `/health` returned HTTP 200 with body
`{"status":"artifact_unavailable","artifact_loaded":false}` (note: **always
HTTP 200** — read the body, never the status code, for this endpoint),
`/version` returned **503**, and `/v1/predict` returned **503**. All three
confirm the release did not become usable — the failure is confirmed
through actual commands, not assumed from one signal alone.

Check the process itself — it is likely still *running* even though the
*application* is not ready (this is the point of the exercise):

```bash
cat deploy/run/version_b.pid
ps -p "$(cat deploy/run/version_b.pid)"
```

Check why, without leaking any request data (the log never contains raw
payloads):

```bash
tail -5 deploy/run/version_b.log
# e.g.: {"event": "startup", "artifact_loaded": false, "error_category": "ArtifactIntegrityError"}
```

---

## 4. Roll back

```bash
./deploy/stop_release.sh version_b
```

If the failed release used a disposable/temporary artifact copy, remove
it now so the broken release is never left lying around:

```bash
rm -f /path/to/new_or_candidate_artifact.joblib /path/to/new_or_candidate_artifact.joblib.manifest.json
```

Redeploy the last known-good release — this **is** the rollback; there is
no special "rollback mode," only redeploying what was already proven
good:

```bash
./deploy/start_release.sh version_a 8000
./deploy/health_check.sh 8000 20
```

**Observed on the tested run:** stopped Version B `08:13:03.552Z` →
restart issued immediately → ready at `08:13:05.053Z` (1.501s).

---

## 5. Confirm the rollback actually worked

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/version
curl -s -X POST http://127.0.0.1:8000/v1/predict -H 'Content-Type: application/json' -d '{"...": "..."}'
```

Checklist — all four must hold before calling the rollback complete:

- [ ] `/health` → `{"status":"ok","artifact_loaded":true}`
- [ ] `/version`'s `artifact_sha256` matches the value recorded in Step 1 (and matches `shasum -a 256 artifacts/frozen_rf_fraud_risk_v1.joblib` on disk, right now)
- [ ] `/v1/predict` with a valid synthetic transaction returns `200 OK`
- [ ] No stale/broken release process is still running (`ps` shows nothing for the old `version_b` pid; `deploy/run/version_b.pid` no longer exists after `stop_release.sh`)

**Observed on the tested run:** all four held. The post-rollback
`/version` response was byte-for-byte identical to the pre-incident one,
and `fraud_probability` for the same synthetic input was identical
(`0.5201984944879683`) before and after.

---

## 6. Stand down

If this was a demonstration (not a real deployment you intend to keep
serving), stop the restored release too — this project does not run an
unnecessary persistent background service:

```bash
./deploy/stop_release.sh version_a
```

---

## Quick reference — exit codes

| Script | Exit 0 means | Non-zero means |
|---|---|---|
| `start_release.sh` | the OS process launched (NOT the same as ready — always follow with `health_check.sh`) | a release with that id is already running, or the venv/uvicorn invocation itself failed |
| `health_check.sh` | the release reported `status=ok, artifact_loaded=true` within the timeout | not ready within the timeout — treat as a failed release |
| `stop_release.sh` | the process was stopped (or was already stopped) and its pidfile removed | (practically always 0 by design — stopping an already-stopped release is not an error) |

## Known limitations of this runbook

See [`docs/STAGE17_DEPLOYMENT_REPORT.md`](STAGE17_DEPLOYMENT_REPORT.md) §11 — most notably: `/health`'s HTTP status code is always 200 regardless of readiness (read the body), there is no auto-restart-on-crash, and this has been tested against exactly one failure class (artifact checksum/corruption).
