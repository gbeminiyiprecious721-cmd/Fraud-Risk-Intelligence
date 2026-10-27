"""Stage 17 -- real local deployment and genuine rollback demonstration.

Runs the ACTUAL operator commands (`deploy/start_release.sh`,
`deploy/health_check.sh`, `deploy/stop_release.sh`) as real subprocess
calls against a real Uvicorn process bound to 127.0.0.1 only. Never
simulates the rollback through description alone.

Sequence (single port, sequential releases -- a real rollback replaces
a release, it does not run two releases side by side):
  1. Start Version A (the real, approved Stage 13 artifact). Verify
     health / version / predict / no-fit / bind-address.
  2. Stop Version A (as if about to deploy a new release).
  3. Prepare a disposable Version B release candidate: a COPY of the
     artifact file with one byte deliberately corrupted, so the
     existing Stage 13 `load_artifact` checksum check refuses to load
     it. The original frozen artifact and all approved source files
     are never touched.
  4. Start Version B. The OS process starts (it is "running"), but the
     FastAPI lifespan's artifact load fails -> app.state.artifact
     stays None. Detect this via the real bounded health check (not
     an assumption) and a real /v1/predict call (expect 503).
  5. Stop Version B and delete its disposable release-candidate
     artifact copy -- the broken release is never left active.
  6. Roll back: restart Version A (identical command, identical
     artifact path).
  7. Verify Version A is healthy again, a real synthetic prediction
     succeeds, and the served artifact_version/sha256 match the
     original approved release exactly.
  8. Stop Version A -- no unnecessary persistent background service is
     left running after the demonstration.

Writes a full structured, timestamped evidence report to
`docs/_stage17_deployment_demo_raw.json`.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python3"
DEPLOY = ROOT / "deploy"
DOCS_DIR = ROOT / "docs"
ORIGINAL_ARTIFACT = ROOT / "artifacts" / "frozen_rf_fraud_risk_v1.joblib"
ORIGINAL_MANIFEST = ORIGINAL_ARTIFACT.with_suffix(ORIGINAL_ARTIFACT.suffix + ".manifest.json")
CANDIDATES_DIR = DEPLOY / "release_candidates"
VERSION_B_DIR = CANDIDATES_DIR / "version_b"
VERSION_B_ARTIFACT = VERSION_B_DIR / "frozen_rf_fraud_risk_v1.joblib"
VERSION_B_MANIFEST = VERSION_B_ARTIFACT.with_suffix(VERSION_B_ARTIFACT.suffix + ".manifest.json")

sys.path.insert(0, str(ROOT))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}  [{now()}]", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, **kwargs)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def checksum(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def listening_bind_addresses(port: int) -> list[str]:
    """Real evidence of what interface a port is actually bound to,
    via the already-installed macOS `lsof` -- never assumed from the
    command line flags alone."""
    result = run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"])
    pattern = re.compile(r"(\S+:" + str(port) + r")(?!\d)")
    addrs = []
    for line in result.stdout.splitlines():
        m = pattern.search(line)
        if m:
            addrs.append(m.group(1))
    return addrs


def process_resource_usage(pid: int) -> dict:
    result = run(["ps", "-o", "rss,%cpu", "-p", str(pid)])
    lines = [ln for ln in result.stdout.splitlines() if ln.strip() and "RSS" not in ln]
    if not lines:
        return {"rss_kb": None, "cpu_percent": None}
    parts = lines[0].split()
    return {"rss_kb": int(parts[0]), "cpu_percent": float(parts[1])}


def build_valid_payload() -> dict:
    from src.production_pipeline import load_artifact

    artifact, _ = load_artifact(ORIGINAL_ARTIFACT)
    payload = {}
    for col in artifact.schema.raw_columns:
        if col in artifact.schema.numeric_columns:
            payload[col] = 100.0
        elif col == "ProductCD":
            payload[col] = "W"
        elif col == "card4":
            payload[col] = "visa"
        elif col == "card6":
            payload[col] = "debit"
        else:
            payload[col] = "unknown"
    payload["TransactionAmt"] = 125.0
    return payload


def main() -> dict:
    report: dict = {"generated_at": now(), "architecture": "local macOS process (no Docker -- see selection rationale in report)"}
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    valid_payload = build_valid_payload()
    report["port"] = port
    report["original_artifact_sha256"] = checksum(ORIGINAL_ARTIFACT)

    try:
        # --- Step 1: start Version A ---------------------------------
        section("1. Start Version A (real, approved Stage 13 artifact)")
        t0 = now()
        r = run([str(DEPLOY / "start_release.sh"), "version_a", str(port)])
        assert r.returncode == 0, "start_release.sh (version_a) failed"
        h = run([str(DEPLOY / "health_check.sh"), str(port), "20"])
        version_a_startup_evidence = {
            "start_command_exit_code": r.returncode,
            "start_command_stdout": r.stdout.strip(),
            "health_check_exit_code": h.returncode,
            "health_check_output": h.stdout.strip(),
            "issued_at": t0,
            "confirmed_ready_at": now(),
        }
        assert h.returncode == 0, "Version A did not become ready"

        health = httpx.get(f"{base_url}/health", timeout=5.0).json()
        version = httpx.get(f"{base_url}/version", timeout=5.0).json()
        predict = httpx.post(f"{base_url}/v1/predict", json=valid_payload, timeout=5.0)
        pid_a = int((DEPLOY / "run" / "version_a.pid").read_text().strip())
        bind_addrs = listening_bind_addresses(port)
        resource_a = process_resource_usage(pid_a)

        print(f"health={health} version={version} predict_status={predict.status_code} bind_addrs={bind_addrs} resource={resource_a}")
        assert health["status"] == "ok" and health["artifact_loaded"] is True
        assert version["artifact_sha256"] == report["original_artifact_sha256"]
        assert predict.status_code == 200
        assert all(addr.startswith("127.0.0.1:") for addr in bind_addrs) and bind_addrs, "port is not loopback-only"

        report["version_a_first_run"] = {
            "release_id": "version_a",
            "pid": pid_a,
            "startup_evidence": version_a_startup_evidence,
            "health": health,
            "version": version,
            "predict_status_code": predict.status_code,
            "predict_fraud_probability": predict.json().get("fraud_probability"),
            "bind_addresses": bind_addrs,
            "resource_usage": resource_a,
        }

        # --- Step 2: stop Version A (about to deploy a new release) --
        section("2. Stop Version A (preparing to deploy a new release)")
        r = run([str(DEPLOY / "stop_release.sh"), "version_a"])
        assert r.returncode == 0
        report["version_a_first_stop"] = {"stdout": r.stdout.strip(), "stopped_at": now()}
        time.sleep(0.3)

        # --- Step 3: prepare the disposable, deliberately broken Version B ---
        section("3. Prepare disposable Version B release candidate (ONE deliberate defect)")
        t_prep0 = time.perf_counter()
        VERSION_B_DIR.mkdir(parents=True, exist_ok=True)
        VERSION_B_MANIFEST.write_bytes(ORIGINAL_MANIFEST.read_bytes())  # unmodified manifest -> still expects the ORIGINAL checksum
        artifact_bytes = bytearray(ORIGINAL_ARTIFACT.read_bytes())
        corrupt_offset = len(artifact_bytes) // 2
        artifact_bytes[corrupt_offset] ^= 0xFF  # flip one byte -- simulates corruption during artifact provisioning
        VERSION_B_ARTIFACT.write_bytes(bytes(artifact_bytes))
        prep_seconds = time.perf_counter() - t_prep0
        version_b_checksum = checksum(VERSION_B_ARTIFACT)
        assert version_b_checksum != report["original_artifact_sha256"], "corruption step had no effect"
        assert checksum(ORIGINAL_ARTIFACT) == report["original_artifact_sha256"], "original artifact must be untouched"
        report["version_b_release_candidate"] = {
            "artifact_path": str(VERSION_B_ARTIFACT.relative_to(ROOT)),
            "preparation_seconds": round(prep_seconds, 4),
            "corrupted_byte_offset": corrupt_offset,
            "resulting_sha256": version_b_checksum,
            "manifest_expected_sha256": report["original_artifact_sha256"],
            "original_artifact_confirmed_unchanged": True,
        }

        # --- Step 4: attempt to deploy Version B ----------------------
        section("4. Attempt to start Version B (expected to fail readiness)")
        t0 = now()
        r = run([str(DEPLOY / "start_release.sh"), "version_b", str(port), str(VERSION_B_ARTIFACT)])
        process_start_exit_code = r.returncode
        pid_b = int((DEPLOY / "run" / "version_b.pid").read_text().strip()) if (DEPLOY / "run" / "version_b.pid").exists() else None
        time.sleep(1.0)  # let the lifespan attempt and fail to load the artifact
        is_alive = pid_b is not None and subprocess.run(["kill", "-0", str(pid_b)]).returncode == 0

        h = run([str(DEPLOY / "health_check.sh"), str(port), "8"])
        health_b = httpx.get(f"{base_url}/health", timeout=5.0).json()
        version_b_resp = httpx.get(f"{base_url}/version", timeout=5.0)
        predict_b = httpx.post(f"{base_url}/v1/predict", json=valid_payload, timeout=5.0)

        print(f"process_running={is_alive} health_check_exit={h.returncode} health_body={health_b} /version_status={version_b_resp.status_code} predict_status={predict_b.status_code}")

        # The defect must be real and must be detected through an actual
        # command -- not assumed. The OS process is running; the
        # application is NOT ready.
        assert is_alive, "Version B process did not even start -- not the intended failure mode"
        assert h.returncode != 0, "health_check.sh unexpectedly reported Version B as ready"
        assert health_b["status"] != "ok" or health_b["artifact_loaded"] is not True
        assert version_b_resp.status_code == 503
        assert predict_b.status_code == 503

        report["version_b_attempt"] = {
            "release_id": "version_b",
            "start_command_exit_code": process_start_exit_code,
            "pid": pid_b,
            "process_alive_after_1s": is_alive,
            "detected_at": now(),
            "health_check_exit_code": h.returncode,
            "health_check_output": h.stdout.strip(),
            "health_body": health_b,
            "version_endpoint_status_code": version_b_resp.status_code,
            "predict_status_code": predict_b.status_code,
            "conclusion": "process running != application ready: the OS process started, but src/api.py's lifespan caught "
            "the Stage 13 ArtifactIntegrityError (checksum mismatch) during artifact load, leaving app.state.artifact=None. "
            "GET /health still returns HTTP 200 by design -- readiness was correctly read from the JSON body, not the HTTP status code.",
        }

        # --- Step 5: stop Version B and delete the broken candidate ---
        section("5. Stop Version B and delete the disposable broken release candidate")
        r = run([str(DEPLOY / "stop_release.sh"), "version_b"])
        assert r.returncode == 0
        VERSION_B_ARTIFACT.unlink(missing_ok=True)
        VERSION_B_MANIFEST.unlink(missing_ok=True)
        try:
            VERSION_B_DIR.rmdir()
            CANDIDATES_DIR.rmdir()
        except OSError:
            pass
        report["version_b_teardown"] = {
            "stop_stdout": r.stdout.strip(),
            "artifact_candidate_deleted": not VERSION_B_ARTIFACT.exists(),
            "stopped_at": now(),
        }

        # --- Step 6: rollback -- restart Version A --------------------
        section("6. Rollback: restart Version A (identical command, identical artifact)")
        t0 = now()
        r = run([str(DEPLOY / "start_release.sh"), "version_a", str(port)])
        assert r.returncode == 0
        h = run([str(DEPLOY / "health_check.sh"), str(port), "20"])
        assert h.returncode == 0, "Version A did not recover after rollback"
        report["rollback_restart"] = {
            "issued_at": t0,
            "confirmed_ready_at": now(),
            "health_check_output": h.stdout.strip(),
        }

        # --- Step 7: verify restored release matches the original ----
        section("7. Verify restored Version A: health, prediction, and checksum identity")
        health2 = httpx.get(f"{base_url}/health", timeout=5.0).json()
        version2 = httpx.get(f"{base_url}/version", timeout=5.0).json()
        predict2 = httpx.post(f"{base_url}/v1/predict", json=valid_payload, timeout=5.0)
        print(f"health={health2} version={version2} predict_status={predict2.status_code}")

        assert health2["status"] == "ok" and health2["artifact_loaded"] is True
        assert predict2.status_code == 200
        assert version2["artifact_sha256"] == report["original_artifact_sha256"]
        assert version2 == version, "post-rollback /version does not match the pre-incident /version response"

        report["post_rollback_verification"] = {
            "health": health2,
            "version": version2,
            "predict_status_code": predict2.status_code,
            "predict_fraud_probability": predict2.json().get("fraud_probability"),
            "matches_original_pre_incident_version_response": version2 == version,
            "matches_original_artifact_checksum": version2["artifact_sha256"] == report["original_artifact_sha256"],
        }

    finally:
        # --- Step 8: stop Version A -- leave no lingering background service ---
        section("8. Stop Version A -- no persistent background service left running")
        r = run([str(DEPLOY / "stop_release.sh"), "version_a"])
        report["final_shutdown"] = {"stdout": r.stdout.strip(), "stopped_at": now()}
        VERSION_B_ARTIFACT.unlink(missing_ok=True)
        VERSION_B_MANIFEST.unlink(missing_ok=True)
        for d in (VERSION_B_DIR, CANDIDATES_DIR):
            try:
                d.rmdir()
            except OSError:
                pass

    final_checksum = checksum(ORIGINAL_ARTIFACT)
    report["original_artifact_checksum_unchanged_end_to_end"] = final_checksum == report["original_artifact_sha256"]
    assert report["original_artifact_checksum_unchanged_end_to_end"]

    out_path = DOCS_DIR / "_stage17_deployment_demo_raw.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull structured report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
