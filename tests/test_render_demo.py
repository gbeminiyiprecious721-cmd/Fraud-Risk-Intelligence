import importlib
import json

import pytest
from fastapi.testclient import TestClient

import src.render_demo as render_demo
from src.api import DEFAULT_ARTIFACT_PATH


@pytest.fixture(scope="module")
def client():
    with TestClient(render_demo.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    """Rate-limit buckets are module-level globals shared by every test
    against the module-scoped `client` fixture; reset them so one test's
    request volume can never make a later, unrelated test flaky."""
    render_demo._rate_limit_by_client.clear()
    render_demo._rate_limit_global.clear()
    yield
    render_demo._rate_limit_by_client.clear()
    render_demo._rate_limit_global.clear()


def test_fixed_synthetic_fixture_is_accepted_and_scored(client):
    resp = client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION)
    assert resp.status_code == 200
    body = resp.json()
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_arbitrary_payload_is_rejected_with_403(client):
    other_payload = dict(render_demo.DEMO_TRANSACTION)
    other_payload["C1"] = other_payload["C1"] + 1.0
    resp = client.post("/v1/predict", json=other_payload)
    assert resp.status_code == 403
    assert "fixed synthetic transaction only" in resp.json()["detail"]


def test_empty_payload_is_rejected_with_403(client):
    resp = client.post("/v1/predict", json={})
    assert resp.status_code == 403


def test_malformed_json_body_is_rejected_with_422_not_500(client):
    resp = client.post(
        "/v1/predict",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Invalid demo JSON."


def test_non_predict_routes_are_unaffected_by_the_restriction_middleware(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["artifact_loaded"] is True

    version = client.get("/version")
    assert version.status_code == 200
    assert "artifact_sha256" in version.json()


def test_single_transaction_page_rewrites_api_base_to_the_deployed_origin(client):
    """As of Stage 5, the dashboard is the homepage (/) and this page
    lives at its own /index.html route -- see test_render_dashboard_
    integration.py for the homepage-routing tests themselves."""
    resp = client.get("/index.html")
    assert resp.status_code == 200
    assert 'const API_BASE = "http://127.0.0.1:8000";' not in resp.text
    assert "const API_BASE = window.location.origin;" in resp.text


def test_single_transaction_page_rewrites_local_demo_language_to_hosted_demo_language(client):
    resp = client.get("/index.html")
    text = resp.text
    assert "Local demonstration only" not in text
    assert "Public portfolio demonstration" in text
    assert "running locally on this machine" not in text
    assert "running on this demonstration service" in text
    assert "not a deployed service and not a commercial product" not in text
    assert "a hosted portfolio demonstration, not a commercial production system" in text


def test_single_transaction_page_still_embeds_the_same_fixed_synthetic_fixture(client):
    resp = client.get("/index.html")
    match = render_demo._MARKER.search(resp.text)
    assert match is not None
    assert json.loads(match.group(1)) == render_demo.DEMO_TRANSACTION


def test_demo_transaction_matches_real_model_raw_schema():
    from src.production_pipeline import load_artifact
    from src.api import DEFAULT_ARTIFACT_PATH

    artifact, _ = load_artifact(DEFAULT_ARTIFACT_PATH)
    assert set(render_demo.DEMO_TRANSACTION.keys()) == set(artifact.schema.raw_columns)


def test_default_app_uses_default_artifact_path_when_override_unset(monkeypatch):
    monkeypatch.delenv("FRAUD_API_ARTIFACT_PATH", raising=False)
    reloaded = importlib.reload(render_demo)
    try:
        with TestClient(reloaded.app) as c:
            health = c.get("/health")
            assert health.status_code == 200
            assert health.json()["artifact_loaded"] is True
    finally:
        importlib.reload(render_demo)  # restore module state for subsequent tests


def test_artifact_path_override_env_var_is_honored(monkeypatch):
    monkeypatch.setenv("FRAUD_API_ARTIFACT_PATH", str(DEFAULT_ARTIFACT_PATH))
    reloaded = importlib.reload(render_demo)
    try:
        with TestClient(reloaded.app) as c:
            health = c.get("/health")
            assert health.status_code == 200
            assert health.json()["artifact_loaded"] is True
    finally:
        monkeypatch.delenv("FRAUD_API_ARTIFACT_PATH", raising=False)
        importlib.reload(render_demo)  # restore default-path module state for later tests


def test_artifact_path_override_pointing_at_nonexistent_file_fails_closed(monkeypatch):
    monkeypatch.setenv("FRAUD_API_ARTIFACT_PATH", "/nonexistent/does_not_exist.joblib")
    reloaded = importlib.reload(render_demo)
    try:
        with TestClient(reloaded.app) as c:
            health = c.get("/health")
            assert health.status_code == 200
            assert health.json()["artifact_loaded"] is False
    finally:
        monkeypatch.delenv("FRAUD_API_ARTIFACT_PATH", raising=False)
        importlib.reload(render_demo)  # restore default-path module state for later tests


def test_oversized_request_body_is_rejected_with_413(client):
    big_body = b"x" * (render_demo.MAX_PREDICT_BODY_BYTES + 1000)
    resp = client.post(
        "/v1/predict",
        content=big_body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"] == "Request body too large for this demo."


def test_per_client_rate_limit_returns_429_after_threshold(client, monkeypatch):
    monkeypatch.setattr(render_demo, "_RATE_LIMIT_MAX_PER_CLIENT", 3)
    monkeypatch.setattr(render_demo, "_RATE_LIMIT_MAX_GLOBAL", 1000)
    for _ in range(3):
        resp = client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION)
        assert resp.status_code == 200
    blocked = client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION)
    assert blocked.status_code == 429
    assert "Too many requests" in blocked.json()["detail"]


def test_global_rate_limit_backstop_returns_429_after_threshold(client, monkeypatch):
    monkeypatch.setattr(render_demo, "_RATE_LIMIT_MAX_PER_CLIENT", 1000)
    monkeypatch.setattr(render_demo, "_RATE_LIMIT_MAX_GLOBAL", 3)
    for _ in range(3):
        resp = client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION)
        assert resp.status_code == 200
    blocked = client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION)
    assert blocked.status_code == 429


def test_health_and_version_are_never_rate_limited(client, monkeypatch):
    monkeypatch.setattr(render_demo, "_RATE_LIMIT_MAX_PER_CLIENT", 1)
    monkeypatch.setattr(render_demo, "_RATE_LIMIT_MAX_GLOBAL", 1)
    exhausted = client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION)
    assert exhausted.status_code == 200
    blocked = client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION)
    assert blocked.status_code == 429
    for _ in range(10):
        assert client.get("/health").status_code == 200
        assert client.get("/version").status_code == 200


def test_client_key_uses_rightmost_x_forwarded_for_entry():
    from starlette.requests import Request as StarletteRequest

    scope = {
        "type": "http",
        "headers": [(b"x-forwarded-for", b"1.2.3.4, 10.0.0.1")],
        "client": ("127.0.0.1", 12345),
    }
    request = StarletteRequest(scope)
    assert render_demo._client_key(request) == "10.0.0.1"


def test_client_key_falls_back_to_direct_peer_when_header_absent():
    from starlette.requests import Request as StarletteRequest

    scope = {"type": "http", "headers": [], "client": ("203.0.113.5", 54321)}
    request = StarletteRequest(scope)
    assert render_demo._client_key(request) == "203.0.113.5"


def test_demo_fixture_loading_raises_clearly_if_marker_is_ever_removed(tmp_path, monkeypatch):
    broken_ui = tmp_path / "index.html"
    broken_ui.write_text("<html>no fixture here</html>", encoding="utf-8")
    monkeypatch.setattr(render_demo, "_UI", broken_ui)
    with pytest.raises(RuntimeError, match="Public demo fixture missing"):
        render_demo._demo_payload()
