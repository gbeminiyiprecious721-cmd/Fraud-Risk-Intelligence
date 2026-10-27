"""Tests for the public, Render-hosted dashboard surface added in
src/render_demo.py: bounded/deterministic batch scoring, static asset
serving from the single hosted origin, and that this surface neither
weakens nor is starved by the existing /v1/predict protections."""
import pytest
from fastapi.testclient import TestClient

import src.render_demo as render_demo


@pytest.fixture(scope="module")
def client():
    with TestClient(render_demo.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_all_rate_limit_state():
    render_demo._rate_limit_by_client.clear()
    render_demo._rate_limit_global.clear()
    render_demo._dashboard_rate_limit_by_client.clear()
    render_demo._dashboard_rate_limit_global.clear()
    yield
    render_demo._rate_limit_by_client.clear()
    render_demo._rate_limit_global.clear()
    render_demo._dashboard_rate_limit_by_client.clear()
    render_demo._dashboard_rate_limit_global.clear()


# ---------------------------------------------------------------------
# Static asset serving from the single hosted origin
# ---------------------------------------------------------------------

def test_dashboard_html_is_served_at_its_own_path(client):
    resp = client.get("/dashboard.html")
    assert resp.status_code == 200
    assert "Fraud Risk Intelligence Dashboard" in resp.text


def test_dashboard_css_served_with_correct_content_type(client):
    resp = client.get("/dashboard.css")
    assert resp.status_code == 200
    assert "text/css" in resp.headers["content-type"]
    assert ":root" in resp.text


def test_dashboard_js_served_with_correct_content_type_and_unmodified(client):
    resp = client.get("/dashboard.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "ui" / "dashboard.js"
    assert resp.text == source.read_text(encoding="utf-8")


def test_root_and_dashboard_html_both_serve_the_dashboard(client):
    """As of Stage 5, the dashboard is the homepage: / and /dashboard.html
    are aliases of the same page."""
    root = client.get("/")
    alias = client.get("/dashboard.html")
    assert root.status_code == 200
    assert alias.status_code == 200
    assert root.text == alias.text
    assert "Fraud Risk Intelligence Dashboard" in root.text


def test_index_html_is_the_separate_single_transaction_route(client):
    """The single-transaction demo is no longer the homepage, but remains
    reachable at its own path, distinct from the dashboard."""
    index_resp = client.get("/index.html")
    assert index_resp.status_code == 200
    assert index_resp.text != client.get("/").text
    assert "Score a synthetic transaction" in index_resp.text


def test_index_page_links_to_dashboard_and_vice_versa(client):
    index_resp = client.get("/index.html")
    assert 'href="dashboard.html"' in index_resp.text
    dashboard_resp = client.get("/")
    assert 'href="index.html"' in dashboard_resp.text


# ---------------------------------------------------------------------
# Bounded, deterministic public batch endpoint
# ---------------------------------------------------------------------

def test_public_queue_ignores_client_supplied_n(client):
    resp = client.get("/v1/dashboard/queue", params={"n": 9999, "seed": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == render_demo.PUBLIC_DASHBOARD_BATCH_SIZE
    assert len(body["transactions"]) == render_demo.PUBLIC_DASHBOARD_BATCH_SIZE


def test_public_queue_seed_is_always_from_the_fixed_allowlist(client):
    seen_seeds = set()
    for client_seed in [0, 1, 2, 3, 4, 5, 999999, -17, 123456789]:
        resp = client.get("/v1/dashboard/queue", params={"seed": client_seed})
        assert resp.status_code == 200
        seen_seeds.add(resp.json()["seed"])
    assert seen_seeds.issubset(set(render_demo.PUBLIC_DASHBOARD_SEED_ALLOWLIST))


def test_public_queue_allowlisted_seeds_pass_through_unchanged(client):
    """ui/dashboard.js's 'Demo batch' dropdown sends exactly these 5
    values. They must map to themselves (not be shifted by the modulo
    fallback), so the dropdown produces the SAME batch here as it does
    against src/api.py's local, unrestricted route."""
    for allowlisted_seed in render_demo.PUBLIC_DASHBOARD_SEED_ALLOWLIST:
        resp = client.get("/v1/dashboard/queue", params={"seed": allowlisted_seed})
        assert resp.json()["seed"] == allowlisted_seed


def test_public_queue_is_deterministic_for_a_given_client_seed(client):
    first = client.get("/v1/dashboard/queue", params={"seed": 42}).json()
    second = client.get("/v1/dashboard/queue", params={"seed": 42}).json()
    assert first["transactions"] == second["transactions"]


def test_public_queue_scores_are_real_model_output_in_bounds(client):
    body = client.get("/v1/dashboard/queue", params={"seed": 7}).json()
    for txn in body["transactions"]:
        assert 0.0 <= txn["risk_score"] <= 1.0
    assert body["model"]["artifact_sha256"] is not None


# ---------------------------------------------------------------------
# Rate limiting is separate from, and does not break, /v1/predict
# ---------------------------------------------------------------------

def test_dashboard_rate_limit_returns_429_after_threshold(client, monkeypatch):
    monkeypatch.setattr(render_demo, "_DASHBOARD_RATE_LIMIT_MAX_PER_CLIENT", 3)
    monkeypatch.setattr(render_demo, "_DASHBOARD_RATE_LIMIT_MAX_GLOBAL", 1000)
    for _ in range(3):
        assert client.get("/v1/dashboard/queue").status_code == 200
    blocked = client.get("/v1/dashboard/queue")
    assert blocked.status_code == 429


def test_dashboard_global_rate_limit_backstop(client, monkeypatch):
    monkeypatch.setattr(render_demo, "_DASHBOARD_RATE_LIMIT_MAX_PER_CLIENT", 1000)
    monkeypatch.setattr(render_demo, "_DASHBOARD_RATE_LIMIT_MAX_GLOBAL", 3)
    for _ in range(3):
        assert client.get("/v1/dashboard/queue").status_code == 200
    assert client.get("/v1/dashboard/queue").status_code == 429


def test_exhausting_dashboard_rate_limit_does_not_affect_predict(client, monkeypatch):
    monkeypatch.setattr(render_demo, "_DASHBOARD_RATE_LIMIT_MAX_PER_CLIENT", 1)
    monkeypatch.setattr(render_demo, "_DASHBOARD_RATE_LIMIT_MAX_GLOBAL", 1)
    assert client.get("/v1/dashboard/queue").status_code == 200
    assert client.get("/v1/dashboard/queue").status_code == 429
    predict_resp = client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION)
    assert predict_resp.status_code == 200


def test_exhausting_predict_rate_limit_does_not_affect_dashboard(client, monkeypatch):
    monkeypatch.setattr(render_demo, "_RATE_LIMIT_MAX_PER_CLIENT", 1)
    monkeypatch.setattr(render_demo, "_RATE_LIMIT_MAX_GLOBAL", 1)
    assert client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION).status_code == 200
    assert client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION).status_code == 429
    dashboard_resp = client.get("/v1/dashboard/queue")
    assert dashboard_resp.status_code == 200


def test_health_and_version_still_unaffected_by_dashboard_rate_limit(client, monkeypatch):
    monkeypatch.setattr(render_demo, "_DASHBOARD_RATE_LIMIT_MAX_PER_CLIENT", 1)
    monkeypatch.setattr(render_demo, "_DASHBOARD_RATE_LIMIT_MAX_GLOBAL", 1)
    client.get("/v1/dashboard/queue")
    client.get("/v1/dashboard/queue")  # now rate-limited
    assert client.get("/health").status_code == 200
    assert client.get("/version").status_code == 200


# ---------------------------------------------------------------------
# Fail-closed artifact handling, mirroring src/api.py's behavior
# ---------------------------------------------------------------------

def test_public_queue_returns_503_when_artifact_unavailable(client):
    original_artifact = render_demo.app.state.artifact
    original_manifest = render_demo.app.state.manifest
    try:
        render_demo.app.state.artifact = None
        render_demo.app.state.manifest = None
        resp = client.get("/v1/dashboard/queue")
        assert resp.status_code == 503
        assert resp.json()["error_category"] == "ArtifactUnavailable"
    finally:
        render_demo.app.state.artifact = original_artifact
        render_demo.app.state.manifest = original_manifest


# ---------------------------------------------------------------------
# Regression: the existing synthetic-only single-transaction flow
# ---------------------------------------------------------------------

def test_existing_synthetic_only_predict_flow_is_unchanged(client):
    ok = client.post("/v1/predict", json=render_demo.DEMO_TRANSACTION)
    assert ok.status_code == 200
    other_payload = dict(render_demo.DEMO_TRANSACTION)
    other_payload["C1"] = other_payload["C1"] + 1.0
    rejected = client.post("/v1/predict", json=other_payload)
    assert rejected.status_code == 403
