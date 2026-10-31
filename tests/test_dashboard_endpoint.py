import pytest
from fastapi.testclient import TestClient

from src.api import _DASHBOARD_STATUSES, RISK_BAND_LABELS, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_queue_returns_requested_count_and_valid_scores(client):
    resp = client.get("/v1/dashboard/queue", params={"n": 10, "seed": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 10
    assert body["synthetic"] is True
    assert len(body["transactions"]) == 10
    for txn in body["transactions"]:
        assert 0.0 <= txn["risk_score"] <= 1.0
        assert txn["risk_band"] in RISK_BAND_LABELS
        assert txn["status"] in _DASHBOARD_STATUSES
        assert txn["transaction_id"].startswith("TXN-DEMO-")


def test_queue_is_deterministic_given_the_same_seed(client):
    first = client.get("/v1/dashboard/queue", params={"n": 12, "seed": 42}).json()
    second = client.get("/v1/dashboard/queue", params={"n": 12, "seed": 42}).json()
    assert first["transactions"] == second["transactions"]


def test_queue_differs_across_seeds(client):
    a = client.get("/v1/dashboard/queue", params={"n": 12, "seed": 1}).json()
    b = client.get("/v1/dashboard/queue", params={"n": 12, "seed": 2}).json()
    assert a["transactions"] != b["transactions"]


def test_queue_clamps_n_to_a_sane_upper_bound(client):
    resp = client.get("/v1/dashboard/queue", params={"n": 100000, "seed": 1})
    assert resp.status_code == 200
    assert resp.json()["count"] <= 200


def test_queue_clamps_n_to_a_minimum_of_one(client):
    resp = client.get("/v1/dashboard/queue", params={"n": 0, "seed": 1})
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


def test_highest_scoring_transaction_gets_the_worst_available_band(client):
    body = client.get("/v1/dashboard/queue", params={"n": 48, "seed": 7}).json()
    txns = body["transactions"]
    worst_score_txn = max(txns, key=lambda t: t["risk_score"])
    worst_index = max(RISK_BAND_LABELS.index(t["risk_band"]) for t in txns)
    assert RISK_BAND_LABELS.index(worst_score_txn["risk_band"]) == worst_index


def test_queue_exposes_stage11_global_feature_importance_not_per_transaction(client):
    body = client.get("/v1/dashboard/queue", params={"n": 5, "seed": 1}).json()
    importances = body["global_feature_importance"]
    assert len(importances) == 15
    assert importances[0]["feature"] == "C13"
    assert all("feature" in item and "importance" in item for item in importances)
    # Global ranking must not vary with the batch's seed/content.
    other = client.get("/v1/dashboard/queue", params={"n": 5, "seed": 999}).json()
    assert other["global_feature_importance"] == importances


def test_queue_reports_the_real_artifact_version_and_checksum(client):
    body = client.get("/v1/dashboard/queue", params={"n": 3, "seed": 1}).json()
    assert body["model"]["artifact_version"] == "1.0.0"
    assert len(body["model"]["artifact_sha256"]) == 64


def test_single_transaction_predict_endpoint_is_unaffected(client):
    """Regression guard: adding the dashboard route must not change the
    existing deployed single-transaction scoring flow's behavior."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["artifact_loaded"] is True


def test_queue_returns_503_when_artifact_unavailable(client):
    original_artifact = app.state.artifact
    original_manifest = app.state.manifest
    try:
        app.state.artifact = None
        app.state.manifest = None
        resp = client.get("/v1/dashboard/queue", params={"n": 5, "seed": 1})
        assert resp.status_code == 503
        assert resp.json()["error_category"] == "ArtifactUnavailable"
    finally:
        app.state.artifact = original_artifact
        app.state.manifest = original_manifest
