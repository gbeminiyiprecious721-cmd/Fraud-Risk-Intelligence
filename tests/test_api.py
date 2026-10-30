import ast
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import ALLOWED_UI_ORIGINS, DEFAULT_ARTIFACT_PATH, EXPECTED_ARTIFACT_ID, API_CONTRACT_VERSION, create_app
from src.production_pipeline import load_artifact

MODULE_SOURCE_PATH = Path(__file__).resolve().parent.parent / "src" / "api.py"


def _real_schema():
    artifact, manifest = load_artifact(DEFAULT_ARTIFACT_PATH)
    return artifact, manifest


def _valid_payload() -> dict:
    artifact, _ = _real_schema()
    schema = artifact.schema
    payload = {}
    for col in schema.raw_columns:
        if col in schema.numeric_columns:
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


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Lifespan / artifact loading
# ---------------------------------------------------------------------------

def test_testclient_context_manager_triggers_lifespan_and_loads_artifact(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["artifact_loaded"] is True


def test_artifact_loading_failure_is_reported_via_health_and_predict_not_a_crash():
    app = create_app(artifact_path="/nonexistent/path/does_not_exist.joblib")
    with TestClient(app) as c:
        health = c.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "artifact_unavailable", "artifact_loaded": False}

        predict_resp = c.post("/v1/predict", json={"anything": 1})
        assert predict_resp.status_code == 503
        assert predict_resp.json()["error_category"] == "ArtifactUnavailable"


# ---------------------------------------------------------------------------
# Valid request / response schema / version metadata
# ---------------------------------------------------------------------------

def test_valid_predict_request_returns_a_probability_and_version_metadata(client):
    resp = client.post("/v1/predict", json=_valid_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["api_contract_version"] == API_CONTRACT_VERSION
    assert body["artifact_version"] == "1.0.0"
    assert body["manifest_schema_version"] == "1.1.0"
    assert isinstance(body["artifact_sha256"], str) and len(body["artifact_sha256"]) == 64
    assert "request_id" in body


def test_health_endpoint_schema(client):
    resp = client.get("/health")
    assert set(resp.json().keys()) == {"status", "artifact_loaded"}


def test_version_endpoint_matches_the_real_manifest(client):
    _, manifest = _real_schema()
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["api_contract_version"] == API_CONTRACT_VERSION
    assert body["artifact_id"] == EXPECTED_ARTIFACT_ID == manifest["artifact_id"]
    assert body["artifact_version"] == manifest["artifact_version"]
    assert body["manifest_schema_version"] == manifest["manifest_schema_version"]
    assert body["artifact_sha256"] == manifest["artifact_sha256"]


# ---------------------------------------------------------------------------
# Malformed / empty / duplicate-key JSON (API-layer concerns, before Stage 13 ever sees a DataFrame)
# ---------------------------------------------------------------------------

def test_malformed_json_returns_422(client):
    resp = client.post("/v1/predict", content=b"{not valid json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert resp.json()["error_category"] == "MalformedJSON"


def test_empty_request_body_returns_422(client):
    resp = client.post("/v1/predict", content=b"", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


def test_empty_json_object_returns_422(client):
    resp = client.post("/v1/predict", content=b"{}", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert resp.json()["error_category"] == "EmptyOrInvalidRequest"


def test_duplicate_top_level_field_name_returns_422(client):
    payload = _valid_payload()
    body = json.dumps(payload)[:-1]  # drop closing brace
    duplicated = body + f', "TransactionAmt": 999.0}}'
    resp = client.post("/v1/predict", content=duplicated.encode(), headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert resp.json()["error_category"] == "DuplicateField"


# ---------------------------------------------------------------------------
# Missing / incompatible fields -- delegated to Stage 13's own SchemaValidationError
# ---------------------------------------------------------------------------

def test_missing_required_field_returns_422(client):
    payload = _valid_payload()
    del payload["TransactionAmt"]
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error_category"] == "SchemaValidationError"
    assert "Missing required" in resp.json()["detail"]


def test_non_numeric_value_in_numeric_field_returns_422(client):
    payload = _valid_payload()
    payload["TransactionAmt"] = "not_a_number"
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error_category"] == "SchemaValidationError"
    assert "non-numeric" in resp.json()["detail"]


def test_extra_unexpected_field_is_ignored_not_rejected(client):
    payload = _valid_payload()
    payload["some_future_field_the_api_does_not_know_about"] = "ignored"
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Numeric extremes / unseen categorical values / permitted missingness
# (the exact edge cases Stage 12/13 already proved the underlying pipeline handles)
# ---------------------------------------------------------------------------

def test_unsupported_numeric_extreme_does_not_crash(client):
    payload = _valid_payload()
    payload["TransactionAmt"] = 1_000_000.0
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["fraud_probability"] <= 1.0


def test_unseen_categorical_value_does_not_crash(client):
    payload = _valid_payload()
    payload["ProductCD"] = "ZZ_never_seen_category"
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["fraud_probability"] <= 1.0


def test_permitted_missing_numeric_value_is_accepted(client):
    artifact, _ = _real_schema()
    payload = _valid_payload()
    # JSON null -> Python None -> pandas NaN -> imputed by the approved pipeline, same as Stage 12/13.
    some_numeric_col = next(c for c in artifact.schema.numeric_columns if c != "TransactionAmt")
    payload[some_numeric_col] = None
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["fraud_probability"] <= 1.0


# ---------------------------------------------------------------------------
# Negative TransactionAmt -- investigated using EXISTING evidence only, not
# a newly-invented API-layer prohibition. Stage 2's own validation found
# the real data's TransactionAmt is always positive (min observed: 0.251),
# and neither Stage 12's nor Stage 13's edge-case checks ever added a
# sign check to the approved pipeline. This API therefore does NOT reject
# a negative amount either -- it is passed through exactly as Stage 13
# already handles it (a real, if physically-implausible, prediction, no
# error) -- and the inherited gap is documented, not silently accepted
# without comment (see docs/STAGE14_API_REPORT.md).
# ---------------------------------------------------------------------------

def test_negative_transaction_amount_is_not_rejected_matching_stage13_behavior(client):
    payload = _valid_payload()
    payload["TransactionAmt"] = -50.0
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200, "Stage 13's pipeline does not reject this value; the API must not invent a new rejection either."
    assert 0.0 <= resp.json()["fraud_probability"] <= 1.0


# ---------------------------------------------------------------------------
# No refitting during requests / repeated-request consistency / identity consistency
# ---------------------------------------------------------------------------

def test_no_refitting_occurs_across_requests(client):
    artifact, _ = _real_schema()
    imputer_stats_before = artifact.preprocessor.named_transformers_["numeric_plain"].named_steps["impute"].statistics_.copy()
    n_estimators_before = len(artifact.model.estimators_)
    client.post("/v1/predict", json=_valid_payload())
    client.post("/v1/predict", json=_valid_payload())
    imputer_stats_after = artifact.preprocessor.named_transformers_["numeric_plain"].named_steps["impute"].statistics_
    assert (imputer_stats_before == imputer_stats_after).all()
    assert len(artifact.model.estimators_) == n_estimators_before


def test_repeated_identical_requests_return_identical_probabilities(client):
    payload = _valid_payload()
    resp1 = client.post("/v1/predict", json=payload)
    resp2 = client.post("/v1/predict", json=payload)
    assert resp1.status_code == resp2.status_code == 200
    # Parallel RF summation varies at ~1e-15 on macOS (already covered by
    # test_production_pipeline's tolerance-based repeated-inference check).
    # Keep production scoring unchanged; reject differences above 1e-12.
    assert resp1.json()["fraud_probability"] == pytest.approx(
        resp2.json()["fraud_probability"], rel=0, abs=1e-12
    )


def test_artifact_identity_is_consistent_across_health_version_and_predict(client):
    version_resp = client.get("/version").json()
    predict_resp = client.post("/v1/predict", json=_valid_payload()).json()
    assert version_resp["artifact_sha256"] == predict_resp["artifact_sha256"]
    assert version_resp["artifact_version"] == predict_resp["artifact_version"]


# ---------------------------------------------------------------------------
# Privacy-conscious logging
# ---------------------------------------------------------------------------

def test_logs_never_contain_raw_field_values(client, caplog):
    payload = _valid_payload()
    marker_value = "THIS_EXACT_STRING_MUST_NEVER_APPEAR_IN_LOGS"
    payload["ProductCD"] = marker_value  # deliberately an unseen category -- still a 200, still logged
    with caplog.at_level(logging.INFO, logger="fraud_risk_api"):
        client.post("/v1/predict", json=payload)
    assert marker_value not in caplog.text


def test_logs_do_not_contain_schema_validation_detail_text(client, caplog):
    payload = _valid_payload()
    payload["TransactionAmt"] = "not_a_number_marker_xyz"
    with caplog.at_level(logging.INFO, logger="fraud_risk_api"):
        client.post("/v1/predict", json=payload)
    assert "not_a_number_marker_xyz" not in caplog.text
    assert "SchemaValidationError" in caplog.text  # the CATEGORY is fine to log; the detail text is not


# ---------------------------------------------------------------------------
# final_test isolation (same discipline as the rest of this project's new modules)
# ---------------------------------------------------------------------------

def test_module_imports_nothing_that_reads_datasets():
    tree = ast.parse(MODULE_SOURCE_PATH.read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    forbidden = {"src.leakage_boundaries", "src.model_selection", "pandas.read_csv"}
    assert not (imported_names & forbidden)


def test_module_never_calls_with_allow_final_test_true():
    tree = ast.parse(MODULE_SOURCE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                assert not (kw.arg == "allow_final_test" and isinstance(kw.value, ast.Constant) and kw.value.value is True)


def test_module_has_no_artifact_upload_endpoint():
    source = MODULE_SOURCE_PATH.read_text()
    assert "UploadFile" not in source
    assert "multipart" not in source.lower()


# ---------------------------------------------------------------------------
# Stage 15 -- narrowly scoped CORS for the local static UI only
# ---------------------------------------------------------------------------

def test_cors_allowed_origin_receives_the_allow_origin_header(client):
    allowed = ALLOWED_UI_ORIGINS[0]
    resp = client.post("/v1/predict", json=_valid_payload(), headers={"Origin": allowed})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == allowed


def test_cors_disallowed_origin_does_not_receive_the_allow_origin_header(client):
    resp = client.post(
        "/v1/predict", json=_valid_payload(),
        headers={"Origin": "http://evil.example.com"},
    )
    # The request itself still succeeds server-side (CORS is enforced by the
    # BROWSER refusing to expose the response to the calling page, not by the
    # server refusing to answer) -- what matters is the header's absence.
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers.keys()}


def test_cors_real_preflight_for_allowed_origin(client):
    allowed = ALLOWED_UI_ORIGINS[0]
    resp = client.options(
        "/v1/predict",
        headers={
            "Origin": allowed,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == allowed
    assert "POST" in resp.headers.get("access-control-allow-methods", "")


def test_cors_real_preflight_for_disallowed_origin_is_not_approved(client):
    resp = client.options(
        "/v1/predict",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    # Starlette's CORSMiddleware answers preflight with 400 for a disallowed
    # origin rather than silently approving it -- verified directly, not assumed.
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers.keys()}


def test_cors_does_not_allow_credentials():
    # allow_credentials=False is a deliberate choice (no cookies/auth exist to
    # protect) -- re-read directly from the app's own middleware stack rather
    # than inferring it from a single response header.
    app = create_app()
    cors_middlewares = [m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
    assert len(cors_middlewares) == 1
    assert cors_middlewares[0].kwargs.get("allow_credentials") is False


def test_cors_allowed_origins_are_127_0_0_1_and_localhost_only():
    assert ALLOWED_UI_ORIGINS == ["http://127.0.0.1:8080", "http://localhost:8080"]
    assert all("*" not in origin for origin in ALLOWED_UI_ORIGINS)
