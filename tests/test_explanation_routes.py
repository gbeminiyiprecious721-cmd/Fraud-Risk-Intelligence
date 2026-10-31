"""Read-only and synthetic-only explanation resource boundaries."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.api import create_app
from src import explanation_routes

@pytest.fixture
def client():
    with TestClient(create_app()) as c:yield c


def test_global_aggregate_methods_and_no_historical_records(client):
    r=client.get('/v1/dashboard/explainability');assert r.status_code==200
    a=r.json();assert set(a['methods'])=={'mdi','shap'} and a['capabilities']['local_shap']
    assert a['population']['n']==512 and a['population']['labels_used'] is False
    assert 'groups' not in a and 'records' not in a and 'TransactionID' not in r.text
    assert len(a['methods']['mdi'])==len(a['methods']['shap'])==20


@pytest.mark.parametrize('identifier',['TXN-DEMO-9999-0000','TXN-DEMO-0101-0048','../model','3155871','CASE-123','TXN-DEMO-0101--001'])
def test_unsupported_identifiers_rejected(client,identifier):
    r=client.get('/v1/dashboard/explanation',params={'transaction_id':identifier})
    assert r.status_code in (404,422)


def test_local_reconciles_and_writes_are_unsupported(client):
    r=client.get('/v1/dashboard/explanation',params={'transaction_id':'TXN-DEMO-0101-0047'});assert r.status_code==200
    a=r.json();assert abs(a['baseline']+sum(c['value'] for c in a['contributions'])+a['other']-a['score'])<=1e-8
    assert len(a['contributions'])==10 and len(r.content)<16384
    for path in ['/v1/dashboard/explanation','/v1/dashboard/explainability']:
        assert client.post(path,json={'notes':'must not be stored'}).status_code==405
    for path in ['/explainability_assets/local-v1.json','/explainability_assets/global-v1.json','/local-v1.json']:
        assert client.get(path).status_code==404


@pytest.mark.parametrize('kind',['absent','malformed','wrong_model','extra_field','bad_score','oversized_public'])
def test_corrupt_assets_fail_closed(client,tmp_path,monkeypatch,kind):
    global_path=tmp_path/'global.json';local_path=tmp_path/'local.json'
    if kind!='absent':
        a=json.loads(explanation_routes.GLOBAL.read_text());b=json.loads(explanation_routes.LOCAL.read_text())
        if kind=='wrong_model':a['model_sha256']='0'*64
        if kind=='extra_field':a['historical_records']=[{'notes':'private'}]
        if kind=='bad_score':b['records'][0]['score']=2
        if kind=='oversized_public':a['unavailable_reason']='x'*20000
        global_path.write_text('{' if kind=='malformed' else json.dumps(a));local_path.write_text(json.dumps(b))
    monkeypatch.setattr(explanation_routes,'GLOBAL',global_path);monkeypatch.setattr(explanation_routes,'LOCAL',local_path)
    assert client.get('/v1/dashboard/explainability').status_code==503


def test_public_lookup_scope_preserves_public_prediction_guard():
    from src.render_demo import app
    with TestClient(app) as c:
        assert c.get('/v1/dashboard/explanation',params={'transaction_id':'TXN-DEMO-0101-0023'}).status_code==200
        assert c.get('/v1/dashboard/explanation',params={'transaction_id':'TXN-DEMO-0101-0024'}).status_code==404
        assert c.post('/v1/predict',json={'C1':999}).status_code==403


def test_validated_mdi_only_fallback_is_explicit(client,tmp_path,monkeypatch):
    a=json.loads(explanation_routes.GLOBAL.read_text());b=json.loads(explanation_routes.LOCAL.read_text())
    a['methods'].pop('shap');a['capabilities'].update(global_shap=False,local_shap=False);a['baseline']=None
    a['stability']={'status':'insufficient','reason':'Deliberate test-only unavailable SHAP'};b['records']=[]
    g=tmp_path/'global.json';l=tmp_path/'local.json';g.write_text(json.dumps(a));l.write_text(json.dumps(b))
    monkeypatch.setattr(explanation_routes,'GLOBAL',g);monkeypatch.setattr(explanation_routes,'LOCAL',l)
    r=client.get('/v1/dashboard/explainability');assert r.status_code==200 and set(r.json()['methods'])=={'mdi'}
    assert client.get('/v1/dashboard/explanation',params={'transaction_id':'TXN-DEMO-0101-0000'}).status_code==503


def test_loaded_artifact_identity_and_scope_cannot_be_overridden(client):
    for params in [{'transaction_id':'TXN-DEMO-0101-0048','limit':200},{'transaction_id':'TXN-DEMO-2024-0000','seed':101}]:
        assert client.get('/v1/dashboard/explanation',params=params).status_code==404
    client.app.state.manifest={**client.app.state.manifest,'artifact_sha256':'0'*64}
    assert client.get('/v1/dashboard/explainability').status_code==503
