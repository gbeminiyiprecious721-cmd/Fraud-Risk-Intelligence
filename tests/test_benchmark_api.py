"""Public surface must remain fixed, aggregate-only and non-executable."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.api import create_app

ROOT=Path(__file__).resolve().parents[1]


def test_fixed_aggregate_and_closed_write_surface():
    client=TestClient(create_app())
    response=client.get('/v1/dashboard/benchmark')
    assert response.status_code==200
    assert response.headers['cache-control']=='no-store'
    body=response.json()
    assert body==json.loads((ROOT/'ui/benchmark-v1.json').read_text())
    assert body['rows']==75560 and body['positives']==2617
    assert len(body['models'])==4
    for method in ['post','put','delete','patch']:
        assert getattr(client,method)('/v1/dashboard/benchmark').status_code==405
    assert client.get('/v1/dashboard/benchmark/../../artifacts/stage20').status_code==404


def test_aggregate_has_no_individual_records_or_native_artifacts():
    body=json.loads((ROOT/'ui/benchmark-v1.json').read_text())
    forbidden={'TransactionID','isFraud','labels','scores','predictions','records','raw_data','artifact_path','model_path'}
    def walk(value):
        if isinstance(value,dict):
            assert not forbidden.intersection(value)
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(body)
    assert body['frozen_model_sha256']=='2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04'
    assert body['models'][0]['training_seconds'] is None
    for model in body['models']:
        assert model['scoring']['repeat_consistent']
        points=model['metrics']['points']
        assert len(points)==201 and points[0]['k']==0 and points[-1]['k']==75560
        assert all(a['tp']<=b['tp'] for a,b in zip(points,points[1:]))
        for point in points:
            assert point['tp']+point['fn']==2617
            assert point['tp']+point['fp']==point['k']
            assert point['tn']+point['fp']==75560-2617
        assert model['metrics']['independent_checkpoint_checks']==201
        if model['training']:
            assert model['training']['selection']['stopping_population']=='inner_stop only'
            assert model['training']['final']['native_reload_consistent']
            assert model['training']['selection']['selected_rounds']==model['training']['final']['selected_rounds']
    assert body['bootstrap']['valid_draws']>=900


def test_absent_evidence_is_explicit_unavailable(monkeypatch,tmp_path):
    import src.benchmark_routes as routes
    monkeypatch.setattr(routes,'_AGGREGATE',tmp_path/'absent.json')
    assert TestClient(create_app()).get('/v1/dashboard/benchmark').status_code==503


def test_reference_matches_preexisting_capacity_evidence():
    benchmark=json.loads((ROOT/'ui/benchmark-v1.json').read_text())['models'][0]['metrics']
    capacity=json.loads((ROOT/'ui/capacity-v1.json').read_text())
    assert benchmark['average_precision']==pytest.approx(capacity['average_precision'],rel=0,abs=1e-12)
    assert len(benchmark['points'])==len(capacity['points'])==201
    for old,new in zip(capacity['points'],benchmark['points']):
        for key in ['k','tp','fp','fn','tn','precision','recall']:assert old[key]==new[key]


def test_public_demo_serves_only_aggregate_and_preserves_scoring_guard():
    import src.render_demo as public
    public._rate_limit_by_client.clear();public._rate_limit_global.clear()
    try:
        with TestClient(public.app) as client:
            assert client.get('/v1/dashboard/benchmark').status_code==200
            assert client.get('/benchmark.js').status_code==200
            assert client.get('/artifacts/stage20/xgboost.json').status_code==404
            assert client.post('/v1/dashboard/benchmark').status_code==405
            payload=dict(public.DEMO_TRANSACTION);payload['C1']+=1
            assert client.post('/v1/predict',json=payload).status_code==403
    finally:
        public._rate_limit_by_client.clear();public._rate_limit_global.clear()
