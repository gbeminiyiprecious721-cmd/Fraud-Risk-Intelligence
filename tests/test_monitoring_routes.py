"""Fixed aggregate-only route: corrupt evidence fails closed, writes unsupported."""
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from src import monitoring_routes

@pytest.fixture
def client():
    app=FastAPI();app.include_router(monitoring_routes.router)
    return TestClient(app)

def test_read_only_aggregate(client):
    r=client.get('/v1/dashboard/monitoring');assert r.status_code==200
    a=r.json();assert a['source']=='historical_validation'
    assert len(a['windows'])==3
    assert 'records' not in a and 'isFraud' not in r.text
    for method in ['post','put','patch','delete']:
        assert getattr(client,method)('/v1/dashboard/monitoring').status_code==405

@pytest.mark.parametrize('content',[None,'{','{}','{"schema_version":1,"records":[1]}'])
def test_absent_or_corrupt_aggregate(client,tmp_path,monkeypatch,content):
    p=tmp_path/'monitoring.json'
    if content is not None:p.write_text(content)
    monkeypatch.setattr(monitoring_routes,'AGGREGATE',p)
    r=client.get('/v1/dashboard/monitoring');assert r.status_code==503
    assert r.json()=={'detail':'Historical monitoring evidence unavailable.'}
