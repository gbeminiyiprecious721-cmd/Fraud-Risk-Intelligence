import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.capacity_simulator import build_capacity_points


def test_counts_and_random_expectation_against_independent_sklearn():
    y = np.array([0, 1, 1, 0, 1])
    scores = np.array([.2, .9, .4, .8, .1])
    ids = np.arange(5)
    points = build_capacity_points(y, scores, ids, range(6))
    for k, p in enumerate(points):
        predicted = np.zeros(5, dtype=int)
        predicted[sorted(range(5), key=lambda i: (-scores[i], ids[i]))[:k]] = 1
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        assert (p['tp'], p['fp'], p['fn'], p['tn']) == (tp, fp, fn, tn)
        assert p['recall'] == pytest.approx(recall_score(y, predicted))
        assert p['precision'] == (pytest.approx(precision_score(y, predicted)) if k else None)
        assert p['random_tp_expected'] == pytest.approx(k * 3 / 5)
        assert p['random_recall_expected'] == pytest.approx(k / 5)
    assert points[-1]['tp'] == 3 and points[-1]['fn'] == 0
    assert [p['tp'] for p in points] == sorted(p['tp'] for p in points)


def test_ties_are_resolved_by_id_and_input_order_does_not_matter():
    a = build_capacity_points([1, 0], [.5, .5], [20, 10], [0, 1, 2])
    b = build_capacity_points([0, 1], [.5, .5], [10, 20], [0, 1, 2])
    assert a == b and a[1]['tp'] == 0


def test_zero_positive_population_preserves_counts_and_undefined_recall():
    p = build_capacity_points([0, 0], [.9, .1], [1, 2], [0, 1, 2])
    assert all(x['recall'] is None and x['random_recall_expected'] is None for x in p)
    assert p[0]['precision'] is None
    assert p[1]['precision'] == 0 and p[1]['fp'] == 1


@pytest.mark.parametrize('y,scores,ids,ks', [
    ([], [], [], [0]), ([2], [.5], [1], [1]), ([1], [float('nan')], [1], [1]),
    ([1], [.5], [1], [-1]), ([1], [.5], [1], [2]), ([1], [.5], [1], [.5]),
    ([1], [.5], [1], [True]), ([1, 0], [.5, .4], [1, 1], [1]),
    ([1], [.5, .6], [1], [1]), ([1], [1.1], [1], [1]),
])
def test_invalid_inputs_are_refused(y, scores, ids, ks):
    with pytest.raises(ValueError):
        build_capacity_points(y, scores, ids, ks)


def test_aggregate_endpoint_is_fixed_read_only_and_contains_no_row_records():
    from fastapi.testclient import TestClient
    from src.api import create_app
    from src.render_demo import app as public_app
    for app in [create_app(), public_app]:
        response = TestClient(app).get('/v1/dashboard/capacity')
        assert response.status_code == 200
        body = response.json()
        assert body['schema_version'] == '1.0.0' and body['population_id'] == 'stage4-validation'
        assert len(body['points']) == 201
        assert not {'scores', 'labels', 'transactions', 'TransactionID'} & set(body)
        assert TestClient(app).post('/v1/dashboard/capacity', json={}).status_code == 405
        if app is public_app:
            assert 'window.CapacitySimulator' in TestClient(app).get('/capacity.js').text


def test_validation_reader_skips_non_validation_features_and_labels(tmp_path):
    import pandas as pd
    from scripts.build_capacity_aggregate import read_validation_only
    source = tmp_path / 'source.csv'
    source.write_text('TransactionID,isFraud,feature\n1,DO_NOT_PARSE,FORBIDDEN\n2,1,1.5\n3,DO_NOT_PARSE,FORBIDDEN\n')
    assignment = pd.DataFrame({'TransactionID': [1, 2, 3], 'partition': ['train', 'validation', 'final_test']})
    result = read_validation_only(source, assignment, ['feature'])
    assert result.to_dict('records') == [{'TransactionID': 2, 'isFraud': 1, 'feature': 1.5}]


def test_all_published_checkpoints_are_consistent_and_monotonic():
    body = json.loads((Path(__file__).resolve().parent.parent / 'ui/capacity-v1.json').read_text())
    n, positives = body['n'], body['positives']
    last_tp = -1
    for p in body['points']:
        assert p['tp'] >= last_tp
        assert p['tp'] + p['fn'] == positives
        assert p['tn'] + p['fp'] == n - positives
        assert p['tp'] + p['fp'] == p['k']
        assert p['precision'] == (p['tp'] / p['k'] if p['k'] else None)
        assert p['recall'] == p['tp'] / positives
        assert p['capacity_fraction'] == p['k'] / n
        assert p['random_tp_expected'] == p['k'] * positives / n
        last_tp = p['tp']
    assert body['points'][0]['tp'] == 0 and body['points'][-1]['recall'] == 1
    assert body['independent_checks'] == len(body['points'])


def test_builder_has_no_fit_or_final_evaluation_artifact_access():
    import ast
    path = Path(__file__).resolve().parent.parent / 'scripts/build_capacity_aggregate.py'
    tree = ast.parse(path.read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in {'fit', 'fit_transform', 'partial_fit'}]
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not [s for s in literals if '_stage10_final_test' in s or 'test_transaction.csv' == s]


def test_missing_aggregate_fails_with_clear_503(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from src.api import create_app
    import src.capacity_routes as routes
    monkeypatch.setattr(routes, '_AGGREGATE', tmp_path / 'absent.json')
    response = TestClient(create_app()).get('/v1/dashboard/capacity')
    assert response.status_code == 503
    assert str(tmp_path) not in response.text
