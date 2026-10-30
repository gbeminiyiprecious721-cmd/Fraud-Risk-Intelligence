"""Independent Stage 20 release verification; inference only, never training.

Re-read authorised validation labels only; recompute metrics/checkpoints and
all 1,000 bootstrap draws using sklearn on literal resampled rows, rather
than the benchmark's optimised weighted AP implementation. Public evidence
contains aggregates/hashes only. Requires private Stage 20 caches locally.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRIVATE = ROOT / 'artifacts/stage20'
EVIDENCE = ROOT / 'docs/stage20_challenger_evidence'
EXPECTED_IDS = '14e5f38141f4ea86cf6228d76bf9af398c7cc3116d1ca49ddecc3b502e9ff5bc'
EXPECTED_FROZEN = '2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from scripts.run_stage20_benchmark import load_native, predict

    assert digest(ROOT / 'artifacts/frozen_rf_fraud_risk_v1.joblib') == EXPECTED_FROZEN
    aggregate = json.loads((ROOT / 'ui/benchmark-v1.json').read_text())
    ids = np.load(PRIVATE / 'validation-ids.npy')
    times = np.load(PRIVATE / 'validation-times.npy')
    y = np.load(PRIVATE / 'validation-y.npy')
    fingerprint = hashlib.sha256('\x1f'.join(map(str, sorted(map(int, ids)))).encode()).hexdigest()
    assert fingerprint == EXPECTED_IDS == aggregate['validation_id_sha256']
    assert len(ids) == len(np.unique(ids)) == len(y) == len(times) == 75560
    assert int(y.sum()) == 2617 and set(np.unique(y)) == {0, 1}
    assignment = pd.read_csv(ROOT / 'docs/_stage4_split_assignment.csv',
                             usecols=['TransactionID', 'TransactionDT', 'partition'])
    authorised = assignment.loc[assignment.partition.eq('validation')].sort_values(['TransactionDT', 'TransactionID'])
    assert np.array_equal(ids, authorised.TransactionID.to_numpy())
    assert np.array_equal(times, authorised.TransactionDT.to_numpy())
    assert assignment.loc[assignment.partition.eq('train'), 'TransactionDT'].max() < times.min()
    # Locate physical rows by ID alone; skip other rows BEFORE label parsing.
    source = ROOT / 'data/raw/ieee-fraud-detection/train_transaction.csv'
    source_ids = pd.read_csv(source, usecols=['TransactionID']).TransactionID
    keep = source_ids.isin(set(ids)).to_numpy()
    labels = pd.read_csv(source, usecols=['TransactionID', 'TransactionDT', 'isFraud'],
                         skiprows=lambda row: row > 0 and not keep[row-1])
    labels = labels.sort_values(['TransactionDT', 'TransactionID'])
    assert np.array_equal(ids, labels.TransactionID.to_numpy())
    assert np.array_equal(y, labels.isFraud.to_numpy())
    assert np.array_equal(times, labels.TransactionDT.to_numpy())
    matrix = np.load(PRIVATE / 'validation.npy', mmap_mode='r')
    assert matrix.shape == (75560, 813)
    scores = []
    summaries = []
    count = 0
    for key, model in zip(['rf', 'lgb', 'xgb', 'cat'], aggregate['models'], strict=True):
        raw = np.load(PRIVATE / f'{key}-scores.npy')
        assert raw.shape == y.shape and np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
        assert model['scoring']['rows'] == len(y)
        assert model['scoring']['validation_id_sha256'] == fingerprint
        path = ROOT / 'artifacts/frozen_rf_fraud_risk_v1.joblib' if key == 'rf' else PRIVATE / {'lgb':'lightgbm.txt', 'xgb':'xgboost.json', 'cat':'catboost.cbm'}[key]
        assert digest(path) == model['scoring']['native_artifact_sha256']
        native = load_native(key)
        reloaded = np.concatenate([predict(native, key, matrix[i:i+4096]) for i in range(0, len(y), 4096)])
        difference = float(np.max(np.abs(raw-reloaded)))
        assert np.allclose(raw, reloaded, rtol=0, atol=1e-12)
        s = np.round(raw, 12)
        metrics = model['metrics']
        ap = float(average_precision_score(y, s))
        roc = float(roc_auc_score(y, s))
        p, r, _ = precision_recall_curve(y, s)
        assert math.isclose(ap, metrics['average_precision'], rel_tol=0, abs_tol=1e-12)
        assert math.isclose(roc, metrics['roc_auc'], rel_tol=0, abs_tol=1e-12)
        assert math.isclose(float(auc(r, p)), metrics['trapezoidal_pr_auc'], rel_tol=0, abs_tol=1e-12)
        order = np.lexsort((ids, -s))
        points = metrics['points']
        assert len(points) == 201
        assert [p['k'] for p in points] == [math.ceil(len(y)*i/200) for i in range(201)]
        for point in points:
            selected = np.zeros(len(y), dtype=bool)
            selected[order[:point['k']]] = True
            tp = int(np.count_nonzero(selected & (y == 1)))
            fp = int(np.count_nonzero(selected & (y == 0)))
            fn = int(np.count_nonzero(~selected & (y == 1)))
            tn = int(np.count_nonzero(~selected & (y == 0)))
            assert [point[k] for k in ['tp', 'fp', 'fn', 'tn']] == [tp, fp, fn, tn]
            assert point['precision'] == (tp/(tp+fp) if tp+fp else None)
            assert point['recall'] == tp/(tp+fn)
            count += 1
        for fraction, name in [(.01, 'top1'), (.05, 'top5')]:
            assert metrics[name] == next(p for p in points if p['k'] == math.ceil(len(y)*fraction))
        scores.append(s)
        summaries.append({'name':model['name'], 'average_precision':ap, 'roc_auc':roc,
                          'checkpoints_verified':201, 'native_reload_max_difference':difference})
    # Reconstruct timestamp-tie-safe blocks independently, then sample literal
    # row indices. No call to paired_block_bootstrap or weighted_ap_prepared.
    cuts = [0]
    for i in range(1, 20):
        cut = len(y)*i//20
        while cut < len(y) and times[cut] == times[cut-1]:
            cut += 1
        if cuts[-1] < cut < len(y):
            cuts.append(cut)
    cuts.append(len(y))
    assert len(cuts) == 21
    blocks = [np.arange(a, b) for a, b in zip(cuts, cuts[1:])]
    assert all(times[a[-1]] < times[b[0]] for a, b in zip(blocks, blocks[1:]))
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(1000):
        sampled = np.concatenate([blocks[i] for i in rng.integers(0, 20, 20)])
        sampled_y = y[sampled]
        if len(np.unique(sampled_y)) != 2:
            continue
        draws.append([average_precision_score(sampled_y, s[sampled]) for s in scores])
    draws = np.asarray(draws)
    assert len(draws) == aggregate['bootstrap']['valid_draws'] == 1000
    assert aggregate['bootstrap']['degenerate_draws'] == 0
    assert aggregate['bootstrap']['block_count'] == 20 and aggregate['bootstrap']['seed'] == 42
    for i, model in enumerate(aggregate['models']):
        ap_ci = np.quantile(draws[:, i], [.025, .975])
        delta_ci = np.quantile(draws[:, i]-draws[:, 0], [.025, .975])
        assert np.allclose(ap_ci, model['uncertainty']['ap_ci'], rtol=0, atol=1e-12)
        assert np.allclose(delta_ci, model['uncertainty']['ap_difference_ci'], rtol=0, atol=1e-12)
        summaries[i].update(ap_ci=ap_ci.tolist(), ap_difference_ci=delta_ci.tolist())
    assert digest(ROOT / 'artifacts/frozen_rf_fraud_risk_v1.joblib') == EXPECTED_FROZEN
    result = {'reviewed_commit':'cf4fea1141c7f721f59300a8afb17dae48a73627',
              'authorised_rows':len(y), 'positives':int(y.sum()), 'validation_id_sha256':fingerprint,
              'source_validation_labels_independently_matched':True, 'checkpoints_verified':count,
              'bootstrap_method':'sklearn Average Precision on literal paired resampled block rows; independent of weighted implementation',
              'valid_draws':len(draws), 'degenerate_draws':0, 'models':summaries,
              'no_training_performed':True, 'closed_evaluation_values_accessed':False,
              'frozen_sha256':EXPECTED_FROZEN}
    (EVIDENCE / 'release-independent-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
