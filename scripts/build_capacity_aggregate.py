"""Stage 19: inference-only validation aggregate; never opens final-test artifacts.

The initial source scan parses ONLY TransactionID. Feature/label parsing skips
all non-validation physical rows, unlike a read-all-then-filter label loader.
No row-level scores or labels are saved. Run from the repository root.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.capacity_simulator import build_capacity_points
from src.leakage_boundaries import get_partition
from src.production_pipeline import load_artifact, predict_fraud_probability

EXPECTED = '2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04'


def fingerprint(ids) -> str:
    return hashlib.sha256('\x1f'.join(str(i) for i in sorted(ids)).encode()).hexdigest()


def read_validation_only(source: Path, assignment: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    validation = get_partition(assignment, ['validation'])
    allowed = set(validation.TransactionID)
    index = pd.read_csv(source, usecols=['TransactionID']).TransactionID
    if index.duplicated().any() or not allowed.issubset(set(index)):
        raise ValueError('Source IDs must uniquely cover the validation partition')
    physical_rows = set(np.flatnonzero(index.isin(allowed)).tolist())
    # CSV has one header line and one physical line per IEEE-CIS transaction.
    frame = pd.read_csv(source, usecols=['TransactionID', 'isFraud'] + columns,
                        skiprows=lambda row: row > 0 and row - 1 not in physical_rows)
    if set(frame.TransactionID) != allowed or len(frame) != len(allowed):
        raise ValueError('Validation row alignment failed')
    return frame


def main() -> None:
    artifact_path = ROOT / 'artifacts/frozen_rf_fraud_risk_v1.joblib'
    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Frozen artifact differs from approved checksum')
    artifact, manifest = load_artifact(artifact_path)
    assignment_path = ROOT / 'docs/_stage4_split_assignment.csv'
    assignment = pd.read_csv(assignment_path, usecols=['TransactionID', 'partition'])
    train = get_partition(assignment, ['train'])
    validation = get_partition(assignment, ['validation'])
    if manifest['training_partition']['name'] != 'train' or len(train) != manifest['training_partition']['n_rows']:
        raise ValueError('Artifact must have been trained on the approved train partition only')
    if fingerprint(train.TransactionID) != manifest['training_partition']['row_fingerprint']:
        raise ValueError('Train partition fingerprint mismatch')
    if set(train.TransactionID) & set(validation.TransactionID):
        raise ValueError('Train and validation IDs overlap')
    frame = read_validation_only(ROOT / 'data/raw/ieee-fraud-detection/train_transaction.csv', assignment, list(artifact.schema.raw_columns))
    y = frame.isFraud.to_numpy()
    scores = np.concatenate([predict_fraud_probability(artifact, frame.iloc[start:start + 2048]) for start in range(0, len(frame), 2048)])
    # Canonicalize only numerical summation noise (documented ~1e-15).
    scores = np.round(scores, 12)
    n = len(frame)
    ks = sorted({math.ceil(n * i / 200) for i in range(201)})
    points = build_capacity_points(y, scores, frame.TransactionID.to_numpy(), ks)
    # Independent masks + sklearn confusion metrics, without reusing cumulative sums.
    order = sorted(range(n), key=lambda i: (-scores[i], int(frame.TransactionID.iloc[i])))
    for point in points:
        k = point['k']; predicted = np.zeros(n, dtype=int); predicted[order[:k]] = 1
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        assert [int(tp), int(fp), int(fn), int(tn)] == [point[x] for x in ['tp', 'fp', 'fn', 'tn']]
        if k: assert np.isclose(precision_score(y, predicted), point['precision'])
        if y.sum(): assert np.isclose(recall_score(y, predicted), point['recall'])
    payload = dict(schema_version='1.0.0', population='Chronological validation — development backtesting',
                   population_id='stage4-validation', n=n, positives=int(y.sum()), prevalence=float(y.mean()),
                   average_precision=float(average_precision_score(y, scores)),
                   model_sha256=EXPECTED, artifact_version=manifest['artifact_version'],
                   validation_id_fingerprint=fingerprint(frame.TransactionID),
                   train_id_fingerprint=fingerprint(train.TransactionID),
                   split_metadata_sha256=hashlib.sha256(assignment.to_csv(index=False).encode()).hexdigest(),
                   ranking='Descending frozen-model score rounded to 12 decimals; ascending TransactionID breaks ties',
                   independent_checks=len(points), checkpoint_count=len(points),
                   limitations=['Validation was consulted during model development: estimates may be optimistic.',
                                'Fixed historical batch, not a daily staffing forecast or future capture guarantee.',
                                'Scores are uncalibrated ranking signals; no automated declines or monetary claims.',
                                'Random baseline is an expectation under uniform sampling without replacement.',
                                'Only precomputed capacity checkpoints are selectable; metrics are never interpolated.',
                                'Synthetic demo filters and batches do not change this historical population.'], points=points)
    output = ROOT / 'ui/capacity-v1.json'
    output.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != EXPECTED: raise ValueError('Artifact changed')
    print(json.dumps({key: value for key, value in payload.items() if key != 'points'}, indent=2))


if __name__ == '__main__':
    main()
