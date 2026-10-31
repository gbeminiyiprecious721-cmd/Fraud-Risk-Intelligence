"""Build Stage22 aggregate from authorised validation features only, never labels.

No persistent row-level output. ID-first physical filtering prevents parsing
closed partition features. Fixed model loaded only after checksum verification.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.drift_intelligence import FROZEN_SHA256, FEATURES, build_payload, fingerprint, split_windows, validate_aggregate
from src.production_pipeline import load_artifact,predict_fraud_probability

EXPECTED_VALIDATION='14e5f38141f4ea86cf6228d76bf9af398c7cc3116d1ca49ddecc3b502e9ff5bc'


def read_validation_features(source,metadata,columns):
    if not metadata.partition.eq('validation').all() or metadata.TransactionID.duplicated().any():raise ValueError('Authorised validation IDs required')
    if 'isFraud' in columns:raise ValueError('Outcome access prohibited')
    ids=pd.read_csv(source,usecols=['TransactionID']).TransactionID
    if ids.duplicated().any():raise ValueError('Duplicate source IDs')
    allowed=set(metadata.TransactionID)
    if not allowed.issubset(set(ids)):raise ValueError('Missing validation IDs')
    keep=ids.isin(allowed).to_numpy()
    columns=list(dict.fromkeys(['TransactionID','TransactionDT',*columns]))
    x=pd.read_csv(source,usecols=columns,skiprows=lambda row:row>0 and not keep[row-1])
    if len(x)!=len(metadata) or set(x.TransactionID)!=allowed:raise ValueError('Population mismatch')
    times=metadata.set_index('TransactionID').TransactionDT
    if not np.array_equal(x.TransactionDT,times.loc[x.TransactionID]):raise ValueError('Timestamp mismatch')
    x=x.copy()
    x['partition']='validation'
    return x.sort_values(['TransactionDT','TransactionID']).reset_index(drop=True)


def independent_check(payload,frame,scores):
    """Independent SciPy/NumPy assertions, not calls to production metric functions."""
    x=frame.copy();x['_score']=scores
    x['daily_phase']=((x.TransactionDT%86400)//3600).map(lambda h:f'{int(h):02d}')
    ws=split_windows(x);checks=0
    for result,w in zip(payload['comparisons'],ws[1:]):
        for name,r in [*result['features'].items(),('_score',result['score'])]:
            a=ws[0][name];b=w[name]
            assert r['reference_quality']['missing']==int(a.isna().sum())
            assert r['comparison_quality']['missing']==int(b.isna().sum())
            if r['kind']=='numeric':
                a=pd.to_numeric(a,errors='coerce');b=pd.to_numeric(b,errors='coerce');a=a[np.isfinite(a)];b=b[np.isfinite(b)]
                if r['ks'] is not None:
                    grid=np.unique(np.r_[a,b])
                    distance=float(np.max(np.abs(np.searchsorted(np.sort(a),grid,side='right')/len(a)-np.searchsorted(np.sort(b),grid,side='right')/len(b))))
                    assert np.isclose(r['ks'],distance,atol=1e-12)
                    assert np.isclose(r['ks'],ks_2samp(a,b,method='asymp').statistic,atol=1e-12)
                if len(r['labels'])==2 and r['labels'][0]=='Reference value':
                    ac=[len(a),0];bc=[int((b==a.iloc[0]).sum()),int((b!=a.iloc[0]).sum())]
                else:
                    bins=np.r_[-np.inf,np.asarray(r['edges']),np.inf]
                    ac=np.histogram(a,bins)[0];bc=np.histogram(b,bins)[0]
                assert np.array_equal(ac,r['reference_counts']) and np.array_equal(bc,r['comparison_counts'])
                if r['psi'] is not None:
                    p=np.array(ac,dtype=float);q=np.array(bc,dtype=float);p=p/p.sum()+.0001;q=q/q.sum()+.0001;p/=p.sum();q/=q.sum()
                    assert np.isclose(r['psi'],np.dot(q-p,np.log(q/p)),atol=1e-12)
            else:
                cats=r['categories'][:-2]
                def counts(s):return [int(s.eq(c).sum()) for c in cats]+[int(s.isna().sum()),int((s.notna()&~s.isin(cats)).sum())]
                ac=counts(a);bc=counts(b)
                assert ac==r['reference_counts'] and bc==r['comparison_counts']
                if r['js'] is not None:assert np.isclose(r['js'],jensenshannon(ac,bc,base=2)**2,atol=1e-12)
            checks+=1
    return checks


def main():
    artifact_path=ROOT/'artifacts/frozen_rf_fraud_risk_v1.joblib'
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest()==FROZEN_SHA256
    manifest=json.loads(Path(str(artifact_path)+'.manifest.json').read_text())
    columns=manifest['raw_feature_schema']
    metadata=pd.read_csv(ROOT/'docs/_stage4_split_assignment.csv',usecols=['TransactionID','TransactionDT','partition'])
    metadata=metadata.loc[metadata.partition.eq('validation')].copy()
    assert len(metadata)==75560 and fingerprint(metadata.TransactionID)==EXPECTED_VALIDATION
    frame=read_validation_features(ROOT/'data/raw/ieee-fraud-detection/train_transaction.csv',metadata,columns)
    artifact,_=load_artifact(artifact_path)
    scores=np.concatenate([predict_fraud_probability(artifact,frame.iloc[i:i+2048][columns]) for i in range(0,len(frame),2048)])
    scores=np.round(scores,12) # Canonicalise documented parallel summation noise only.
    payload=build_payload(frame,scores)
    checks=independent_check(payload,frame,scores)
    validate_aggregate(payload)
    canonical=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    target=ROOT/'ui/monitoring-v1.json'
    previous=target.read_text() if target.exists() else None
    target.write_text(canonical)
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest()==FROZEN_SHA256
    evidence={'population':'authorised validation features only; no outcomes','n':len(frame),'model_sha256':FROZEN_SHA256,
              'validation_id_sha256':EXPECTED_VALIDATION,'aggregate_sha256':hashlib.sha256(canonical.encode()).hexdigest(),
              'independent_distribution_checks':checks,'identical_to_previous_build':previous==canonical if previous else None,
              'windows':[{k:v for k,v in w.items() if k not in ['score','tail_fraction']} for w in payload['windows']]}
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
