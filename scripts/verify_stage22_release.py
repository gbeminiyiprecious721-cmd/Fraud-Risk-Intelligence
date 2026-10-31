"""Independent Stage22 release audit; label-free authorised validation only.

Reconstruct population membership and all published summaries/distributions.
Reuse only the existing ID-first access boundary and trusted frozen inference;
no production metric builders, labels, row-level outputs or model fitting.
Run: .venv/bin/python scripts/verify_stage22_release.py
"""
import sys,json,hashlib,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.distance import jensenshannon
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_monitoring_aggregate import read_validation_features
from src.production_pipeline import load_artifact,predict_fraud_probability
EXPECTED='2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04'


def main():
    assert hashlib.sha256((ROOT/'artifacts/frozen_rf_fraud_risk_v1.joblib').read_bytes()).hexdigest()==EXPECTED
    published=json.loads((ROOT/'ui/monitoring-v1.json').read_text());checks=0
    meta=pd.read_csv(ROOT/'docs/_stage4_split_assignment.csv',usecols=['TransactionID','TransactionDT','partition'])
    meta=meta.loc[meta.partition.eq('validation')].sort_values(['TransactionDT','TransactionID'])
    assert len(meta)==75560
    ranges=[(11093842,11915077,25186),(11915125,12739542,25187),(12739588,13452406,25187)]
    ids=[]
    for descriptor,(start,end,n) in zip(published['windows'],ranges):
     w=meta.loc[meta.TransactionDT.between(start,end)];assert len(w)==n
     fingerprint=hashlib.sha256('\x1f'.join(map(str,sorted(w.TransactionID))).encode()).hexdigest()
     assert descriptor['id_sha256']==fingerprint and descriptor['n']==n and descriptor['start']==start and descriptor['end']==end
     ids.append(set(w.TransactionID));checks+=1
    assert set.union(*ids)==set(meta.TransactionID)
    assert all(not a&b for i,a in enumerate(ids) for b in ids[i+1:])
    artifact,_=load_artifact(ROOT/'artifacts/frozen_rf_fraud_risk_v1.joblib')
    f=read_validation_features(ROOT/'data/raw/ieee-fraud-detection/train_transaction.csv',meta,list(artifact.schema.raw_columns))
    f['_score']=np.round(np.concatenate([predict_fraud_probability(artifact,f.iloc[i:i+2048][list(artifact.schema.raw_columns)]) for i in range(0,len(f),2048)]),12)
    f['daily_phase']=((f.TransactionDT%86400)//3600).map(lambda h:f'{int(h):02d}')
    windows=[f.loc[f.TransactionID.isin(i)] for i in ids]
    def eq(a,b):
     nonlocal checks
     if isinstance(a,(int,float,np.number)) and b is not None:assert np.isclose(a,b,rtol=1e-12,atol=1e-12),(a,b)
     else:assert a==b,(a,b)
     checks+=1
    def summary(values,stats):
     x=pd.to_numeric(values,errors='coerce');x=x[np.isfinite(x)]
     eq(len(x),stats['n']);eq(float(x.mean()),stats['mean'])
     for a,b in zip(np.quantile(x,[.01,.1,.5,.9,.95,.99]),stats['quantiles']):eq(a,b)
    for w,d in zip(windows,published['windows']):
     summary(w._score,d['score']);eq(int(w.TransactionID.duplicated().sum()),d['duplicate_ids']);eq(int((w.TransactionAmt<0).sum()),d['negative_amounts'])
    threshold=float(np.quantile(windows[0]._score,.95));eq(threshold,published['reference_tail_threshold'])
    for w,d in zip(windows,published['windows']):eq(float((w._score>threshold).mean()),d['tail_fraction'])
    for w,result in zip(windows[1:],published['comparisons']):
     for name,r in [*result['features'].items(),('_score',result['score'])]:
      a=windows[0][name];b=w[name]
      for values,q in [(a,r['reference_quality']),(b,r['comparison_quality'])]:
       eq(len(values),q['n']);eq(int(values.isna().sum()),q['missing']);eq(float(values.isna().mean()),q['missing_rate'])
       if r['kind']=='numeric':
        z=pd.to_numeric(values,errors='coerce');eq(int((z.isna()&values.notna()).sum()),q['parse_failures']);eq(int((z.notna()&~np.isfinite(z)).sum()),q['nonfinite']);eq(int(np.isfinite(z).sum()),q['finite'])
      eq(float(b.isna().mean()-a.isna().mean()),r['missing_delta'])
      if r['kind']=='numeric':
       summary(a,r['reference_summary']);summary(b,r['comparison_summary'])
       a=np.sort(np.asarray(a.dropna(),dtype=float));b=np.sort(np.asarray(b.dropna(),dtype=float));a=a[np.isfinite(a)];b=b[np.isfinite(b)]
       edges=np.unique(np.quantile(a,np.linspace(0,1,11)))[1:-1]
       eq(edges.tolist(),r['edges'])
       ac=np.histogram(a,np.r_[-np.inf,edges,np.inf])[0];bc=np.histogram(b,np.r_[-np.inf,edges,np.inf])[0]
       eq(ac.tolist(),r['reference_counts']);eq(bc.tolist(),r['comparison_counts'])
       p=ac/ac.sum()+.0001;q=bc/bc.sum()+.0001;p=p/p.sum();q=q/q.sum();computed_psi=float(np.dot(q-p,np.log(q/p)))
       grid=np.unique(np.r_[a,b]);computed_ks=float(np.max(np.abs(np.searchsorted(a,grid,side='right')/len(a)-np.searchsorted(b,grid,side='right')/len(b))))
       eq(computed_psi,r['psi']);eq(computed_ks,r['ks'])
       eq('Investigate shift' if computed_psi>=.2 or computed_ks>=.1 else 'No material shift under selected rules',r['signal'])
      else:
       cats=sorted(set(a.dropna().astype(str)));eq(cats+['[Missing]','[Unseen]'],r['categories'])
       def counts(s):
        v=s.dropna().astype(str)
        return [int(v.eq(c).sum()) for c in cats]+[int(s.isna().sum()),int((~v.isin(cats)).sum())]
       ac=counts(a);bc=counts(b);eq(ac,r['reference_counts']);eq(bc,r['comparison_counts']);eq(bc[-1],r['unseen_count'])
       js=float(jensenshannon(ac,bc,base=2)**2);eq(js,r['js']);eq('Investigate shift' if js>=.05 else 'No material shift under selected rules',r['signal'])
    assert hashlib.sha256((ROOT/'artifacts/frozen_rf_fraud_risk_v1.joblib').read_bytes()).hexdigest()==EXPECTED
    print(json.dumps({'reviewed_commit':'d4995276fd4dec6e93ab33e5d18ba62bb24e568e','window_rows':[len(w) for w in windows],'strict_chronology':True,'nonoverlap_and_exact_validation_cover':True,'independent_scalar_and_vector_assertions':checks,'metrics':['PSI','KS direct empirical CDF','Jensen-Shannon divergence via squared SciPy distance','missingness','quality counts','score summaries/quantiles','fixed-q95 tail fractions','reference-only histogram bins/counts'],'all_published_values_match':True,'outcomes_read':False,'frozen_sha256':EXPECTED},indent=2))


if __name__=='__main__':main()
