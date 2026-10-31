"""Approved offline Stage23 build. No fit, outcomes, closed data or public inference."""
from __future__ import annotations
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.production_pipeline import load_artifact,predict_fraud_probability,compute_ordered_schema_fingerprint
from src.deployed_explainability import group_lineage,group_signed,rankings,leaf_baseline,reconcile,rank_stability
from src.explanation_assets import FROZEN,SEEDS,LIMITATIONS,generator_fingerprint,validate_assets
from scripts.build_monitoring_aggregate import read_validation_features,fingerprint

OUT=ROOT/'explainability_assets';EVIDENCE=ROOT/'docs/stage23_explainability_evidence'


def canonical(obj):
    def rounded(v):
        if isinstance(v,float):return round(v,12)
        if isinstance(v,dict):return {k:rounded(x) for k,x in v.items()}
        if isinstance(v,list):return [rounded(x) for x in v]
        return v
    return json.dumps(rounded(obj),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'


def main():
    started=time.perf_counter();EVIDENCE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    path=ROOT/'artifacts/frozen_rf_fraud_risk_v1.joblib'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==FROZEN
    artifact,manifest=load_artifact(path);cols=list(artifact.schema.raw_columns);names=list(artifact.schema.transformed_feature_names)
    assert tuple(artifact.preprocessor.get_feature_names_out())==artifact.schema.transformed_feature_names and len(names)==artifact.model.n_features_in_==813
    groups=group_lineage(names,cols,[c for c in cols if c not in artifact.schema.numeric_columns])
    mdi=np.array([artifact.model.feature_importances_[g['indices']].sum() for g in groups.values()]);assert np.isclose(mdi.sum(),1,atol=1e-12)
    meta=pd.read_csv(ROOT/'docs/_stage4_split_assignment.csv',usecols=['TransactionID','TransactionDT','partition'])
    meta=meta.loc[meta.partition.eq('validation')].copy();assert len(meta)==75560 and fingerprint(meta.TransactionID)=='14e5f38141f4ea86cf6228d76bf9af398c7cc3116d1ca49ddecc3b502e9ff5bc'
    selected=np.random.default_rng(42).choice(np.sort(meta.TransactionID.to_numpy()),size=512,replace=False)
    selected_meta=meta.loc[meta.TransactionID.isin(selected)].copy()
    sample=read_validation_features(ROOT/'data/raw/ieee-fraud-detection/train_transaction.csv',selected_meta,cols)
    from src.api import _synthetic_dashboard_row,build_dashboard_queue_payload
    synthetic=[];identifiers=[]
    for seed in SEEDS:
        rng=random.Random(seed)
        for i in range(48):synthetic.append(_synthetic_dashboard_row(artifact.schema,rng));identifiers.append(f'TXN-DEMO-{seed:04d}-{i:04d}')
    synthetic=pd.DataFrame(synthetic);baseline=leaf_baseline(artifact.model)
    header={'schema_version':'1.0.0','model_sha256':FROZEN,'ordered_schema_sha256':compute_ordered_schema_fingerprint(names),'generator_sha256':generator_fingerprint(cols)}
    global_data={**header,'groups':{k:{t:v for t,v in g.items() if t!='indices'} for k,g in groups.items()},'methods':{'mdi':rankings(mdi,groups,20)},'population':{'partition':'validation','n':512,'eligible':75560,'id_sha256':fingerprint(selected),'labels_used':False},'baseline':None,'mdi_total':float(mdi.sum()),'capabilities':{'mdi':True,'global_shap':False,'local_shap':False},'stability':{'status':'insufficient','reason':'SHAP unavailable'},'limitations':LIMITATIONS,'unavailable_reason':None}
    local_data={**header,'records':[]};evidence={'schema_order_verified':True,'raw_groups':len(groups),'transformed_features':len(names),'population':global_data['population'],'method':'MDI and exact path-dependent TreeSHAP; no fitting','python':platform.python_version(),'leaf_baseline':baseline,'checks':{}}
    # A SHAP failure never publishes partial explanations. Validated MDI is the fallback.
    try:
        import shap
        explainer=shap.TreeExplainer(artifact.model,feature_perturbation='tree_path_dependent',model_output='raw')
        if list(artifact.model.classes_)!=[0,1]:raise ValueError('Unsupported class mapping')
        def explain(frame,label):
            t=time.perf_counter();x=artifact.preprocessor.transform(frame[cols]);x=x.toarray() if hasattr(x,'toarray') else x
            result=explainer(x,check_additivity=True);v=np.asarray(result.values);b=np.asarray(result.base_values)
            if v.shape!=(len(frame),813,2) or b.shape!=(len(frame),2):raise ValueError('Unsupported SHAP shape')
            values=v[:,:,1];bases=b[:,1];scores=artifact.model.predict_proba(x)[:,1]
            if not np.allclose(bases,baseline,rtol=0,atol=1e-10):raise ValueError('Independent baseline mismatch')
            deployed=predict_fraud_probability(artifact,frame[cols]);assert np.allclose(scores,deployed,rtol=0,atol=1e-12)
            check=reconcile(values,bases,scores);grouped=group_signed(values,groups);reconcile(grouped,bases,deployed)
            evidence['checks'][label]={**check,'seconds':time.perf_counter()-t,'baseline_max_error':float(np.max(np.abs(bases-baseline)))}
            return grouped,scores
        probe=synthetic.iloc[[0,48]]
        explain(probe,'compatibility_probe')
        values,scores=explain(sample,'validation_global')
        global_data['methods']['shap']=rankings(np.abs(values).mean(axis=0),groups,20)
        global_data['baseline']=baseline;global_data['capabilities']['global_shap']=True
        global_data['stability']=rank_stability(values,groups,sample.TransactionDT.to_numpy(),sample.TransactionID.to_numpy())
        try:
            stress=[];missing=synthetic.iloc[0].copy();missing['dist1']=None;stress.append(missing)
            unseen=synthetic.iloc[0].copy();unseen['card4']='__STAGE23_UNSEEN__';stress.append(unseen)
            counts=sample.card4.dropna().astype(str).value_counts();rare=sorted((int(n),str(k)) for k,n in counts.items() if n<=5)
            if rare:
                row=synthetic.iloc[0].copy();row['card4']=rare[0][1];stress.append(row)
                evidence['rare_category_probe']={'eligible':True,'sample_count':rare[0][0],'scope':'rarity in fixed512 sample only'}
            else:evidence['rare_category_probe']={'eligible':False,'reason':'No nonmissing card4 category with1–5 sample observations'}
            explain(pd.DataFrame(stress),'synthetic_stress')
            local_values,local_scores=explain(synthetic,'synthetic_local')
            for seed in SEEDS:
                offset=SEEDS.index(seed)*48
                for n in [24,48]:
                    queue=build_dashboard_queue_payload(artifact,manifest,n,seed)
                    for i,record in enumerate(queue['transactions']):
                        if record['transaction_id']!=identifiers[offset+i] or abs(record['risk_score']-local_scores[offset+i])>5.00001e-7:raise ValueError('Queue score/generator mismatch')
            for identifier,vector,score in zip(identifiers,local_values,local_scores):
                top=sorted(zip(groups,vector),key=lambda v:(-abs(v[1]),v[0]))[:10];kept={name for name,_ in top}
                record={'transaction_id':identifier,'baseline':baseline,'score':float(score),'contributions':[{'name':name,'value':float(value)} for name,value in top],'other':float(sum(value for name,value in zip(groups,vector) if name not in kept))}
                record['other_positive']=float(sum(max(0,value) for name,value in zip(groups,vector) if name not in kept))
                record['other_negative']=float(sum(min(0,value) for name,value in zip(groups,vector) if name not in kept))
                assert abs(record['baseline']+sum(c['value'] for c in record['contributions'])+record['other']-record['score'])<=1e-8
                local_data['records'].append(record)
            global_data['capabilities']['local_shap']=True
        except Exception as exc:
            local_data['records']=[];global_data['unavailable_reason']='Local SHAP validation failed; only validated global methods are available.'
            evidence['local_blocker']=str(exc)
    except Exception as exc:
        global_data['methods'].pop('shap',None);global_data['capabilities'].update(global_shap=False,local_shap=False);global_data['baseline']=None
        global_data['stability']={'status':'insufficient','reason':'SHAP validation unavailable'};local_data['records']=[]
        global_data['unavailable_reason']='SHAP validation failed; validated MDI only.';evidence['shap_blocker']=str(exc)
    global_text=canonical(global_data);local_text=canonical(local_data)
    validate_assets(json.loads(global_text),json.loads(local_text),artifact)
    assert len(global_text.encode())<=131072 and len(local_text.encode())<=1048576
    hashes={};identical={}
    for name,text in [('global-v1.json',global_text),('local-v1.json',local_text)]:
        target=OUT/name;identical[name]=target.read_text()==text if target.exists() else None
        temporary=target.with_suffix('.tmp');temporary.write_text(text);temporary.replace(target);hashes[name]=hashlib.sha256(text.encode()).hexdigest()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==FROZEN
    evidence.update(seconds=time.perf_counter()-started,asset_sha256=hashes,identical_to_previous=identical,capabilities=global_data['capabilities'],frozen_sha256=FROZEN)
    print(json.dumps(evidence,indent=2,allow_nan=False))


if __name__=='__main__':main()
