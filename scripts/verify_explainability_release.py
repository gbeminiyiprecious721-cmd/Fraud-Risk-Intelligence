"""Independent Stage23 release audit against fitted transformer metadata.

No fitting, outcomes, closed records or row-level output. Reconstruct grouping
from fitted imputer/encoder metadata instead of the builder's feature-name parser;
recompute SHAP on the approved512 validation and240 deterministic synthetic rows.
"""
from __future__ import annotations
import hashlib
import json
import random
import sys
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.production_pipeline import load_artifact,predict_fraud_probability
from src.api import _synthetic_dashboard_row
from scripts.build_monitoring_aggregate import read_validation_features
from src.explanation_assets import FROZEN,SEEDS


def public_scope():
    # A fresh subprocess proves the serving import graph does not import SHAP.
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from src.render_demo import app
    assert 'shap' not in sys.modules
    allowed=denied=0
    with TestClient(app) as client:
        artifact=app.state.artifact
        with patch.object(artifact.model,'predict_proba',side_effect=AssertionError('Public inference forbidden')) as predict, patch.object(artifact.preprocessor,'transform',side_effect=AssertionError('Public transform forbidden')) as transform:
            global_response=client.get('/v1/dashboard/explainability');assert global_response.status_code==200 and len(global_response.content)<=16384
            assert 'records' not in global_response.json() and 'groups' not in global_response.json()
            for seed in SEEDS:
                for i in range(48):
                    identifier=f'TXN-DEMO-{seed:04d}-{i:04d}'
                    response=client.get('/v1/dashboard/explanation',params={'transaction_id':identifier,'limit':200,'n':200})
                    if i<24:
                        assert response.status_code==200 and response.json()['transaction_id']==identifier and len(response.content)<=16384;allowed+=1
                    else:assert response.status_code==404;denied+=1
            for identifier in ['TXN-DEMO-9999-0000','TXN-DEMO-0101-0048','3155871','../artifact']:
                assert client.get('/v1/dashboard/explanation',params={'transaction_id':identifier}).status_code in (404,422)
            for resource in ['/explainability_assets/global-v1.json','/explainability_assets/local-v1.json','/local-v1.json']:
                assert client.get(resource).status_code==404
            for method in ['post','put','patch','delete']:
                for route in ['/v1/dashboard/explainability','/v1/dashboard/explanation']:
                    assert client.request(method.upper(),route,json={'notes':'Synthetic test-only canary'}).status_code==405
            assert predict.call_count==transform.call_count==0 and 'shap' not in sys.modules
    print(json.dumps({'approved_identifiers_accessible':allowed,'nonpublic_local_identifiers_denied':denied,'scope_override_ignored':True,'static_private_paths_denied':True,'write_methods_denied':True,'serving_shap_imported':False,'request_time_inference_calls':0,'request_time_transform_calls':0,'response_size_bounds_verified':True}))


def main():
    started=time.perf_counter();path=ROOT/'artifacts/frozen_rf_fraud_risk_v1.joblib'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==FROZEN
    artifact,_=load_artifact(path);ct=artifact.preprocessor;raw=list(artifact.schema.raw_columns);ordered=list(artifact.schema.transformed_feature_names)
    global_data=json.loads((ROOT/'explainability_assets/global-v1.json').read_text());local_data=json.loads((ROOT/'explainability_assets/local-v1.json').read_text())
    mapping=[];reconstructed=[];counts={name:{'numeric':0,'missingness_indicator':0,'categorical_onehot':0} for name in raw}
    for name,transformer,cols in ct.transformers_:
        start=len(mapping)
        if name in ['numeric_plain','numeric_indicator']:
            imputer=transformer.named_steps['impute'];assert not pd.isna(imputer.statistics_).any()
            for col in cols:mapping.append(col);reconstructed.append(f'{name}__{col}');counts[col]['numeric']+=1
            if imputer.add_indicator:
                for index in imputer.indicator_.features_:
                    col=cols[index];mapping.append(col);reconstructed.append(f'{name}__missingindicator_{col}');counts[col]['missingness_indicator']+=1
        elif name=='categorical':
            encoder=transformer.named_steps['onehot'];assert encoder.drop_idx_ is None
            for col,categories in zip(cols,encoder.categories_):
                for category in categories:
                    mapping.append(col);reconstructed.append(f'{name}__{col}_{category}');counts[col]['categorical_onehot']+=1
        else:raise ValueError('Unexpected fitted transformer')
        assert ct.output_indices_[name]==slice(start,len(mapping))
    assert reconstructed==ordered==list(ct.get_feature_names_out()) and len(mapping)==813 and len(counts)==392
    assert counts==global_data['groups']
    names=sorted(raw);indices={name:i for i,name in enumerate(names)}
    matrix=np.zeros((813,392))
    for i,name in enumerate(mapping):matrix[i,indices[name]]=1
    assert np.all(matrix.sum(axis=1)==1) and np.all(matrix.sum(axis=0)>0)
    mdi=artifact.model.feature_importances_@matrix;assert abs(mdi.sum()-1)<=1e-12
    mdi_order=sorted(range(392),key=lambda i:(-mdi[i],names[i]))[:20]
    assert [names[i] for i in mdi_order]==[r['name'] for r in global_data['methods']['mdi']]
    for r in global_data['methods']['mdi']:assert abs(mdi[indices[r['name']]]-r['value'])<=1e-12
    root_baseline=float(np.mean([t.tree_.value[0,0,1]/t.tree_.value[0,0,:].sum() for t in artifact.model.estimators_]))
    assert abs(root_baseline-global_data['baseline'])<=1e-10 and list(artifact.model.classes_)==[0,1]
    meta=pd.read_csv(ROOT/'docs/_stage4_split_assignment.csv',usecols=['TransactionID','TransactionDT','partition'])
    meta=meta.loc[meta.partition.eq('validation')];assert len(meta)==75560
    chosen=np.random.default_rng(42).choice(np.sort(meta.TransactionID.to_numpy()),512,replace=False)
    sample_digest=hashlib.sha256('\x1f'.join(map(str,sorted(chosen))).encode()).hexdigest()
    assert sample_digest==global_data['population']['id_sha256'] and global_data['population']['labels_used'] is False
    frame=read_validation_features(ROOT/'data/raw/ieee-fraud-detection/train_transaction.csv',meta.loc[meta.TransactionID.isin(chosen)],list(artifact.schema.raw_columns))
    assert len(frame)==512 and 'isFraud' not in frame
    import shap
    explainer=shap.TreeExplainer(artifact.model,feature_perturbation='tree_path_dependent',model_output='raw')
    errors=[]
    def explain(rows):
        x=ct.transform(rows[list(artifact.schema.raw_columns)]);x=x.toarray() if hasattr(x,'toarray') else x
        explanation=explainer(x);v=np.asarray(explanation.values);b=np.asarray(explanation.base_values)
        assert v.shape==(len(rows),813,2) and b.shape==(len(rows),2)
        assert np.max(np.abs(b[:,1]-root_baseline))<=1e-10
        scores=predict_fraud_probability(artifact,rows)
        grouped=v[:,:,1]@matrix
        error=float(np.max(np.abs(grouped.sum(axis=1)+b[:,1]-scores)));assert error<=1e-8;errors.append(error)
        return grouped,scores
    grouped,_=explain(frame);importance=np.abs(grouped).mean(axis=0)
    ranking=sorted(range(392),key=lambda i:(-importance[i],names[i]))[:20]
    assert [names[i] for i in ranking]==[r['name'] for r in global_data['methods']['shap']]
    for r in global_data['methods']['shap']:assert abs(importance[indices[r['name']]]-r['value'])<=1e-12
    rows=[];ids=[]
    for seed in SEEDS:
        rng=random.Random(seed)
        for i in range(48):rows.append(_synthetic_dashboard_row(artifact.schema,rng));ids.append(f'TXN-DEMO-{seed:04d}-{i:04d}')
    local_grouped,scores=explain(pd.DataFrame(rows));records={r['transaction_id']:r for r in local_data['records']};assert set(ids)==set(records) and len(records)==240
    score_errors=[];value_errors=[];serialised_errors=[]
    for identifier,vector,score in zip(ids,local_grouped,scores):
        record=records[identifier];score_errors.append(abs(record['score']-score))
        keep={c['name'] for c in record['contributions']};assert len(keep)==10
        for c in record['contributions']:value_errors.append(abs(c['value']-vector[indices[c['name']]]))
        omitted=np.array([value for name,value in zip(names,vector) if name not in keep])
        assert max(abs(value) for name,value in zip(names,vector) if name not in keep)<=min(abs(c['value']) for c in record['contributions'])+1e-10
        value_errors.extend([abs(record['other']-omitted.sum()),abs(record['other_positive']-omitted[omitted>0].sum()),abs(record['other_negative']-omitted[omitted<0].sum())])
        serialised_errors.append(abs(record['baseline']+sum(c['value'] for c in record['contributions'])+record['other']-record['score']))
    assert max(score_errors)<=1e-10 and max(value_errors)<=1e-10 and max(serialised_errors)<=1e-8
    assert hashlib.sha256(path.read_bytes()).hexdigest()==FROZEN
    scope_result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--public-scope'],cwd=ROOT,capture_output=True,text=True,check=True)
    scope=json.loads(scope_result.stdout)
    print(json.dumps({'mapped_columns':813,'raw_groups':392,'components_from_fitted_metadata':True,'complete_unique_mapping':True,'mdi_group_total':float(mdi.sum()),'mdi_top20_reproduced':True,'global_shap_rows':512,'selected_id_sha256':sample_digest,'global_shap_top20_reproduced':True,'synthetic_records_recomputed':240,'local_signed_contributions_and_other_reproduced':True,'public_identifier_inventory':120,'root_baseline':root_baseline,'max_raw_grouped_additivity_error':max(errors),'max_serialised_reconciliation_error':max(serialised_errors),'max_synthetic_score_error':max(score_errors),'max_attribution_difference':max(value_errors),'frozen_sha256':FROZEN,'outcomes_or_closed_records_accessed':False,'public_scope_and_no_inference':scope,'seconds':time.perf_counter()-started},indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['--public-scope']:public_scope()
    else:main()
