"""Stage 25 source, numerical, documentation and local public-security audit.

    .venv/bin/python scripts/verify_stage25_release.py --serving-python <minimal-env>/bin/python

Reads published summaries/approved aggregates only. Never opens a closed final
partition/prediction file or fits a model. Access-control probes use a separately
loaded trusted artifact in a disposable TestClient, not the live service.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from urllib.parse import unquote, urlsplit

import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'docs/stage25_release_evidence'
BASE='3fd88535b89d554c8955c0dcf0a4333f06ca6e92'
FROZEN='2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04'
DOCS=['README.md','docs/ARCHITECTURE.md','docs/PORTFOLIO_WALKTHROUGH.md','docs/PORTFOLIO_CASE_STUDY.md','docs/ML_WORKFLOW_STATE.md','docs/STAGE19_CAPACITY_SIMULATOR.md','docs/STAGE20_CHALLENGER_BENCHMARK.md','docs/STAGE21_CASE_MANAGEMENT.md','docs/STAGE21_QUEUE_FILTER_CORRECTIONS.md','docs/STAGE22_MODEL_MONITORING.md','docs/STAGE23_MODEL_EXPLAINABILITY.md','docs/STAGE24_PORTFOLIO_DOCUMENTATION.md','docs/STAGE25_FINAL_RELEASE_REPORT.md']


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def read(path):return json.loads((ROOT/path).read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--serving-python',type=Path)
    parser.add_argument('--docs-only',action='store_true')
    args=parser.parse_args()
    if not args.docs_only and args.serving_python is None:
        parser.error('--serving-python is required for the full local audit')
    result={'base':BASE,'closed_final_test_partition_or_predictions_opened':False}
    spec=importlib.util.spec_from_file_location('documentation_helpers',ROOT/'docs/stage24_portfolio_evidence/verify_documentation.py')
    docs=importlib.util.module_from_spec(spec);spec.loader.exec_module(docs)
    links=images=diagrams=0;errors=[]
    for name in DOCS:
        path=ROOT/name;text=path.read_text();assert text.count('```')%2==0,name
        for node in docs.walk(docs.parse(text)):
            if node['type']=='block_code' and node.get('attrs',{}).get('info')=='mermaid':diagrams+=1
            if node['type'] not in ('link','image'):continue
            url=urlsplit(node['attrs']['url'])
            if url.scheme or url.netloc:continue
            links+=1;images+=node['type']=='image'
            target=path.parent/unquote(url.path) if url.path else path
            if not target.exists():errors.append((name,node['attrs']['url'],'missing file'))
            elif url.fragment and target.suffix=='.md' and unquote(url.fragment) not in docs.anchors(target):errors.append((name,node['attrs']['url'],'missing anchor'))
    assert not errors,errors
    # This broader scope includes historical Mermaid blocks as well; the current
    # three architecture sources are separately rendered in real Chrome.
    result['documentation']={'files':DOCS,'internal_targets':links,'inline_images':images,'all_mermaid_blocks_in_scope':diagrams,'errors':errors}
    inventory=(OUT/'changed-files.txt').read_text().splitlines()
    observed=set(git('diff','--name-only',BASE).splitlines())
    observed.update(git('ls-files','--others','--exclude-standard').splitlines())
    assert inventory and inventory==sorted(observed), 'Release inventory is empty or differs from the actual changed files'
    result['documentation']['changed_file_count']=len(inventory)
    if args.docs_only:
        OUT.mkdir(exist_ok=True)
        (OUT/'documentation-checks.json').write_text(json.dumps(result['documentation'],indent=2)+'\n')
        print(json.dumps(result['documentation'],indent=2))
        return
    manifest=read('artifacts/frozen_rf_fraud_risk_v1.joblib.manifest.json')
    assert hashlib.sha256((ROOT/'artifacts/frozen_rf_fraud_risk_v1.joblib').read_bytes()).hexdigest()==manifest['artifact_sha256']==FROZEN
    assert len(manifest['raw_feature_schema'])==392 and len(manifest['transformed_feature_schema'])==813
    registry=[json.loads(line) for line in (ROOT/'docs/experiment_registry.jsonl').read_text().splitlines()]
    heldout=[r['metrics'] for r in registry if r.get('workflow_stage')==10 and (r.get('metrics') or {}).get('n_rows')==76855]
    assert len(heldout)==1
    h=heldout[0];assert (h['average_precision'],h['roc_auc'])==(.42784,.86637)
    assert (h['k_frac_0.01']['precision_at_k'],h['k_frac_0.01']['recall_at_k'])==(.8583,.2461)
    b=read('ui/benchmark-v1.json');c=read('ui/capacity-v1.json');m=read('ui/monitoring-v1.json')
    assert (b['rows'],b['positives'],c['n'],c['positives'])==(75560,2617,75560,2617)
    assert len(c['points'])==201 and sum(len(x['metrics']['points']) for x in b['models'])==804
    published={name:(float(ap),float(roc)) for name,ap,roc in re.findall(r'\| (Random Forest|LightGBM|XGBoost|CatBoost) \| (0\.\d+) \| (0\.\d+) \|',(ROOT/'README.md').read_text())}
    assert published=={x['name']:tuple(round(x['metrics'][k],5) for k in ['average_precision','roc_auc']) for x in b['models']}
    assert [w['n'] for w in m['windows']]==[25186,25187,25187]
    assert all(a['end']<z['start'] for a,z in zip(m['windows'],m['windows'][1:]))
    g=read('explainability_assets/global-v1.json');l=read('explainability_assets/local-v1.json')
    assert g['population']['n']==512 and not g['population']['labels_used'] and len(g['groups'])==392 and len(l['records'])==240
    result['numerical']={'stage10_published_metrics':{'ap':h['average_precision'],'roc_auc':h['roc_auc'],'precision_top1':h['k_frac_0.01']['precision_at_k'],'recall_top1':h['k_frac_0.01']['recall_at_k']},'reused_validation_rows':75560,'capacity_checkpoints':201,'benchmark_checkpoints':804,'monitoring_window_counts':[25186,25187,25187],'global_label_free_sample':512,'local_synthetic_records':240,'historical_metrics_or_assets_recomputed':False}
    baseline=read('docs/stage25_release_evidence/baseline.json')
    for path,expected in baseline['source_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    old=subprocess.check_output(['git','show',BASE+':docs/ML_WORKFLOW_STATE.md'],cwd=ROOT,text=True)
    assert (ROOT/'docs/ML_WORKFLOW_STATE.md').read_text().endswith(old.split('\n\n',1)[1])
    for path in ['src/api.py','src/production_pipeline.py','src/preprocessing.py','ui/case-store.js','ui/case-ui.js','artifacts','explainability_assets','requirements-render.txt','requirements.txt','requirements-benchmark.txt','deploy','docs/experiment_registry.jsonl','docs/feature_availability_register.csv','docs/engineered_features_register.csv']:
        assert not git('diff','--name-only',BASE,'--',path),path
    result['preservation']={'frozen_sha256':FROZEN,'protected_sources_and_assets_unchanged':True,'original_workflow_body_preserved':True,'main':git('rev-parse','main')}
    assert result['preservation']['main']=='52c01c919cb40cd2bedf1a41794a73e38c6cf737'
    # Hooks affect a separate loaded test object only, never the artifact bytes.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        from fastapi.testclient import TestClient
        import src.render_demo as public
        with TestClient(public.app) as client:
            artifact=client.app.state.artifact
            calls=[]
            def forbidden(*a,**kw):calls.append('inference_or_transform');raise AssertionError('Unrestricted inference during explanation lookup')
            predict,transform=artifact.model.predict_proba,artifact.preprocessor.transform
            artifact.model.predict_proba=forbidden;artifact.preprocessor.transform=forbidden
            allowed=denied=0
            try:
                assert client.get('/v1/dashboard/explainability').status_code==200
                for seed in public.PUBLIC_DASHBOARD_SEED_ALLOWLIST:
                    for sequence in range(48):
                        identifier=f'TXN-DEMO-{seed:04d}-{sequence:04d}'
                        response=client.get('/v1/dashboard/explanation',params={'transaction_id':identifier})
                        assert response.status_code==(200 if sequence<24 else 404),(identifier,response.status_code)
                        if sequence<24:allowed+=1
                        else:denied+=1
                assert not calls and 'shap' not in sys.modules
            finally:artifact.model.predict_proba=predict;artifact.preprocessor.transform=transform
            private_paths=['/.env','/.git/config','/artifacts/frozen_rf_fraud_risk_v1.joblib','/data/raw/ieee-fraud-detection/train_transaction.csv','/explainability_assets/local-v1.json']
            for path in private_paths:assert client.get(path).status_code==404,path
            for path in ['/v1/cases','/v1/dashboard/cases']:
                for method in ['post','put','patch','delete']:assert getattr(client,method)(path).status_code==404
            marker='stage25_synthetic_logging_probe'
            invalid=client.post('/v1/predict',json={'note':marker});assert invalid.status_code==403
            assert marker not in invalid.text and 'traceback' not in invalid.text.lower()
        result['access_control']={'approved_ids':allowed,'denied_nonpublic_ids':denied,'runtime_inference_or_transforms_for_explanations':len(calls),'shap_imported':False,'private_asset_paths_denied':len(private_paths),'case_write_methods_denied':8,'unsupported_input_generic_403':True,'audit_warnings':[{'category':type(w.message).__name__,'message':str(w.message)} for w in caught]}
    for file in (ROOT/'deploy').glob('*.sh'):subprocess.run(['bash','-n',str(file)],check=True)
    env_python=str(args.serving_python.absolute())
    check=subprocess.run([env_python,'-m','pip','--no-cache-dir','check'],capture_output=True,text=True,check=True)
    code="import json;from pathlib import Path;from importlib.metadata import version;from importlib.util import find_spec;p={l.split('==')[0]:l.split('==')[1] for l in Path('requirements-render.txt').read_text().splitlines() if l and not l.startswith('#')};assert all(version(k)==v for k,v in p.items());print(json.dumps({'pins':p,'offline_modules_present':{k:find_spec(k) is not None for k in ['shap','xgboost','lightgbm','catboost']}}))"
    runtime=json.loads(subprocess.check_output([env_python,'-c',code],cwd=ROOT,text=True))
    assert not any(runtime['offline_modules_present'].values())
    result['deployment_checks']={'shell_syntax':'passed','minimal_runtime':runtime,'pip_check':check.stdout.strip(),'private_artifact_fetch_executed':False,'render_or_external_deployment_triggered':False}
    # Live GETs only: no negative/load tests or inference against Render.
    result['public_urls']=[]
    for url in ['https://fraud-risk-intelligence.onrender.com/health','https://www.kaggle.com/competitions/ieee-fraud-detection/data']:
        attempts=[]
        for attempt in range(2 if '/health' in url else 1):
            try:
                response=requests.get(url,timeout=45)
                attempts.append({'status':response.status_code})
                if '/health' in url:assert response.status_code==200 and response.json().get('artifact_loaded')
                break
            except requests.RequestException as exc:attempts.append({'error':type(exc).__name__})
        result['public_urls'].append({'url':url,**attempts[-1],'attempts':attempts})
    # Strict original dataset URL remains correct; HTTP failure is disclosed,
    # not hidden by redirecting readers to an unrelated dataset.
    OUT.mkdir(exist_ok=True)
    (OUT/'release-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['public_urls'][0].get('status')==200, result['public_urls'][0]
    assert 'https://www.kaggle.com/competitions/ieee-fraud-detection/data' in (ROOT/'README.md').read_text()
    result['audit_completed']=True
    OUT.mkdir(exist_ok=True);(OUT/'release-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
