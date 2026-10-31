"""Approved local-only Stage 20 experiment. Never trains or saves the incumbent.

Run with .venv/bin/python scripts/run_stage20_benchmark.py all. Private outputs
are gitignored. Each fit/scoring phase runs in a fresh, monitored subprocess.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRIVATE = ROOT / 'artifacts/stage20'
EVIDENCE = ROOT / 'docs/stage20_challenger_evidence'
FROZEN = ROOT / 'artifacts/frozen_rf_fraud_risk_v1.joblib'
FROZEN_HASH = '2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04'
NAMES = {'rf': 'Random Forest', 'lgb': 'LightGBM', 'xgb': 'XGBoost', 'cat': 'CatBoost'}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)  # XGBoost's missing-value sentinel, not a metric
        if isinstance(value, dict):
            return {key:finite(child) for key,child in value.items()}
        if isinstance(value, (list,tuple)):
            return [finite(child) for child in value]
        return value
    path.write_text(json.dumps(finite(obj), indent=2, allow_nan=False) + '\n')


def fingerprint(ids):
    return hashlib.sha256('\x1f'.join(map(str, sorted(map(int, ids)))).encode()).hexdigest()


def check_environment():
    from importlib.metadata import version
    for name, expected in {'lightgbm':'4.7.0','xgboost':'3.4.1','catboost':'1.2.10','graphviz':'0.21','plotly':'7.1.0'}.items():
        if version(name) != expected:
            raise RuntimeError(f'Unapproved dependency version: {name}')


def check_frozen():
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_HASH


def read_authorised_rows(source, allowed, columns):
    """Locate rows by IDs only, then skip disallowed rows BEFORE parsing values."""
    import pandas as pd
    id_frame = pd.read_csv(source, usecols=['TransactionID'])
    keep = id_frame.TransactionID.isin(set(allowed)).to_numpy()
    frame = pd.read_csv(source, usecols=list(dict.fromkeys(['TransactionID', 'TransactionDT', 'isFraud', *columns])),
                        skiprows=lambda row: row > 0 and not keep[row-1])
    if len(frame) != len(allowed) or frame.TransactionID.duplicated().any() or set(frame.TransactionID) != set(allowed):
        raise ValueError('Authorised population mismatch')
    return frame.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)


def prepare():
    import numpy as np
    import pandas as pd
    from numpy.lib.format import open_memmap
    from src.production_pipeline import load_artifact, predict_fraud_probability
    from src.preprocessing import build_preprocessing_pipeline, select_indicator_columns
    from src.challenger_benchmark import chronological_split
    check_frozen()
    expected = json.loads((EVIDENCE / 'design-inspection.json').read_text())
    metadata = pd.read_csv(ROOT / 'docs/_stage4_split_assignment.csv', usecols=['TransactionID', 'TransactionDT', 'partition'])
    train_ids = metadata.loc[metadata.partition.eq('train')]
    val_ids = metadata.loc[metadata.partition.eq('validation')]
    assert set(train_ids.TransactionID).isdisjoint(val_ids.TransactionID)
    inner_ids, stop_ids = chronological_split(train_ids)
    for name, frame in [('train', train_ids), ('validation', val_ids), ('inner_fit', inner_ids), ('inner_stop', stop_ids)]:
        assert len(frame) == expected[name]['n'] and fingerprint(frame.TransactionID) == expected[name]['id_sha256']
    assert train_ids.TransactionDT.max() < val_ids.TransactionDT.min()
    artifact, _ = load_artifact(FROZEN)
    columns = list(artifact.schema.raw_columns)
    source = ROOT / 'data/raw/ieee-fraud-detection/train_transaction.csv'
    start = time.perf_counter()
    train = read_authorised_rows(source, train_ids.TransactionID, columns)
    val = read_authorised_rows(source, val_ids.TransactionID, columns)
    for frame, assigned in [(train, train_ids), (val, val_ids)]:
        times = assigned.set_index('TransactionID').TransactionDT
        assert np.array_equal(frame.TransactionDT.to_numpy(), times.loc[frame.TransactionID].to_numpy())
    fit = train.iloc[:len(inner_ids)]
    stop = train.iloc[len(inner_ids):]
    assert fit.TransactionDT.max() < stop.TransactionDT.min()
    numeric = list(artifact.schema.numeric_columns)
    cat = [c for c in columns if c not in numeric]
    missing = (fit[numeric].isna().mean()*100).round(3).sort_values(ascending=False)
    groups = select_indicator_columns(missing)
    inner = build_preprocessing_pipeline(groups['without_indicator'], groups['with_indicator'], cat)
    fit_start = time.perf_counter()
    inner.fit(fit[columns])
    inner_fit_seconds = time.perf_counter()-fit_start
    assert tuple(artifact.preprocessor.get_feature_names_out()) == tuple(artifact.schema.transformed_feature_names)
    transforms = {}
    for name, frame, transformer in [('inner_fit',fit,inner), ('inner_stop',stop,inner), ('train',train,artifact.preprocessor), ('validation',val,artifact.preprocessor)]:
        started = time.perf_counter()
        width = len(transformer.get_feature_names_out())
        matrix = open_memmap(PRIVATE / f'{name}.npy', mode='w+', dtype='float32', shape=(len(frame),width))
        for offset in range(0,len(frame),4096):
            batch = transformer.transform(frame.iloc[offset:offset+4096][columns])
            dense = batch.toarray() if hasattr(batch,'toarray') else batch
            matrix[offset:offset+len(dense)] = dense
        matrix.flush()
        np.save(PRIVATE/f'{name}-y.npy',frame.isFraud.to_numpy(dtype='int8'))
        transforms[name] = {'rows':len(frame), 'features':width, 'transform_seconds':time.perf_counter()-started}
    raw_rf = np.concatenate([predict_fraud_probability(artifact,val.iloc[i:i+4096][columns]) for i in range(0,len(val),4096)])
    np.save(PRIVATE/'rf-raw-reference.npy',raw_rf)
    np.save(PRIVATE/'validation-ids.npy',val.TransactionID.to_numpy())
    np.save(PRIVATE/'validation-times.npy',val.TransactionDT.to_numpy())
    result = {'populations':{k:expected[k] for k in ['train','inner_fit','inner_stop','validation']},
              'validation_positives':int(val.isFraud.sum()),'inner_preprocessor_fit_seconds':inner_fit_seconds,
              'transforms':transforms,'preparation_seconds':time.perf_counter()-start,
              'feature_order_sha256':hashlib.sha256('\n'.join(artifact.schema.transformed_feature_names).encode()).hexdigest(),
              'frozen_sha256':FROZEN_HASH, 'closed_evaluation_accessed':False}
    assert len(columns)==392 and transforms['train']['features']==813 and result['validation_positives']==2617
    write(EVIDENCE/'preparation.json',result)
    check_frozen()


def estimator(key, rounds, selection):
    if key == 'lgb':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective='binary',metric='binary_logloss',n_estimators=rounds,learning_rate=.05,max_depth=6,num_leaves=31,min_child_samples=50,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1,max_bin=255,random_state=42,n_jobs=10,deterministic=True,force_col_wise=True,verbosity=-1)
    if key == 'xgb':
        from xgboost import XGBClassifier
        extra = {'early_stopping_rounds':30} if selection else {}
        return XGBClassifier(booster='gbtree',objective='binary:logistic',eval_metric='logloss',tree_method='hist',device='cpu',n_estimators=rounds,learning_rate=.05,max_depth=6,min_child_weight=5,subsample=.8,colsample_bytree=.8,reg_lambda=1,reg_alpha=0,max_bin=256,random_state=42,n_jobs=10,**extra)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(loss_function='Logloss',eval_metric='Logloss',iterations=rounds,learning_rate=.05,depth=6,l2_leaf_reg=3,bootstrap_type='Bernoulli',subsample=.8,rsm=.8,boosting_type='Plain',border_count=128,random_seed=42,thread_count=10,task_type='CPU',allow_writing_files=False,verbose=False)


def native_path(key):
    return PRIVATE / {'lgb':'lightgbm.txt','xgb':'xgboost.json','cat':'catboost.cbm'}[key]


def load_native(key):
    if key=='rf':
        from src.production_pipeline import load_artifact
        return load_artifact(FROZEN)[0].model
    if key=='lgb':
        from lightgbm import Booster
        return Booster(model_file=str(native_path(key)))
    if key=='xgb':
        from xgboost import XGBClassifier
        model=XGBClassifier(n_jobs=10);model.load_model(native_path(key));return model
    from catboost import CatBoostClassifier
    model=CatBoostClassifier(thread_count=10);model.load_model(str(native_path(key)));return model


def predict(model,key,x):
    if key=='lgb':return model.predict(x,num_threads=10)
    return model.predict_proba(x)[:,1]


def fit(key, selection):
    import numpy as np
    check_frozen()
    prefix='inner_fit' if selection else 'train'
    x=np.load(PRIVATE/f'{prefix}.npy',mmap_mode='r');y=np.load(PRIVATE/f'{prefix}-y.npy')
    rounds=400 if selection else json.loads((PRIVATE/f'{key}-selection.json').read_text())['selected_rounds']
    model=estimator(key,rounds,selection)
    options={}
    if selection:
        stop=np.load(PRIVATE/'inner_stop.npy',mmap_mode='r');sy=np.load(PRIVATE/'inner_stop-y.npy')
        options['eval_set']=[(stop,sy)] if key!='cat' else (stop,sy)
        if key=='lgb':
            from lightgbm import early_stopping
            options['callbacks']=[early_stopping(30,verbose=False)]
        elif key=='cat':options.update(early_stopping_rounds=30,use_best_model=True)
        else:options['verbose']=False
    started=time.perf_counter();model.fit(x,y,**options);elapsed=time.perf_counter()-started
    if selection:
        selected=int(model.best_iteration_) if key=='lgb' else int(model.best_iteration+1) if key=='xgb' else int(model.get_best_iteration()+1)
        assert 1<=selected<=400
        result={'selected_rounds':selected,'round_ceiling_reached':selected+30>=400,'selected_round_ceiling':selected==400,'fit_seconds':elapsed,
                'configured_parameters':model.get_params(),'stopping_population':'inner_stop only','maximum_rounds':400}
        write(PRIVATE/f'{key}-selection.json',result)
    else:
        if key=='lgb':model.booster_.save_model(str(native_path(key)))
        else:model.save_model(str(native_path(key)))
        original=model.predict_proba(np.load(PRIVATE/'validation.npy',mmap_mode='r')[:256])[:,1]
        reloaded=predict(load_native(key),key,np.load(PRIVATE/'validation.npy',mmap_mode='r')[:256])
        assert np.allclose(original,reloaded,rtol=0,atol=1e-12)
        write(PRIVATE/f'{key}-final.json',{'selected_rounds':rounds,'fit_seconds':elapsed,'configured_parameters':model.get_params(),
                                         'native_reload_consistent':True,'artifact_bytes':native_path(key).stat().st_size})
    check_frozen()


def score(key):
    import numpy as np
    check_frozen()
    x=np.load(PRIVATE/'validation.npy',mmap_mode='r');model=load_native(key)
    predict(model,key,x[:4096])
    durations=[];reference=None
    for _ in range(5):
        start=time.perf_counter()
        scores=np.concatenate([predict(model,key,x[i:i+4096]) for i in range(0,len(x),4096)])
        durations.append(time.perf_counter()-start)
        assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
        if reference is None:reference=scores
        else:assert np.allclose(reference,scores,rtol=0,atol=1e-12)
    raw_difference=None
    if key=='rf':
        raw=np.load(PRIVATE/'rf-raw-reference.npy');raw_difference=float(np.max(np.abs(raw-reference)))
        assert np.allclose(raw,reference,rtol=0,atol=1e-12)
    np.save(PRIVATE/f'{key}-scores.npy',reference)
    write(PRIVATE/f'{key}-score.json',{'pass_seconds':durations,'median_seconds':float(np.median(durations)),
        'rows_per_second':float(len(x)/np.median(durations)),'batch_size':4096,'passes':5,'warmup_rows':4096,
        'rows':len(x),'validation_id_sha256':fingerprint(np.load(PRIVATE/'validation-ids.npy')),
        'native_artifact_sha256':hashlib.sha256((FROZEN if key=='rf' else native_path(key)).read_bytes()).hexdigest(),
        'repeat_consistent':True,'rf_raw_path_max_difference':raw_difference,'threads':10 if key!='rf' else 'unchanged n_jobs=-1'})
    check_frozen()


def monitored(phase,key='none'):
    import psutil
    command=[sys.executable,str(Path(__file__)),phase,'--model',key]
    log=PRIVATE/f'{key}-{phase}.log'
    started=time.perf_counter();peak=0;baseline=None
    with log.open('w') as handle:
        process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,cwd=ROOT)
        sampled=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                rss=sampled.memory_info().rss
                if baseline is None and rss>50*1024**2:baseline=rss
                peak=max(peak,rss)
                if rss>24*1024**3 or (phase in ['selection','final'] and time.perf_counter()-started>900):
                    process.kill();process.wait();raise RuntimeError(f'{key}/{phase} resource ceiling exceeded; no retuning authorised')
            except psutil.NoSuchProcess:pass
            time.sleep(.1)
    result={'phase':phase,'model':key,'wall_seconds':time.perf_counter()-started,'rss_peak_bytes':peak,
            'rss_baseline_bytes':baseline,'rss_increment_bytes':peak-(baseline or 0),'sampling_seconds':.1,'exit_code':process.returncode}
    write(EVIDENCE/f'{key}-{phase}-resources.json',result)
    if process.returncode:raise RuntimeError(f'{key}/{phase} failed; see {log}')
    print(json.dumps(result),flush=True)


def aggregate():
    import numpy as np
    from src.challenger_benchmark import benchmark_metrics,paired_block_bootstrap
    y=np.load(PRIVATE/'validation-y.npy');ids=np.load(PRIVATE/'validation-ids.npy');times=np.load(PRIVATE/'validation-times.npy')
    assert fingerprint(ids)=='14e5f38141f4ea86cf6228d76bf9af398c7cc3116d1ca49ddecc3b502e9ff5bc'
    assert len(y)==75560 and int(y.sum())==2617 and len(ids)==len(times)==len(y)
    scores={NAMES[k]:np.load(PRIVATE/f'{k}-scores.npy') for k in NAMES}
    assert all(len(s)==len(y) for s in scores.values())
    bootstrap=paired_block_bootstrap(y,scores,times)
    models=[]
    for key,name in NAMES.items():
        metrics=benchmark_metrics(y,scores[name],ids)
        training=None if key=='rf' else {p:json.loads((PRIVATE/f'{key}-{p}.json').read_text()) for p in ['selection','final']}
        if training:
            selection=training['selection']
            selection['selected_round_ceiling']=selection['selected_rounds']==400
            selection['round_ceiling_reached']=selection['selected_rounds']+30>=400
            selection['ceiling_derivation']='Best round plus patience reaches cap; full patience unavailable if greater than cap.'
        resources={p:json.loads((EVIDENCE/f'{key}-{p}-resources.json').read_text()) for p in (['score'] if key=='rf' else ['selection','final','score'])}
        models.append({'name':name,'role':'Frozen deployed reference' if key=='rf' else 'Experimental; not deployed',
            'metrics':metrics,'uncertainty':bootstrap['models'][name], 'training':training,'resources':resources,
            'scoring':json.loads((PRIVATE/f'{key}-score.json').read_text()),
            'training_seconds':None if training is None else sum(training[p]['fit_seconds'] for p in training)})
    result={'schema_version':1,'population':'Historical validation / development backtesting','rows':len(y),'positives':int(y.sum()),
        'validation_id_sha256':fingerprint(ids),'frozen_model_sha256':FROZEN_HASH,'models':models,
        'bootstrap':{k:v for k,v in bootstrap.items() if k!='models'},
        'preprocessing':json.loads((EVIDENCE/'preparation.json').read_text()),
        'strongest_observed':max(models,key=lambda m:m['metrics']['average_precision'])['name'],
        'limitations':['Reused validation: 15 Stage 7 consultations, including six decision-influencing; selection optimism remains.',
        'Shared 813-feature representation, six bounded challenger fits; native categorical alternatives and tuning deferred.',
        'Conditional block intervals do not remove selection bias, cross-block/entity dependence, temporal drift or training variability; no significance or future-performance claim.',
        'Frozen Random Forest remains deployed. No model promotion, calibration, financial-savings or final-test claim.',
        'CPU timings exclude preprocessing; sampled process RSS includes mapped pages and runtime overhead; host/platform specific. RF current training cost is unavailable, never zero.']}
    check_frozen();write(ROOT/'ui/benchmark-v1.json',result)
    write(EVIDENCE/'result-summary.json',{k:v for k,v in result.items() if k!='models'} | {'models':[{k:v for k,v in m.items() if k!='metrics'}|{'metrics':{k:v for k,v in m['metrics'].items() if k!='points'}} for m in models]})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['all','prepare','selection','final','score','aggregate']);parser.add_argument('--model',choices=[*NAMES,'none'],default='none');args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);EVIDENCE.mkdir(parents=True,exist_ok=True);check_frozen();check_environment()
    if args.phase=='all':
        monitored('prepare')
        for key in ['lgb','xgb','cat']:
            monitored('selection',key);monitored('final',key)
        for key in NAMES:monitored('score',key)
        monitored('aggregate')
    elif args.phase=='prepare':prepare()
    elif args.phase=='selection':fit(args.model,True)
    elif args.phase=='final':fit(args.model,False)
    elif args.phase=='score':score(args.model)
    else:aggregate()

if __name__=='__main__':main()
