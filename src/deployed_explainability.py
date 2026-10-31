"""Offline-only scientific helpers for the frozen artifact; no fitting or data I/O."""
from __future__ import annotations
import numpy as np
from src.explainability import build_feature_lineage


def group_lineage(names,raw,categorical):
    frame=build_feature_lineage(list(names),list(categorical))
    if frame.original_column.isna().any() or set(frame.original_column)!=set(raw) or len(set(names))!=len(names):
        raise ValueError('Incomplete or ambiguous feature lineage')
    groups={name:{'indices':[],'numeric':0,'missingness_indicator':0,'categorical_onehot':0} for name in sorted(raw)}
    for i,row in frame.iterrows():
        if row.feature_type not in groups[row.original_column]:raise ValueError('Unsupported transform')
        g=groups[row.original_column];g['indices'].append(int(i));g[row.feature_type]+=1
    if sorted(i for g in groups.values() for i in g['indices'])!=list(range(len(names))):raise ValueError('Invalid coverage')
    return groups


def group_signed(values,groups):
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Invalid contributions')
    indices=[i for g in groups.values() for i in g['indices']]
    if sorted(indices)!=list(range(values.shape[1])):raise ValueError('Invalid grouping')
    return np.column_stack([values[:,g['indices']].sum(axis=1) for g in groups.values()])


def rankings(values,groups,limit=None):
    values=np.asarray(values,dtype=float)
    if len(values)!=len(groups) or not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid importance')
    result=[{'name':name,'value':float(value),'components':{k:v for k,v in groups[name].items() if k!='indices'}} for name,value in zip(groups,values)]
    return sorted(result,key=lambda r:(-r['value'],r['name']))[:limit]


def leaf_baseline(model):
    if list(model.classes_)!=[0,1]:raise ValueError('Unsupported class mapping')
    bases=[]
    for estimator in model.estimators_:
        if list(estimator.classes_)!=[0,1]:raise ValueError('Unsupported tree classes')
        t=estimator.tree_;leaves=t.children_left==-1;root=t.weighted_n_node_samples[0];weights=t.weighted_n_node_samples[leaves]
        if root<=0 or not np.isclose(weights.sum(),root,rtol=1e-12,atol=1e-8):raise ValueError('Invalid path weights')
        values=t.value[leaves,0,:];totals=values.sum(axis=1)
        if (totals<=0).any():raise ValueError('Invalid leaf values')
        bases.append(float(np.dot(weights/root,values[:,1]/totals)))
    if not bases:raise ValueError('No fitted trees')
    return float(np.mean(bases))


def reconcile(values,baseline,scores,tolerance=1e-8):
    values=np.asarray(values,dtype=float);scores=np.asarray(scores,dtype=float)
    base=np.asarray(baseline,dtype=float)
    if values.ndim!=2 or scores.shape!=(len(values),) or not np.isfinite(values).all() or not np.isfinite(scores).all() or not np.isfinite(base).all() or ((scores<0)|(scores>1)).any() or ((base<0)|(base>1)).any():raise ValueError('Invalid reconciliation inputs')
    errors=np.abs(values.sum(axis=1)+base-scores)
    if not len(errors) or errors.max()>tolerance:raise ValueError('Contributions do not reconcile')
    return {'passed':True,'max_abs_error':float(errors.max()),'tolerance':tolerance,'rows':len(values)}


def rank_stability(values,groups,times,ids):
    values=np.asarray(values,dtype=float);order=np.lexsort((ids,times));values=values[order];times=np.asarray(times)[order];n=len(values)
    bounds=[0]
    for i in range(1,16):
        k=i*n//16
        while 0<k<n and times[k]==times[k-1]:k+=1
        bounds.append(k)
    bounds.append(n)
    if any(b<=a for a,b in zip(bounds,bounds[1:])):return {'status':'insufficient','reason':'Cannot form16 nonempty tie-safe temporal blocks'}
    blocks=[values[a:b] for a,b in zip(bounds,bounds[1:])]
    names=list(groups);base=rankings(np.abs(values).mean(axis=0),groups);top=[r['name'] for r in base[:20]];k=min(10,len(names));reference={r['name'] for r in base[:k]}
    rng=np.random.default_rng(20231023);overlaps=[];ranks={name:[] for name in top}
    for _ in range(200):
        rows=np.concatenate([blocks[i] for i in rng.integers(0,16,size=16)],axis=0)
        ranked=rankings(np.abs(rows).mean(axis=0),groups);lookup={r['name']:i+1 for i,r in enumerate(ranked)}
        overlaps.append(len(reference&{r['name'] for r in ranked[:k]})/k)
        for name in top:ranks[name].append(lookup[name])
    return {'status':'available','seed':20231023,'replicates':200,'blocks':16,'row_weighted':True,
            'overlap_min':float(min(overlaps)),'overlap_median':float(np.median(overlaps)),'overlap_max':float(max(overlaps)),
            'rank_spread':[{'name':name,'quantiles':[float(x) for x in np.quantile(ranks[name],[0,.1,.5,.9,1])]} for name in top]}
