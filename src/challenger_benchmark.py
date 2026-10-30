"""Stage 20 development-only comparison utilities. No production model fitting."""
from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from src.capacity_simulator import build_capacity_points


def chronological_split(frame, fraction=.85):
    if not 0 < fraction < 1 or len(frame)<2:
        raise ValueError('A nonempty chronological fit/stop split is required')
    frame=frame.sort_values(['TransactionDT','TransactionID']).reset_index(drop=True)
    cut=max(1,min(len(frame)-1,int(len(frame)*fraction)))
    while cut<len(frame) and frame.TransactionDT.iloc[cut]==frame.TransactionDT.iloc[cut-1]:cut+=1
    if cut==len(frame):raise ValueError('Timestamp ties leave no later stopping partition')
    return frame.iloc[:cut],frame.iloc[cut:]


def chronological_blocks(times, count=20):
    times=np.asarray(times)
    if times.ndim!=1 or len(times)<count or (np.diff(times)<0).any() or count<2:
        raise ValueError('Sorted times and at least two nonempty blocks required')
    cuts=[0]
    for i in range(1,count):
        cut=len(times)*i//count
        while cut<len(times) and times[cut]==times[cut-1]:cut+=1
        if cut>cuts[-1] and cut<len(times):cuts.append(cut)
    cuts.append(len(times))
    return [np.arange(a,b) for a,b in zip(cuts,cuts[1:])]


def benchmark_metrics(labels,scores,ids):
    y=np.asarray(labels);scores=np.round(np.asarray(scores,dtype=float),12);ids=np.asarray(ids)
    n=len(y);ks=sorted({math.ceil(n*i/200) for i in range(201)}|{math.ceil(n*.01),math.ceil(n*.05)})
    points=build_capacity_points(y,scores,ids,ks)
    order=sorted(range(n),key=lambda i:(-scores[i],ids[i]))
    for p in points:
        selected=np.zeros(n,dtype=int);selected[order[:p['k']]]=1
        tn,fp,fn,tp=confusion_matrix(y,selected,labels=[0,1]).ravel()
        if [int(tp),int(fp),int(fn),int(tn)]!=[p[x] for x in ['tp','fp','fn','tn']]:
            raise AssertionError('Independent confusion calculation failed')
        expected_precision = float(tp/(tp+fp)) if tp+fp else None
        expected_recall = float(tp/(tp+fn)) if tp+fn else None
        if p['precision'] != expected_precision or p['recall'] != expected_recall:
            raise AssertionError('Independent precision/recall calculation failed')
    if any(a['tp']>b['tp'] for a,b in zip(points,points[1:])):
        raise AssertionError('Nonmonotonic historical capture')
    precision,recall,_=precision_recall_curve(y,scores)
    return {'average_precision':float(average_precision_score(y,scores)),
            'roc_auc':float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None,
            'trapezoidal_pr_auc':float(auc(recall,precision)),
            'top1':next(p for p in points if p['k']==math.ceil(n*.01)),
            'top5':next(p for p in points if p['k']==math.ceil(n*.05)),
            'points':points,'independent_checkpoint_checks':len(points)}


def weighted_ap_sorted(labels,scores,weights):
    """Exact weighted AP from fixed score groups; equivalent to repeated rows."""
    order=np.argsort(-scores,kind='stable');s=scores[order];y=labels[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    return weighted_ap_prepared(y,order,starts,weights)


def weighted_ap_prepared(sorted_y,order,starts,weights):
    w=weights[order]
    positives=np.add.reduceat(w*sorted_y,starts)
    totals=np.cumsum(np.add.reduceat(w,starts));tp=np.cumsum(positives)
    if tp[-1]==0:return None
    precision=np.divide(tp,totals,out=np.zeros_like(tp,dtype=float),where=totals>0)
    return float(np.sum(positives*precision)/tp[-1])


def paired_block_bootstrap(labels,scores_by_model,times,draws=1000,block_count=20,min_valid=900):
    y=np.asarray(labels);blocks=chronological_blocks(times,block_count)
    block_ids=np.empty(len(y),dtype=int)
    for i,block in enumerate(blocks):block_ids[block]=i
    prepared={}
    for name,score in scores_by_model.items():
        score=np.round(np.asarray(score),12)
        if len(score)!=len(y):raise ValueError('Models must share identical eligible rows')
        order=np.argsort(-score,kind='stable');s=score[order]
        prepared[name]=(y[order],order,np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1])
    rng=np.random.default_rng(42);samples={name:[] for name in prepared}
    for _ in range(draws):
        counts=np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks));w=counts[block_ids]
        positives=float(np.dot(w,y));total=int(w.sum())
        if positives==0 or positives==total:continue
        for name,args in prepared.items():samples[name].append(weighted_ap_prepared(*args,w))
    valid=len(samples['Random Forest']);models={}
    for name,values in samples.items():
        diff=np.asarray(values)-np.asarray(samples['Random Forest'])
        models[name]={'ap_ci':np.quantile(values,[.025,.975]).tolist() if valid>=min_valid else None,
                      'ap_difference_ci':np.quantile(diff,[.025,.975]).tolist() if valid>=min_valid else None}
    return {'method':'paired chronological-block bootstrap; row multiplicities represented as exactly equivalent sample weights',
            'draws':draws,'valid_draws':valid,'degenerate_draws':draws-valid,'block_count':len(blocks),'seed':42,
            'interpretation':'Exploratory fixed-model historical intervals; not independent future-performance confidence or multiplicity-adjusted significance.',
            'models':models}
