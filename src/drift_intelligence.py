"""Stage22 label-free historical monitoring. Pure aggregate calculations, no I/O.

This module never imports a fitted model or reads a dataset. It deliberately
keeps its numerical contract separate from the approved Stage18 references.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

FROZEN_SHA256='2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04'
FEATURES={'TransactionAmt':'numeric','ProductCD':'categorical','card4':'categorical',
          'C1':'numeric','dist1':'numeric','daily_phase':'categorical'}
MIN_SAMPLE=200
EPSILON=.0001
QUANTILES=[.01,.1,.5,.9,.95,.99]
LIMITATIONS=[
 'Historical development demonstration, not live banking or customer monitoring.',
 'All windows reuse validation previously consulted during development; no independent future-performance claim.',
 'Input or score drift does not prove model deterioration. No new verified fraud labels: current precision, recall and fraud capture cannot be measured.',
 'Random Forest scores are uncalibrated ranking signals, not fraud probabilities or automated decline decisions.',
 'Three unequal-duration windows support descriptive comparisons only, not seasonal or forecast conclusions.',
 'PSI/KS/JS investigation rules are fixed demonstration heuristics, not universal significance or operational thresholds.',
 'Selected features are a diagnostic subset. Drift rank is not model importance or causal attribution.',
 'TransactionDT is relative elapsed time; daily phase is not verified local clock time.',
 'Reference-unseen categories need investigation; they are not necessarily invalid data.',
]


def fingerprint(ids):
    return hashlib.sha256('\x1f'.join(map(str,sorted(map(int,ids)))).encode()).hexdigest()


def split_windows(metadata):
    if not {'TransactionID','TransactionDT','partition'}.issubset(metadata):
        raise ValueError('Missing population metadata')
    if not metadata.partition.eq('validation').all() or metadata.TransactionID.duplicated().any():
        raise ValueError('Only unique authorised validation IDs are allowed')
    if metadata[['TransactionID','TransactionDT']].isna().any().any():
        raise ValueError('Invalid population metadata')
    frame=metadata.sort_values(['TransactionDT','TransactionID']).reset_index(drop=True)
    bounds=[0]
    for fraction in [1/3,2/3]:
        i=int(len(frame)*fraction)
        while 0<i<len(frame) and frame.TransactionDT.iloc[i]==frame.TransactionDT.iloc[i-1]: i+=1
        bounds.append(i)
    bounds.append(len(frame))
    if any(b<=a for a,b in zip(bounds,bounds[1:])):raise ValueError('Insufficient distinct chronological windows')
    return [frame.iloc[a:b].copy() for a,b in zip(bounds,bounds[1:])]


def psi(p,q):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    if p.ndim!=1 or p.shape!=q.shape or not len(p) or not np.isfinite(p).all() or not np.isfinite(q).all() or (p<0).any() or (q<0).any() or p.sum()<=0 or q.sum()<=0:
        raise ValueError('Invalid distribution')
    p=p/p.sum()+EPSILON;q=q/q.sum()+EPSILON;p/=p.sum();q/=q.sum()
    return float(np.sum((q-p)*np.log(q/p)))


def quality(values,numeric=False):
    s=pd.Series(values);missing=int(s.isna().sum());n=len(s)
    converted=pd.to_numeric(s,errors='coerce') if numeric else None
    parse=int((converted.isna()&s.notna()).sum()) if numeric else 0
    nonfinite=int((converted.notna()&~np.isfinite(converted)).sum()) if numeric else 0
    return {'n':n,'missing':missing,'missing_rate':missing/n if n else 0.,
            'parse_failures':parse,'nonfinite':nonfinite,'finite':n-missing-parse-nonfinite}


def _finite(values):
    x=pd.to_numeric(pd.Series(values),errors='coerce').to_numpy(dtype=float)
    return x[np.isfinite(x)]


def summary(values):
    x=_finite(values)
    return {'n':len(x),'mean':float(x.mean()) if len(x) else None,
            'quantiles':[float(v) for v in np.quantile(x,QUANTILES)] if len(x) else [None]*6}


def _base(reference,comparison,kind):
    a=quality(reference,kind=='numeric');b=quality(comparison,kind=='numeric')
    return {'kind':kind,'reference_quality':a,'comparison_quality':b,
            'missing_delta':b['missing_rate']-a['missing_rate']}


def numeric_comparison(reference,comparison):
    out=_base(reference,comparison,'numeric');a=_finite(reference);b=_finite(comparison)
    out.update(reference_summary=summary(a),comparison_summary=summary(b))
    if not len(a):
        edges=[];labels=[];ac=[];bc=[]
    elif np.min(a)==np.max(a):
        edges=[float(a[0])];labels=['Reference value','Other finite value']
        ac=[len(a),0];bc=[int(np.sum(b==a[0])),int(np.sum(b!=a[0]))]
    else:
        edges=np.unique(np.quantile(a,np.linspace(0,1,11)))[1:-1].tolist()
        ac=np.bincount(np.searchsorted(edges,a,side='right'),minlength=len(edges)+1).tolist()
        bc=np.bincount(np.searchsorted(edges,b,side='right'),minlength=len(edges)+1).tolist()
        labels=[f'Bin {i+1}' for i in range(len(edges)+1)]
    enough=min(len(a),len(b))>=MIN_SAMPLE
    out.update(edges=edges,labels=labels,reference_counts=ac,comparison_counts=bc,
               psi=psi(ac,bc) if enough else None,
               ks=float(ks_2samp(a,b,method='asymp').statistic) if enough else None)
    out['signal']='Insufficient evidence' if not enough else ('Investigate shift' if out['psi']>=.2 or out['ks']>=.1 else 'No material shift under selected rules')
    return out


def categorical_comparison(reference,comparison):
    out=_base(reference,comparison,'categorical')
    a=pd.Series(reference);b=pd.Series(comparison)
    vocabulary=sorted(set(a.dropna().astype(str)))
    # Array positions distinguish real category strings from the two sentinel labels.
    def counts(s):
        v=s.dropna().astype(str);known=v.value_counts()
        return [int(known.get(k,0)) for k in vocabulary]+[int(s.isna().sum()),int((~v.isin(vocabulary)).sum())]
    ac=counts(a);bc=counts(b);enough=min(len(a),len(b))>=MIN_SAMPLE
    js=None
    if enough:
        p=np.asarray(ac)/len(a);q=np.asarray(bc)/len(b);m=(p+q)/2
        def term(x):
            good=x>0
            return np.sum(x[good]*np.log2(x[good]/m[good]))
        js=float(np.clip((term(p)+term(q))/2,0,1))
    out.update(categories=vocabulary+['[Missing]','[Unseen]'],reference_counts=ac,comparison_counts=bc,
               js=js,unseen_count=bc[-1],signal='Insufficient evidence' if not enough else ('Investigate shift' if js>=.05 else 'No material shift under selected rules'))
    return out


def build_payload(frame,scores):
    if 'isFraud' in frame or not set(FEATURES).difference({'daily_phase'}).issubset(frame):
        raise ValueError('Label-free feature schema required')
    x=frame.reset_index(drop=True).copy();scores=np.asarray(scores,dtype=float)
    if len(x)!=len(scores) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():raise ValueError('Invalid scores')
    x['_score']=scores
    x['daily_phase']=((pd.to_numeric(x.TransactionDT)%86400)//3600).map(lambda h:f'{int(h):02d}')
    windows=split_windows(x);ref=windows[0];cut=float(np.quantile(ref._score,.95))
    descriptors=[]
    for i,w in enumerate(windows):
        descriptors.append({'id':f'V{i+1}','n':len(w),'start':int(w.TransactionDT.min()),'end':int(w.TransactionDT.max()),
                            'id_sha256':fingerprint(w.TransactionID),'score':summary(w._score),
                            'tail_fraction':float((w._score>cut).mean()),'duplicate_ids':int(w.TransactionID.duplicated().sum()),
                            'negative_amounts':int((pd.to_numeric(w.TransactionAmt,errors='coerce')<0).sum()),'schema_valid':True})
    comparisons=[]
    for desc,w in zip(descriptors[1:],windows[1:]):
        features={k:(numeric_comparison if kind=='numeric' else categorical_comparison)(ref[k],w[k]) for k,kind in FEATURES.items()}
        comparisons.append({'window':desc['id'],'features':features,'score':numeric_comparison(ref._score,w._score)})
    result={'schema_version':1,'source':'historical_validation','model_sha256':FROZEN_SHA256,
            'population_id_sha256':fingerprint(x.TransactionID),'reference':'V1','windows':descriptors,
            'reference_tail_threshold':cut,'reference_tail_fraction':descriptors[0]['tail_fraction'],
            'comparisons':comparisons,'limitations':LIMITATIONS,
            'method':{'minimum_sample':MIN_SAMPLE,'epsilon':EPSILON,'numeric_bins':10,'psi_rule':.2,'ks_rule':.1,'js_rule':.05,'missing_delta_rule':.05,'quantiles':QUANTILES,'tail_operator':'strictly greater than reference q95'}}
    validate_aggregate(result)
    return result


# Explicit field allowlists prohibit accidental row/label/private metadata exposure.
ROOT_FIELDS={'schema_version','source','model_sha256','population_id_sha256','reference','windows','reference_tail_threshold','reference_tail_fraction','comparisons','limitations','method'}
QFIELDS={'n','missing','missing_rate','parse_failures','nonfinite','finite'}
COMMON={'kind','reference_quality','comparison_quality','missing_delta','reference_counts','comparison_counts','signal'}
NFIELDS=COMMON|{'reference_summary','comparison_summary','edges','labels','psi','ks'}
CFIELDS=COMMON|{'categories','js','unseen_count'}
WFIELDS={'id','n','start','end','id_sha256','score','tail_fraction','duplicate_ids','negative_amounts','schema_valid'}
SIGNALS={'Insufficient evidence','Investigate shift','No material shift under selected rules'}


def validate_aggregate(a):
    """Fail closed on structural/count/range corruption; no dynamic user fields."""
    def require(ok):
        if not ok:raise ValueError('Invalid monitoring aggregate')
    def fields(v,keys):require(isinstance(v,dict) and set(v)==keys)
    def number(v,lo=None,hi=None):
        require(type(v) in (float,int) and math.isfinite(v) and (lo is None or v>=lo) and (hi is None or v<=hi))
    def integer(v):require(type(v) is int and v>=0)
    def digest(v):require(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v))
    def summ(v):
        fields(v,{'n','mean','quantiles'});integer(v['n']);require(len(v['quantiles'])==6)
        vals=[v['mean'],*v['quantiles']]
        for z in vals:
            if z is not None:number(z)
        require(all(z is None for z in vals) if not v['n'] else all(z is not None for z in vals))
        if v['n']:require(v['quantiles']==sorted(v['quantiles']))
    fields(a,ROOT_FIELDS)
    require(a['schema_version']==1 and a['source']=='historical_validation' and a['reference']=='V1' and a['model_sha256']==FROZEN_SHA256)
    digest(a['population_id_sha256']);number(a['reference_tail_threshold'],0,1);number(a['reference_tail_fraction'],0,1)
    require(a['method']=={'minimum_sample':MIN_SAMPLE,'epsilon':EPSILON,'numeric_bins':10,'psi_rule':.2,'ks_rule':.1,'js_rule':.05,'missing_delta_rule':.05,'quantiles':QUANTILES,'tail_operator':'strictly greater than reference q95'})
    require(a['limitations']==LIMITATIONS and len(a['windows'])==3 and len(a['comparisons'])==2)
    for i,w in enumerate(a['windows']):
        fields(w,WFIELDS);require(w['id']==f'V{i+1}');integer(w['n']);require(w['n']>0);number(w['start']);number(w['end']);require(w['end']>=w['start']);digest(w['id_sha256'])
        if i:require(w['start']>a['windows'][i-1]['end'])
        summ(w['score']);require(w['score']['n']==w['n']);number(w['tail_fraction'],0,1)
        for z in [w['score']['mean'],*w['score']['quantiles']]:number(z,0,1)
        for k in ['duplicate_ids','negative_amounts']:integer(w[k]);require(w[k]<=w['n'])
        require(w['schema_valid'] is True)
    def distribution(r,kind,refn,compn):
        fields(r,NFIELDS if kind=='numeric' else CFIELDS);require(r['kind']==kind and r['signal'] in SIGNALS)
        for key,n in [('reference_quality',refn),('comparison_quality',compn)]:
            q=r[key];fields(q,QFIELDS)
            for k in QFIELDS-{'missing_rate'}:integer(q[k])
            require(q['n']==n and q['finite']+q['missing']+q['parse_failures']+q['nonfinite']==n)
            number(q['missing_rate'],0,1);require(abs(q['missing_rate']-q['missing']/n)<1e-10)
        number(r['missing_delta'],-1,1);require(abs(r['missing_delta']-(r['comparison_quality']['missing_rate']-r['reference_quality']['missing_rate']))<1e-10)
        for key in ['reference_counts','comparison_counts']:
            require(isinstance(r[key],list))
            for n in r[key]:integer(n)
        require(len(r['reference_counts'])==len(r['comparison_counts']))
        if kind=='numeric':
            require(len(r['labels'])==len(r['reference_counts']) and all(isinstance(v,str) for v in r['labels']))
            for v in r['edges']:number(v)
            require(r['edges']==sorted(set(r['edges'])))
            if r['reference_quality']['finite']:
                require(len(r['edges'])==1 if r['labels']==['Reference value','Other finite value'] else len(r['reference_counts'])==len(r['edges'])+1)
            else:require(not r['edges'] and not r['labels'])
            for k in ['reference','comparison']:
                summ(r[k+'_summary']);require(r[k+'_summary']['n']==r[k+'_quality']['finite'])
                # No-reference histogram can be absent, but no hidden comparison counts are permitted otherwise.
                require(sum(r[k+'_counts'])==r[k+'_quality']['finite'] if r['reference_quality']['finite'] else not r[k+'_counts'])
            enough=min(r['reference_quality']['finite'],r['comparison_quality']['finite'])>=MIN_SAMPLE
            for k,hi in [('psi',None),('ks',1)]:
                require((r[k] is not None)==enough)
                if r[k] is not None:number(r[k],0,hi)
            expected='Insufficient evidence' if not enough else ('Investigate shift' if r['psi']>=.2 or r['ks']>=.1 else 'No material shift under selected rules')
        else:
            require(len(r['categories'])==len(r['reference_counts']) and r['categories'][-2:]==['[Missing]','[Unseen]'] and all(isinstance(v,str) for v in r['categories']))
            require(sum(r['reference_counts'])==refn and sum(r['comparison_counts'])==compn)
            integer(r['unseen_count']);require(r['unseen_count']==r['comparison_counts'][-1])
            enough=min(refn,compn)>=MIN_SAMPLE;require((r['js'] is not None)==enough)
            if r['js'] is not None:number(r['js'],0,1)
            expected='Insufficient evidence' if not enough else ('Investigate shift' if r['js']>=.05 else 'No material shift under selected rules')
        require(r['signal']==expected)
    for i,c in enumerate(a['comparisons']):
        fields(c,{'window','features','score'});require(c['window']==f'V{i+2}');fields(c['features'],set(FEATURES))
        for name,kind in FEATURES.items():distribution(c['features'][name],kind,a['windows'][0]['n'],a['windows'][i+1]['n'])
        distribution(c['score'],'numeric',a['windows'][0]['n'],a['windows'][i+1]['n'])
        require(c['score']['reference_summary']==a['windows'][0]['score'] and c['score']['comparison_summary']==a['windows'][i+1]['score'])
        for label in ['reference_summary','comparison_summary']:
            for z in [c['score'][label]['mean'],*c['score'][label]['quantiles']]:number(z,0,1)
        for z in c['score']['edges']:number(z,0,1)
    require(a['reference_tail_fraction']==a['windows'][0]['tail_fraction'] and a['reference_tail_threshold']==a['windows'][0]['score']['quantiles'][4])
    return a
