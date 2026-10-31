"""Label-free monitoring contracts; fixtures are explicitly synthetic."""
import copy
import math
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from src.drift_intelligence import (
    psi, numeric_comparison, categorical_comparison, split_windows,
    build_payload, validate_aggregate, FEATURES,
)


def test_psi_independent_formula_and_zero_bins():
    a=np.array([1.,0.,0.]);b=np.array([.5,.25,.25]);p=(a+.0001)/(a+.0001).sum();q=(b+.0001)/(b+.0001).sum()
    assert psi(a,b)==pytest.approx(sum((q-p)*np.log(q/p)))
    assert psi(a,a)==0
    with pytest.raises(ValueError):psi([0,0],[1,0])


def test_numeric_ks_and_open_ended_bins():
    a=pd.Series(np.arange(1000,dtype=float));b=a+500
    r=numeric_comparison(a,b)
    assert r['ks']==pytest.approx(ks_2samp(a,b).statistic)
    assert sum(r['comparison_counts'])==1000
    assert r['comparison_counts'][-1]>r['reference_counts'][-1]
    assert r['signal']=='Investigate shift'


def test_numeric_missing_invalid_and_insufficient():
    a=pd.Series([1.]*300+[np.nan]*100);b=pd.Series([2.]*200+[np.inf]*20+[None]*180)
    r=numeric_comparison(a,b)
    assert r['reference_quality']['missing']==100
    assert r['comparison_quality']['nonfinite']==20
    assert r['comparison_quality']['missing']==180
    assert r['missing_delta']==pytest.approx(.2)
    assert sum(r['comparison_counts'])==200
    small=numeric_comparison(pd.Series([None]*210),pd.Series([1]*300))
    assert small['psi'] is None and small['ks'] is None
    assert small['signal']=='Insufficient evidence'
    assert numeric_comparison(pd.Series([1]*300),pd.Series([1]*300))['psi']==0


def test_categorical_common_vocabulary_and_independent_js():
    a=pd.Series(['visa']*200+[None]*100);b=pd.Series(['mastercard']*200+[None]*100)
    r=categorical_comparison(a,b)
    assert r['categories']==['visa','[Missing]','[Unseen]']
    assert r['unseen_count']==200
    assert r['js']==pytest.approx(jensenshannon([2/3,1/3,0],[0,1/3,2/3],base=2)**2)
    assert 0<=r['js']<=1
    assert categorical_comparison(a.iloc[:2],b)['js'] is None


def test_windows_ties_order_nonoverlap_and_failures():
    m=pd.DataFrame({'TransactionID':range(12),'TransactionDT':[1,2,3,4,4,4,5,6,7,8,9,10],'partition':['validation']*12})
    w=split_windows(m)
    assert len(w[0])==6
    assert all(a.TransactionDT.max()<b.TransactionDT.min() for a,b in zip(w,w[1:]))
    assert len(set().union(*(set(x.TransactionID) for x in w)))==12
    with pytest.raises(ValueError):split_windows(m.assign(partition='train'))
    with pytest.raises(ValueError):split_windows(m.assign(TransactionID=1))
    with pytest.raises(ValueError):split_windows(m.assign(TransactionDT=1))


def fixture_frame():
    n=900
    f=pd.DataFrame({'TransactionID':np.arange(n),'TransactionDT':np.arange(n)*100,'partition':['validation']*n})
    for name,kind in FEATURES.items():
        if name=='daily_phase':continue
        f[name]=np.arange(n,dtype=float) if kind=='numeric' else ['W']*n
    return f


def test_aggregate_reproducible_counts_no_labels_or_rows():
    f=fixture_frame();scores=np.linspace(0,1,len(f))
    a=build_payload(f,scores);b=build_payload(f,scores)
    assert a==b
    validate_aggregate(a)
    assert len(a['windows'])==3 and len(a['comparisons'])==2
    assert a['source']=='historical_validation'
    assert 'isFraud' not in str(a) and 'TransactionID' not in str(a)
    assert a['reference_tail_fraction']<.06
    for c in a['comparisons']:
        for r in c['features'].values():
            assert 0<=r['reference_quality']['missing_rate']<=1
    with pytest.raises(ValueError):build_payload(f,scores+2)
    with pytest.raises(ValueError):build_payload(f.drop(columns='card4'),scores)
    with pytest.raises(ValueError):build_payload(f.assign(isFraud=0),scores)


@pytest.mark.parametrize('mutation',[
    lambda a:a.update(source='synthetic'),
    lambda a:a['windows'][1].update(start=a['windows'][0]['end']),
    lambda a:a['comparisons'][0]['features']['TransactionAmt'].update(psi=-1),
    lambda a:a['comparisons'][0]['features']['ProductCD'].update(js=2),
    lambda a:a.update(records=[{'note':'private'}]),
    lambda a:a['comparisons'][0]['features']['TransactionAmt']['comparison_counts'].__setitem__(0,-1),
])
def test_invalid_aggregate_rejected(mutation):
    a=build_payload(fixture_frame(),np.linspace(0,1,900));mutation(a)
    with pytest.raises(ValueError):validate_aggregate(a)


def test_authorised_reader_skips_excluded_values_and_forbids_outcomes(tmp_path):
    from scripts.build_monitoring_aggregate import read_validation_features
    p=tmp_path/'source.csv'
    p.write_text('TransactionID,TransactionDT,TransactionAmt,isFraud\n1,10,4,0\n2,20,DO_NOT_PARSE,1\n3,30,8,0\n')
    m=pd.DataFrame({'TransactionID':[1,3],'TransactionDT':[10,30],'partition':['validation']*2})
    a=read_validation_features(p,m,['TransactionAmt'])
    assert a.TransactionAmt.tolist()==[4,8] and 'isFraud' not in a
    with pytest.raises(ValueError):read_validation_features(p,m,['isFraud'])
    with pytest.raises(ValueError):read_validation_features(p,m.assign(partition='sealed'),['TransactionAmt'])


def test_aggregate_population_and_score_quantiles_match_independently():
    f=fixture_frame();s=np.linspace(0,1,len(f));a=build_payload(f,s)
    assert a['windows'][0]['score']['quantiles']==pytest.approx(np.quantile(s[:300],[.01,.1,.5,.9,.95,.99]))
    assert a['windows'][2]['tail_fraction']==1
    assert a['comparisons'][0]['score']['ks']==1
    bad=copy.deepcopy(a);bad['comparisons'][0]['features']['unsupported']=bad['comparisons'][0]['features']['C1']
    with pytest.raises(ValueError):validate_aggregate(bad)


def test_numeric_parse_failures_and_missing_are_distinct():
    r=numeric_comparison(pd.Series([1.]*300),pd.Series([1.]*300+['bad',None,np.inf]))
    q=r['comparison_quality']
    assert (q['parse_failures'],q['missing'],q['nonfinite'],q['finite'])==(1,1,1,300)
    assert r['psi']==0 and r['comparison_summary']['n']==300


@pytest.mark.parametrize('mutation',[
    lambda a:a['windows'][0]['score'].update(mean=2),
    lambda a:a['comparisons'][0]['score']['comparison_summary']['quantiles'].__setitem__(5,2),
    lambda a:a['comparisons'][0]['score']['reference_summary'].update(mean=.9),
    lambda a:a['comparisons'][0]['features']['TransactionAmt']['edges'].append(999999),
])
def test_invalid_score_bounds_alignment_and_bin_contract(mutation):
    a=build_payload(fixture_frame(),np.linspace(0,1,900));mutation(a)
    with pytest.raises(ValueError):validate_aggregate(a)


def test_tail_consistency_and_historical_synthetic_separation():
    a=build_payload(fixture_frame(),np.linspace(0,1,900))
    a['reference_tail_fraction']=.99
    with pytest.raises(ValueError):validate_aggregate(a)
    a=build_payload(fixture_frame(),np.linspace(0,1,900));a['source']='synthetic'
    with pytest.raises(ValueError):validate_aggregate(a)
