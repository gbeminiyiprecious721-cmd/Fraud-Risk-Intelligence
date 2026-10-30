import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.challenger_benchmark import chronological_split, chronological_blocks, benchmark_metrics, paired_block_bootstrap


def test_inner_split_advances_past_timestamp_tie_and_preserves_ids():
    frame = pd.DataFrame({'TransactionID': range(10), 'TransactionDT': [1,2,3,4,5,6,7,8,8,9]})
    fit, stop = chronological_split(frame, .85)
    assert len(fit)==9 and len(stop)==1
    assert fit.TransactionDT.max() < stop.TransactionDT.min()
    assert set(fit.TransactionID).isdisjoint(stop.TransactionID)


def test_metrics_and_capacity_denominators_are_independently_correct():
    y=np.array([1,0,1,0,1]);score=np.array([.8,.9,.7,.2,.1]);ids=np.arange(5)
    result=benchmark_metrics(y,score,ids)
    assert result['average_precision']==pytest.approx(average_precision_score(y,score))
    assert result['roc_auc']==pytest.approx(roc_auc_score(y,score))
    assert result['top1']['k']==1 and result['top1']['tp']==0
    assert result['points'][-1]['tp']==3 and result['points'][-1]['recall']==1
    for p in result['points']:
        chosen=sorted(range(5),key=lambda i:(-score[i],ids[i]))[:p['k']]
        assert p['tp']==sum(y[i] for i in chosen)
        assert p['tp']+p['fp']==p['k']
        assert p['tp']+p['fn']==3


def test_blocks_keep_ties_and_paired_identical_models_have_zero_difference():
    times=np.repeat(np.arange(10),2)
    blocks=chronological_blocks(times,4)
    assert np.array_equal(np.concatenate(blocks),np.arange(20))
    for a,b in zip(blocks,blocks[1:]):assert times[a[-1]] < times[b[0]]
    y=np.tile([0,1],10);scores=np.arange(20)/20
    result=paired_block_bootstrap(y,{'Random Forest':scores,'Identical':scores},times,draws=40,block_count=4,min_valid=35)
    assert result['valid_draws']==40
    assert result['models']['Identical']['ap_difference_ci']==[0.,0.]


def test_degenerate_bootstrap_withholds_uncertainty_instead_of_inventing_it():
    result=paired_block_bootstrap(np.zeros(20),{'Random Forest':np.arange(20)/20},np.arange(20),draws=20,block_count=4,min_valid=18)
    assert result['valid_draws']==0 and result['models']['Random Forest']['ap_ci'] is None


@pytest.mark.parametrize('fraction', [0,1,-.1,1.1])
def test_invalid_inner_split_is_rejected(fraction):
    with pytest.raises(ValueError):chronological_split(pd.DataFrame({'TransactionID':[1,2],'TransactionDT':[1,2]}),fraction)


def test_weighted_ap_matches_independent_sklearn_with_tied_scores():
    from src.challenger_benchmark import weighted_ap_sorted
    y=np.array([1,0,1,0,1]);s=np.array([.5,.5,.9,.1,.1]);w=np.array([2,0,1,3,2])
    assert weighted_ap_sorted(y,s,w)==pytest.approx(average_precision_score(y,s,sample_weight=w))


def test_reader_skips_disallowed_values_before_numeric_parsing(tmp_path):
    from scripts.run_stage20_benchmark import read_authorised_rows
    source=tmp_path/'transactions.csv'
    source.write_text('TransactionID,TransactionDT,isFraud,value\n1,10,0,2.5\n2,20,CLOSED,DO_NOT_PARSE\n3,30,1,3.5\n')
    frame=read_authorised_rows(source,[1,3],['value'])
    assert frame.TransactionID.tolist()==[1,3]
    assert frame.isFraud.tolist()==[0,1] and frame.value.dtype.kind=='f'


def test_weighted_bootstrap_matches_literal_repeated_rows():
    from src.challenger_benchmark import weighted_ap_sorted
    rng=np.random.default_rng(42)
    for _ in range(20):
        y=rng.integers(0,2,40);score=rng.integers(0,6,40)/6;weights=rng.integers(0,4,40)
        assert weighted_ap_sorted(y,score,weights)==pytest.approx(average_precision_score(np.repeat(y,weights),np.repeat(score,weights)))


def test_library_nan_sentinel_is_documented_without_invalid_json(tmp_path):
    import json
    from scripts.run_stage20_benchmark import write
    path=tmp_path/'params.json'
    write(path,{'missing':float('nan'),'metric':.5})
    assert json.loads(path.read_text())=={'missing':'nan','metric':.5}
