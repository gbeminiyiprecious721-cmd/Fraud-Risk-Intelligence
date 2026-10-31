"""Independent synthetic scientific contracts; never fits/reads historical data."""
import numpy as np
import pytest
from types import SimpleNamespace
from src.deployed_explainability import group_lineage, group_signed, rankings, leaf_baseline, reconcile, rank_stability


def test_lineage_covers_every_index_and_groups_signed_before_absolute():
 names=['numeric_plain__A','numeric_indicator__B','numeric_indicator__missingindicator_B','categorical__cat_x','categorical__cat_y']
 groups=group_lineage(names,['A','B','cat'],['cat'])
 values=np.array([[.1,.3,-.2,.4,-.4],[-.1,-.3,.2,-.2,.3]])
 grouped=group_signed(values,groups)
 assert grouped.shape==(2,3)
 assert grouped[:,1]==pytest.approx([.1,-.1])
 assert grouped[:,2]==pytest.approx([0,.1])
 assert np.sum(grouped,axis=1)==pytest.approx(values.sum(axis=1))
 assert rankings(np.abs(grouped).mean(axis=0),groups)[0]['name']=='A'
 with pytest.raises(ValueError):group_lineage(['bad__X'],['A'],[])
 with pytest.raises(ValueError):group_lineage(names,['A','B'],['cat'])


def test_leaf_baseline_uses_weighted_leaf_probabilities_not_mean_of_leaves():
 tree=SimpleNamespace(children_left=np.array([1,-1,-1]),weighted_n_node_samples=np.array([10,9,1]),value=np.array([[[.9,.1]],[[1,0]],[[0,1]]]))
 model=SimpleNamespace(classes_=np.array([0,1]),estimators_=[SimpleNamespace(tree_=tree,classes_=np.array([0.,1.]))])
 assert leaf_baseline(model)==pytest.approx(.1)
 tree.weighted_n_node_samples[0]=0
 with pytest.raises(ValueError):leaf_baseline(model)


def test_reconcile_rejects_wrong_scale_shape_and_nonfinite():
 assert reconcile(np.array([[.2,-.1]]),.3,np.array([.4]))['passed']
 for v,b,s in [(np.array([[.2]]),.3,np.array([.7])),(np.array([[np.nan]]),.3,np.array([.4])),(np.array([[.2]]),.3,np.array([2]))]:
  with pytest.raises(ValueError):reconcile(v,b,s)


def test_rank_stability_deterministic_and_insufficient():
 names=[f'f{i}' for i in range(12)];groups={n:{'indices':[i]} for i,n in enumerate(names)}
 a=np.tile(np.arange(12),(512,1)).astype(float)
 s=rank_stability(a,groups,np.arange(512),np.arange(512))
 assert s['status']=='available' and s['overlap_median']==1
 assert s==rank_stability(a,groups,np.arange(512),np.arange(512))
 assert rank_stability(a,groups,np.ones(512),np.arange(512))['status']=='insufficient'


def test_grouping_mdi_and_mapping_order_are_not_interchangeable():
    names=['numeric_plain__B','numeric_plain__A']
    groups=group_lineage(names,['A','B'],[])
    assert group_signed(np.array([[.2,.1]]),groups)[0].tolist()==pytest.approx([.1,.2])
    with pytest.raises(ValueError):group_signed(np.array([[1,2,3]]),groups)
    with pytest.raises(ValueError):rankings(np.array([-.1,.2]),groups)


def test_shap_exact_single_feature_identity_on_explicit_synthetic_tree():
    # A disposable toy estimator validates SHAP's stated score-space identity;
    # it never reads or alters the production artifact.
    import shap
    from sklearn.tree import DecisionTreeClassifier
    model=DecisionTreeClassifier(max_depth=1,random_state=42).fit([[0],[0],[1],[1]],[0,0,1,1])
    explainer=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent',model_output='raw')
    x=np.array([[0.],[1.]])
    r=explainer(x)
    assert r.values.shape==(2,1,2)
    assert r.base_values[:,1]==pytest.approx([.5,.5])
    assert r.values[:,0,1]==pytest.approx([-.5,.5])
    assert reconcile(r.values[:,:,1],r.base_values[:,1],model.predict_proba(x)[:,1])['passed']
