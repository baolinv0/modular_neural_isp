"""No leakage/target shortcuts in the real-content relative diagnostic."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

D=Path(__file__).parents[1]/"diagnostics"
sys.path.insert(0,str(D))
import dng_relative_experiment as exp


def test_group_balanced_training_prior_excludes_diagnostic(tmp_path):
    records=[]
    for i,(group,split,value) in enumerate([
        ("a","train",.1),("a","train",.1),("b","train",.8),
        ("c","diagnostic",.4)]):
        source=f"s{i}"
        np.savez(tmp_path/f"{source}.npz",green=np.full((1,2,2),value))
        records.append({"source_id":source,"camera":"iphone","split":split,"group_id":group})
    prior,groups=exp.build_prior(tmp_path,records,"iphone",bins=10,pseudomass=0)
    assert groups==["a","b"]
    assert prior[2]==pytest.approx(.5)
    assert prior[9]==pytest.approx(.5)
    assert np.count_nonzero(prior)==2


def test_continuous_prior_weights_include_endpoint_atoms():
    prior=np.array([.1,.2,.3,.4])
    x,w=exp.prior_nodes(prior,4)
    assert w.sum()==pytest.approx(1)
    assert w[0]==pytest.approx(.1)
    assert w[-1]==pytest.approx(.4)
    assert x[0]==0 and x[-1]==1
    assert np.all((x[1:-1]>0)&(x[1:-1]<1))


def test_no_spatial_window_crosses_repeats_or_patch_boundaries():
    codes=np.zeros((2,2,8,8),dtype=np.int16)
    codes[1,0]=2000
    codes[0,1]=4000
    result=exp.wiener_target(codes,1,7)
    assert np.all(result[0,0]==0)
    assert np.all(result[1,1]==0)
    assert np.all(result[1,0]>0)
    assert np.all(result[0,1]>result[1,0])


def test_policy_action_uses_only_frozen_scalar_feature():
    features=np.zeros((4,len(exp.FEATURE_NAMES)))
    features[:,0]=[.1,.2,.3,.4]
    rule={"feature":"mean_brightness","threshold":.25,"left":1,"right":5}
    actions=exp.policy_actions(features,rule)
    np.testing.assert_array_equal(actions,[1,1,5,5])
