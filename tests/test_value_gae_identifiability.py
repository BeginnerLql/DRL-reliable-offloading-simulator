"""Offline mathematical checks plus a short real frozen-policy replay test."""
import numpy as np
import pytest
import torch
from diagnostics.value_gae_identifiability import (
    ROOT,CHECKPOINT,smdp_mc,production_gae,split_episodes,matched_indices,
    train_normalization,normalize_target,inverse_target,model_hash,saturation,
    make_agent,run_episode,replay_checks,
)


def test_mc_smdp_uses_elapsed_time_not_step_count():
    r=np.array([1.,2.,3.]);dt=np.array([.5,2.,9.]);d=[False,False,True]
    result=smdp_mc(r,dt,d,.9)
    assert result[-1]==3
    assert result[1]==pytest.approx(2+.9**2*3)
    assert result[0]==pytest.approx(1+.9**.5*(2+.9**2*3))


def test_episode_boundary_blocks_cross_episode_return():
    result=smdp_mc([1,2,1000],[1,1,1],[False,True,True],.9)
    assert result.tolist()==[2.8,2.,1000.]
    with pytest.raises(ValueError):smdp_mc([1],[1],[False],.9)


def test_split_has_no_episode_overlap_and_matched_states_are_test_only():
    split=split_episodes()
    assert [len(split[k]) for k in ('train','validation','test')]==[180,60,60]
    assert set(split['train']).isdisjoint(split['test'])
    assert set(split['train']).isdisjoint(split['validation'])
    assert set(split['test']).isdisjoint(split['validation'])
    assert set(sum(split.values(),[]))==set(range(1,301))
    matched=matched_indices(split)
    assert len(matched)==1000 and matched.state_id.nunique()==1000
    assert set(matched.episode)<=set(split['test'])
    assert matched.equals(matched_indices(split))


def test_normalization_uses_only_train_statistics():
    mu,sd=train_normalization([1.,3.,1e9],[1,1,2],[1])
    assert (mu,sd)==(2.,1.)
    mu2,sd2=train_normalization([1.,3.,-1e12],[1,1,2],[1])
    assert (mu,sd)==(mu2,sd2)


def test_target_inverse_transform_round_trip():
    y=np.array([-50.,100.,1000.])
    assert np.allclose(inverse_target(normalize_target(y,42.,17.),42.,17.),y)


def test_gae_matches_independent_recurrence_and_terminal_mask():
    r=np.array([2.,3.,5.]);dt=np.array([.2,1.5,10.]);d=np.array([False,False,True])
    v=np.array([7.,8.,9.]);nv=np.array([8.,9.,999.])
    td,a,n,ret=production_gae(r,dt,d,v,nv,.9,.95)
    expected_delta=r+.9**dt*nv*(~d)-v
    expected=np.zeros(3)
    for i in range(2,-1,-1):expected[i]=expected_delta[i]+(.9**dt[i]*.95*expected[i+1] if i<2 else 0)
    assert np.allclose(td,expected_delta,atol=1e-6)
    assert np.allclose(a,expected,atol=2e-6)
    assert np.allclose(ret,a+v,atol=1e-6)
    assert abs(n.mean())<1e-6 and abs(n.std()-1)<1e-6


def test_huber_fraction_is_residual_threshold_not_active_loss():
    result=saturation([2.,.5,0.],[0.,0.,0.])
    assert result['hypothetical_huber_saturation_fraction']==pytest.approx(1/3)
    assert result['absolute_residual_p50']==.5


def test_model_hash_detects_parameter_change():
    net=torch.nn.Linear(2,1);before=model_hash(net)
    with torch.no_grad():net.weight.add_(1.)
    assert model_hash(net)!=before


@pytest.fixture(scope='module')
def replay_fixture():
    if not (CHECKPOINT/'actor.pt').exists() or not (ROOT/'data/server_info.xlsx').exists():
        pytest.skip('Local formal checkpoint/data unavailable')
    torch.set_num_threads(1);agent,_=make_agent();before=model_hash(agent.policy_old)
    seeds=dict(arrival_seed=456,spatial_seed=789,action_seed=123)
    ref,_=run_episode(agent,seeds,capture_ids=range(1,9),max_tasks=8)
    target=next((i+1 for i,m in enumerate(ref['effective_mask']) if m.sum()>1),1)
    choices=np.flatnonzero(ref['effective_mask'][target-1]);action=int(choices[-1])
    branch,_=run_episode(agent,seeds,details=False,capture_ids=[target],forced=(target,action),cache=ref,max_tasks=8)
    return agent,before,ref,branch,target,dict(agent.snapshots[target])


def test_frozen_policy_hash_unchanged_during_real_rollout(replay_fixture):
    agent,before,*_=replay_fixture
    assert model_hash(agent.policy_old)==before
    assert not agent.update_diagnostics
    assert all(not p.requires_grad for p in agent.policy_old.parameters())


def test_matched_prefix_replay_is_identical(replay_fixture):
    _,_,ref,branch,target,snapshot=replay_fixture
    assert not any(replay_checks(snapshot,ref,target).values())
    assert np.array_equal(branch['actions'][:target-1],ref['actions'][:target-1])


def test_counterfactual_branch_preserves_common_randomness(replay_fixture):
    _,_,ref,branch,_,_=replay_fixture
    assert np.array_equal(ref['arrival_trace'],branch['arrival_trace'])
    assert np.array_equal(ref['rng_trace'],branch['rng_trace'])
