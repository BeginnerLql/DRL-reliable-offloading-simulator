"""Diagnostic-only upload instrumentation and augmented-state integrity tests."""
from types import SimpleNamespace
import numpy as np
import pytest
import simpy
import torch
from diagnostics.state_sufficiency_audit import (
    OUT, InFlightAuditEnvironmentState, instrumented_upload_duration,
    snapshot_decision_state, make_variants, set_up_agent, collect_replay_episode, model_hash,
)
from diagnostics.run_state_sufficiency_audit import trajectory_regression
from diagnostics.state_sufficiency_audit import build_value_model, target_from_saved_value_gae, params
from diagnostics.value_gae_identifiability import run_episode, make_agent


def _fixture_state():
    env=SimpleNamespace(servers={},audit_inflight={},tasks={},spatial_risk_field=np.linspace(-.2,.2,8))
    env.get_server_backlog_time=lambda sid,now: float(sid)/10
    env.get_active_failure_rate=lambda sid: .001*sid
    for sid in range(1,9):
        server=SimpleNamespace(server_id=sid,processing_frequency=10.+sid)
        env.servers[sid]={'server_object':server,'running_replica':None,'waiting_replicas':[]}
    task=SimpleNamespace(id=1,env=simpy.Environment(),input_data_size_mb=2.,computation_demand=20.,reliability_requirement=.99)
    reliabilities=np.linspace(.97,.9999,28);safe=reliabilities>=.99
    effective=safe.copy()
    decision={'state':np.linspace(0,1,35,dtype=np.float32),'reliabilities':reliabilities,
        'safe_mask':safe,'effective_mask':effective,'safe_set_empty':False,
        'best_achievable_reliability':float(reliabilities.max()),'requirement':.99}
    return env,task,decision


def test_inflight_upload_enters_cpu_queue_without_changing_duration():
    env=InFlightAuditEnvironmentState();env.servers[1]={'server_object':SimpleNamespace(server_id=1),
        'waiting_replicas':[],'running_replica':None}
    task_env=simpy.Environment();server=SimpleNamespace(server_id=1,processing_frequency=10.0)
    task=SimpleNamespace(id=1,env=task_env,env_state=env,primaryNode=server,backupNode=SimpleNamespace(server_id=2),
                         input_data_size_mb=1.0,computation_demand=40.0)
    duration,download=instrumented_upload_duration(task,server,lambda *_:(2.0,0.0))
    assert (duration,download)==(2.0,0.0)
    assert len(env.audit_inflight)==1
    task_env.run(until=2.0)
    env.register_waiting_replica(1,task,'primary',4.0)
    assert not env.audit_inflight
    assert len(env.audit_upload_history)==1
    assert env.audit_upload_history[0]['cpu_service_time']==4.0
    assert len(env.servers[1]['waiting_replicas'])==1


def test_no_future_information_is_used_by_decision_snapshot():
    env,task,decision=_fixture_state()
    before=snapshot_decision_state(task,env,decision)
    env.future_arrivals=[object()];env.future_rewards=[1e9];env.future_spatial_draws=[-999]
    after=snapshot_decision_state(task,env,decision)
    for key in ['flight_features','reliability_features','state1','state2','state3','state4',
                'cpu_backlog_seconds','pair_reliability','effective_mask']:
        assert np.array_equal(before[key],after[key])


def test_effective_and_safe_mask_features_preserve_decision_support():
    env,task,decision=_fixture_state();snapshot=snapshot_decision_state(task,env,decision)
    s1=snapshot['state1']
    assert np.array_equal(s1[35+28:35+56],decision['effective_mask'])
    assert np.array_equal(s1[35+56:35+84],decision['safe_mask'])
    assert np.array_equal(snapshot['effective_mask'],decision['effective_mask'])
    assert snapshot['reliability_summary'][0]==decision['safe_mask'].sum()


def test_state_variant_dimensions_and_no_current_state_mutation():
    env,task,decision=_fixture_state();snapshot=snapshot_decision_state(task,env,decision)
    arrays={
        'states':np.stack([decision['state']]*2),
        'reliability_features':np.stack([snapshot['reliability_features']]*2),
        'flight_features':np.stack([snapshot['flight_features']]*2),
        'cpu_population_features':np.stack([snapshot['cpu_population_features']]*2),
        'effective_failure_rates':np.stack([snapshot['effective_failure_rates']]*2),
        'spatial_risk_field':np.stack([snapshot['spatial_risk_field']]*2),
    }
    variants=make_variants(arrays)
    assert {k:v.shape[1] for k,v in variants.items()}=={'S0':35,'S1':126,'S2':75,'S3':166,'S4':223}
    assert np.array_equal(variants['S0'][0],decision['state'])


def test_episode_split_reuses_parent_frozen_split():
    import json
    from diagnostics.value_gae_identifiability import split_episodes
    parent=json.loads((OUT.parent/'value_gae_identifiability/frozen_policy_dataset_manifest.json').read_text())
    assert parent['split']==split_episodes()
    assert set(parent['split']['train']).isdisjoint(parent['split']['test'])
    assert parent['matched_count']==1000


@pytest.fixture(scope='module')
def short_frozen_regression():
    torch.set_num_threads(1)
    seeds={'arrival_seed':2468,'spatial_seed':1357,'action_seed':2026}
    baseline,_=make_agent();before=model_hash(baseline.policy_old)
    ref,ref_loop=run_episode(baseline,seeds,details=True,max_tasks=8)
    ref_delays={int(row[1]):float(row[22]) for row in ref_loop.task_Assignments_info}
    ref['task_delay']=np.asarray([ref_delays[int(tid)] for tid in ref['task_ids']])
    telemetry,_=set_up_agent();before_telemetry=model_hash(telemetry.policy_old)
    observed,loop=collect_replay_episode(telemetry,seeds,max_tasks=8)
    return before,before_telemetry,ref,observed,loop


def test_frozen_trajectory_and_actor_checkpoint_replay_exact(short_frozen_regression):
    before,before_telemetry,ref,observed,loop=short_frozen_regression
    check=trajectory_regression(ref,observed,1)
    assert check['action_mismatch']==0 and check['reward_max_abs_diff']==0
    assert check['delay_max_abs_diff']==0 and check['mask_mismatch']==0
    assert before==before_telemetry==model_hash(loop.model.policy_old)
    assert len(loop.env_state.audit_upload_history)==16


def test_upload_telemetry_has_no_terminal_inflight_records(short_frozen_regression):
    *_,loop=short_frozen_regression
    assert not loop.env_state.audit_inflight
    assert len(loop.env_state.audit_upload_history)==16
    assert all(np.isclose(r['cpu_queue_entry_time'],r['upload_start_time']+r['expected_upload_duration'],atol=1e-8,rtol=0)
               for r in loop.env_state.audit_upload_history)


def test_augmented_state_value_model_fits_finite_targets():
    torch.manual_seed(7)
    model=build_value_model(126,seed=7)
    x=torch.randn(12,126)
    target=torch.linspace(-1,1,12)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    for _ in range(3):
        prediction=model(x);loss=torch.nn.functional.mse_loss(prediction,target)
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    assert torch.isfinite(model(x)).all()


def test_matched_state_gae_recomputation_uses_variant_value_predictions():
    episode=np.asarray([9,9,9])
    data={'episode':episode,'rewards':np.asarray([1.,-.25,2.]),
          'delta_t':np.asarray([.5,1.25,.75]),'done':np.asarray([False,False,True])}
    value=np.asarray([.2,.4,.1],dtype=float)
    next_value=np.asarray([.4,.1,0.],dtype=float)
    recomputed=target_from_saved_value_gae(value,next_value,data,9)
    expected=__import__('diagnostics.value_gae_identifiability',fromlist=['production_gae']).production_gae(
        data['rewards']*params.reward_scale_ppo,data['delta_t'],data['done'],value,next_value,
        params.gamma_ppo,params.gae_lambda_ppo)[3]
    assert np.allclose(recomputed,expected,rtol=0,atol=0)
    alternative=target_from_saved_value_gae(value+np.asarray([.1,0.,-.1]),next_value,data,9)
    assert not np.array_equal(recomputed,alternative)
