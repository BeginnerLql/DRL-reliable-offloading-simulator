import numpy as np
import pytest
import torch

from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent
from tools.run_origin_vs_event_credit_10seed import (
    new_agent, legacy_agent, check_formal, digest_state, zero_runs,
)
from diagnostics.run_masked_pair_ppo_10seed import formal_spec
from Project_main import build_pair_correlations


def test_arms_have_identical_historical_actor_critic_and_masks():
    _, plan = formal_spec()
    trial = {k:int(v) for k,v in plan.iloc[0].items()}
    _, rho = build_pair_correlations()
    initial = check_formal(trial, rho)
    old = legacy_agent(trial, rho)
    origin = new_agent(trial, rho, 'origin_task')
    event = new_agent(trial, rho, 'event_interval')
    assert isinstance(old, ReliabilityMaskedPairPPOAgent)
    assert origin.actor_mode == event.actor_mode == 'pair_scoring'
    assert origin.gradient_clipping == event.gradient_clipping == 'joint'
    assert origin.gamma == event.gamma == old.gamma
    assert origin.gae_lambda == event.gae_lambda == old.gae_lambda
    assert digest_state(origin.policy_net.state_dict()) == initial['actor_initial_sha']
    assert digest_state(event.policy_net.state_dict()) == initial['actor_initial_sha']
    assert digest_state(origin.value_net.state_dict()) == initial['critic_initial_sha']
    assert digest_state(event.value_net.state_dict()) == initial['critic_initial_sha']
    for network in ('policy_net', 'policy_old', 'value_net'):
        a, b = getattr(origin, network), getattr(event, network)
        for key, value in a.state_dict().items():
            assert torch.equal(value, b.state_dict()[key])
    assert origin.pairs == event.pairs == old.pairs
    np.testing.assert_array_equal(origin.pair_correlations, event.pair_correlations)


def test_zero_runs_and_joint_clipping_matches_historical_operation():
    assert zero_runs([1,0,0,2,0]) == [2,1]
    assert zero_runs([1,2]) == [0]
    _,plan=formal_spec();trial={k:int(v) for k,v in plan.iloc[0].items()}
    _,rho=build_pair_correlations()
    old=legacy_agent(trial,rho); new=new_agent(trial,rho,'origin_task')
    for agent in (old,new):
        for index,p in enumerate(list(agent.policy_net.parameters())+list(agent.value_net.parameters())):
            p.grad=torch.full_like(p,100.0+index)
        agent._clip_gradients()
    for old_p,new_p in zip(list(old.policy_net.parameters())+list(old.value_net.parameters()),
                           list(new.policy_net.parameters())+list(new.value_net.parameters())):
        assert torch.equal(old_p.grad,new_p.grad)


def test_gradient_minibatches_are_aggregated_once_per_episode(tmp_path, monkeypatch):
    import pandas as pd
    from types import SimpleNamespace
    import tools.run_origin_vs_event_credit_10seed as runner
    monkeypatch.setattr(runner, 'TRAIN_EPISODES', 1)
    row = {'episode':1}
    agent = SimpleNamespace(
        credit_rows=[row], gae_rows=[row], critic_rows=[row],
        gradient_rows=[{'episode':1,'actor_norm_before':1.,'critic_norm_before':10.},
                       {'episode':1,'actor_norm_before':3.,'critic_norm_before':30.}],
        reassignment_rows=[row], assignment_rows=[row], credit_diagnostics=[{**row, "return_residual":0.}],
        update_diagnostics=[{'first_minibatch_ratio_max_abs_error':0., 'ratio_min':1.}],
    )
    runner.save_training_diagnostics(agent,tmp_path,'event')
    gradient=pd.read_csv(tmp_path/'gradient_statistics_by_episode.csv')
    assert len(gradient)==1
    assert gradient.loc[0,'minibatches']==2
    assert gradient.loc[0,'actor_norm_before']==2.
    assert gradient.loc[0,'critic_norm_before']==20.


def test_temporal_shift_reconstruction_uses_actual_first_finish(monkeypatch):
    import pandas as pd
    import tools.run_origin_vs_event_credit_10seed as runner
    from config.params import params
    monkeypatch.setattr(runner,'TASKS',2)
    rng=np.random.default_rng(123)
    t1=float(rng.exponential(scale=1/params.TASK_ARRIVAL_RATE))
    t2=t1+float(rng.exponential(scale=1/params.TASK_ARRIVAL_RATE))
    tasks=pd.DataFrame({'episode':[1,1],'task_id':[1,2],
                        'Primary_End':[t2+.5,t2+1.],
                        'Backup_End':[t2+.7,t2+1.2]})
    result=runner.reconstruct_temporal_shifts(tasks,123,0,'origin')
    assert result.index_shift.tolist()==[1,0]
    assert result.resolution_delay.tolist()==pytest.approx([t2+.5-t1,1.])


def test_direct_decision_trace_gate_checks_state_and_delta_t(tmp_path, monkeypatch):
    import json
    import tools.run_origin_vs_event_credit_10seed as runner
    monkeypatch.setattr(runner, 'TRAIN_EPISODES', 1)
    monkeypatch.setattr(runner, 'TASKS', 20)
    gate=tmp_path/'gate'
    gate.mkdir()
    (gate/'gate_report.json').write_text(json.dumps({'exact':True}))
    result=runner.run_trace_gate(tmp_path)
    assert result['decisions_checked']==20
    assert result['exact']
    assert result['states_mismatches']==0
    assert result['delta_t_mismatches']==0
    assert result['old_log_probability_mismatches']==0
