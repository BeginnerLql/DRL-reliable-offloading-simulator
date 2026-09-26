import numpy as np
import pytest
import torch
from agents.ppo_agent import PPOPairScoringPolicyNetwork, PPOContextPairScoringPolicyNetwork
from agents.context_masked_pair_ppo_agent import ContextMaskedPairPPOAgent, event_interval_rewards


def net():
    return PPOContextPairScoringPolicyNetwork(35, 28, [16, 8], 8, np.zeros(28))


def test_context_pair_preserves_backlog_cpu_association():
    n = net()
    a = torch.zeros(35); a[15] = 1; a[23] = 1/3
    b = a.clone(); b[16], b[23] = a[23].clone(), a[16].clone()
    old = PPOPairScoringPolicyNetwork(35,28,[16,8],8,np.zeros(28))
    assert torch.equal(old.build_pair_features(a)[6], old.build_pair_features(b)[6])
    assert not torch.equal(n.build_pair_features(a)[6], n.build_pair_features(b)[6])


def test_context_scores_symmetric_complete_node_swap():
    n = net(); f = n.build_pair_features(torch.randn(3,35))
    swapped = torch.cat((f[...,4:8], f[...,:4], f[...,8:]), dim=-1)
    torch.testing.assert_close(n.score_pair_features(f), n.score_pair_features(swapped))
    assert n(torch.randn(35)).shape == (28,)
    assert n(torch.randn(3,35)).shape == (3,28)


def test_unselected_server_context_can_change_relative_pair_scores():
    n = net()
    # Explicit representable scorer: pair node frequency interacting with global backlog.
    class Interaction(torch.nn.Module):
        def forward(self, f): return (f[...,1] * f[...,12+16+7]).unsqueeze(-1)
    n.scorer = Interaction()
    a = torch.zeros(35); a[8:16] = torch.arange(8)/8
    b = a.clone(); b[23] = .5
    assert (n(a)[0]-n(a)[13]).item() != (n(b)[0]-n(b)[13]).item()


def test_event_credit_includes_earlier_task_outcome_and_conserves_return():
    times = [0.,1.,3.]
    outcomes = [(1,10.,2.),(3,4.,5.),(2,6.,3.)]
    r = event_interval_rewards(times,5.,outcomes,.9)
    np.testing.assert_allclose(r,[0.,9.,6.+.9**2*4.])
    assert np.dot(.9**np.array(times),r) == pytest.approx(sum(.9**t*v for _,v,t in outcomes))


def test_event_credit_zero_duration_and_terminal_boundary():
    np.testing.assert_allclose(event_interval_rewards([0.,1.,1.],2.,[(1,2.,1.),(2,3.,2.)],1.),[0.,0.,5.])


@pytest.mark.parametrize('outcomes', [[(1,1.,1.),(1,2.,2.)],[(1,1.,4.)],[(1,float('nan'),1.)]])
def test_event_credit_rejects_invalid_outcomes(outcomes):
    with pytest.raises(ValueError): event_interval_rewards([0.,1.],3.,outcomes,.9)


def test_independent_clipping_does_not_rescale_small_actor_gradient():
    a = ContextMaskedPairPPOAgent(num_states=35,num_actions=28,hidden_layers=[8],num_servers=8,pair_correlations=np.zeros(28))
    actor = list(a.policy_net.parameters()); critic = list(a.value_net.parameters())
    for p in actor: p.grad = torch.zeros_like(p)
    for p in critic: p.grad = torch.full_like(p,1000)
    actor[0].grad.flatten()[0] = .1
    a._clip_gradients()
    assert actor[0].grad.flatten()[0].item() == pytest.approx(.1)
    assert torch.sqrt(sum((p.grad**2).sum() for p in critic)) <= .50001


def test_origin_ablation_is_bitwise_legacy_training(capsys):
    from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent
    from config.params import params
    from core.main_loop import MainLoop
    from Project_main import build_pair_correlations
    from tools.paired_ppo_experiment import _agent_kwargs, scoped_environment_seeds
    _, rho = build_pair_correlations()
    agents=[]; logs=[]
    for legacy in (True,False):
        torch.manual_seed(941)
        kwargs=_agent_kwargs('pair_scoring',np.asarray(rho),992)
        a=(ReliabilityMaskedPairPPOAgent(**kwargs) if legacy else
           ContextMaskedPairPPOAgent(**kwargs,credit_mode='origin_task',gradient_clipping='joint'))
        torch.manual_seed(1234)
        with scoped_environment_seeds(891,892):
            loop=MainLoop(a,2,20,params.num_states,params.num_actions); loop.EP()
        agents.append(a); logs.append(loop.task_Assignments_info)
    assert logs[0] == logs[1]
    for name in ('policy_net','policy_old','value_net'):
        for key,value in getattr(agents[0],name).state_dict().items():
            assert torch.equal(value,getattr(agents[1],name).state_dict()[key])


def test_event_agent_real_loop_training_preserves_task_logs_and_masks(capsys):
    from core.main_loop import MainLoop
    from config.params import params
    from Project_main import build_pair_correlations
    from tools.paired_ppo_experiment import _agent_kwargs
    _,rho=build_pair_correlations()
    kwargs=_agent_kwargs('pair_scoring',np.asarray(rho),99); kwargs['actor_mode']='pair_context'
    a=ContextMaskedPairPPOAgent(**kwargs)
    before={k:v.clone() for k,v in a.policy_net.state_dict().items()}
    loop=MainLoop(a,1,20,params.num_states,params.num_actions); loop.EP()
    assert len(loop.task_Assignments_info)==20
    assert a.credit_diagnostics[0]['task_reward_sum']==pytest.approx(loop.ep_reward_list[0])
    assert a.credit_diagnostics[0]['return_residual'] < 1e-10
    assert a.update_diagnostics[0]['first_minibatch_ratio_max_abs_error']<1e-5
    assert all(r['safe_set_empty'] or r['selected_action_safe'] for r in a.selection_archive)
    assert any(not torch.equal(before[k],v) for k,v in a.policy_net.state_dict().items())
    assert not a.outcome_records and not a.decision_times


def test_prior_task_is_affected_by_later_upload_overtaking(tmp_path):
    import pandas as pd
    import simpy
    from core.task import Task
    from core.server import Server
    from core.env_state import EnvironmentState
    from core.main_loop import MainLoop
    path=tmp_path/'tasks.xlsx'
    pd.DataFrame({'Task_ID':[1,2],'Input_Data_Size_MB':[2.,.5],
                  'Computation_Demand':[20.,50.],'Reliability_Requirement':[.9,.9]}).to_excel(path,index=False)
    def replay(overlap):
        env=simpy.Environment(); st=EnvironmentState()
        servers=[Server(env,'Edge',i+1,f,.001,-37.8,144.9,u)
                 for i,(f,u) in enumerate(zip([10,11,12,14],[16,20,32,40]))]
        for s in servers: st.add_server_and_init_environment(s)
        old=Task(env,st,1,params_file=path);st.add_task(old)
        old.initialize_reliability_evaluation(*servers[:2]);env.process(old.execute_task(*servers[:2]))
        def later():
            yield env.timeout(.1)
            new=Task(env,st,2,params_file=path);st.add_task(new)
            chosen=servers[:2] if overlap else servers[2:]
            new.initialize_reliability_evaluation(*chosen)
            yield env.process(new.execute_task(*chosen))
        env.process(later());env.run()
        loop=MainLoop.__new__(MainLoop);loop.env_state=st
        return loop.calcReward(1)
    fast,slow=replay(False),replay(True)
    assert fast[1]==pytest.approx(2.618181818181818)
    assert slow[1]==pytest.approx(6.663636363636364)
    assert fast[0]>slow[0]
