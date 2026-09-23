"""Single-seed formal-parameter smoke: legacy Pair PPO vs new masked Pair PPO.

The new policy deploys by masked stochastic sampling. Masked greedy is an
explicit ablation. No existing runner's greedy default is reused for it.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from unittest.mock import patch

import core.main_loop as main_loop_module
from agents.masked_pair_ppo_agent import (
    ReliabilityMaskedPairPPOAgent, effective_action_mask, production_reliability_vector,
)
from config.masked_pair_ppo import MASKED_PAIR_PPO
from config.params import params
from core.main_loop import MainLoop
from Project_main import build_pair_correlations
from diagnostics.evaluate_reliability_masked_policy import (
    ArrivalTraceLoop, TrackingEnvironmentState, summarize_server_load,
)
from tools.pair_policy_diagnostics import DiagnosticPPOAgent, TASK_ASSIGNMENT_COLUMNS
from tools.paired_ppo_experiment import (
    FrozenEvaluationPPOAgent, _agent_kwargs, _state_dict_snapshot,
    assert_state_dict_unchanged, scoped_environment_seeds, sha256_file,
)

PRIOR=ROOT/'diagnostics/results/policy_oracle_alignment'
FROZEN_MASKS=ROOT/'diagnostics/results/reliability_masked_policy/action_diagnostics_masked_sample.csv'
DEFAULT_OUT=ROOT/'diagnostics/results/masked_pair_ppo_smoke'


class BaselineGreedyAgent(FrozenEvaluationPPOAgent):
    """Audit baseline greedy decisions; leave its selection method unchanged."""
    def select_action(self,state,epsilon=0.0,use_softmax=False,temperature=1.5):
        action=super().select_action(state,epsilon,use_softmax,temperature)
        task,env_state,episode=self.current_context
        reliability=production_reliability_vector(task,env_state,self.pairs)
        safe,effective,empty,best=effective_action_mask(reliability,task.reliability_requirement)
        with torch.no_grad():
            logits=self.policy_old(self._to_tensor(state).unsqueeze(0)).squeeze(0)
            distribution=Categorical(logits=logits)
        self.selection_archive.append({
            'episode':episode,'task_id':int(task.id),'action_index':action,
            'R_req':float(task.reliability_requirement),'safe_set_size':int(safe.sum()),
            'safe_set_empty':bool(empty),'safe_mask':safe.astype(int).tolist(),
            'effective_mask':effective.astype(int).tolist(),
            'best_achievable_reliability':best,
            'reliability_deficit':max(float(task.reliability_requirement)-best,0.0),
            'masked_entropy':float(distribution.entropy().item()),
            'selected_pair_reliability':float(reliability[action]),
            'selected_rho':float(self.pair_correlations[action]),
            'selected_action_safe':bool(safe[action]),
        })
        return action


def check_external_trace(candidate,reference):
    arrivals_a,risk_a=candidate
    arrivals_b,risk_b=reference
    if not np.array_equal(arrivals_a,arrivals_b) or not risk_a.equals(risk_b):
        raise RuntimeError('Matched experiments received different arrival/spatial-risk streams')


def trace(loop):
    risk=pd.DataFrame(loop.episode_spatial_risk_log).sort_values(['episode','server_id']).reset_index(drop=True)
    return np.asarray(loop.interarrival_trace),risk


def persist_assignments(loop,output,name):
    frame=pd.DataFrame(loop.task_Assignments_info,columns=TASK_ASSIGNMENT_COLUMNS)
    frame=frame.sort_values(['episode','task_id']).reset_index(drop=True)
    frame.to_csv(output/f'{name}_task_assignments.csv',index=False)
    return frame


def persist_decisions(agent,assignments,output,name):
    frame=pd.DataFrame(agent.selection_archive).sort_values(['episode','task_id']).reset_index(drop=True)
    if len(frame)!=len(assignments): raise RuntimeError('Decision/task log length mismatch')
    if not np.array_equal(frame.action_index,assignments.action_index):
        raise RuntimeError('Selected actions disagree with simulator outcome log')
    if not np.allclose(frame.selected_pair_reliability,assignments.Execution_Reliability,atol=1e-12,rtol=0):
        raise RuntimeError('Mask-side and simulator-side reliability disagree')
    if not np.array_equal(frame.selected_action_safe.astype(bool),assignments.Reliability_Satisfied.astype(bool)):
        raise RuntimeError('Mask-side and simulator-side feasibility disagree')
    for column in ('safe_mask','effective_mask'):
        frame[column]=frame[column].map(json.dumps)
    frame.to_csv(output/f'{name}_decisions.csv',index=False)
    return frame


def episode_metrics(assignments,decisions):
    joined=assignments[['episode','task_id','Reliability_Satisfied','Task_Reward','Task_Delay','action_index']].merge(
        decisions[['episode','task_id','safe_set_size','safe_set_empty','masked_entropy']],
        on=['episode','task_id'],validate='one_to_one')
    rows=[]
    for episode,group in joined.groupby('episode'):
        feasible=~group.safe_set_empty.astype(bool)
        satisfied=group.Reliability_Satisfied.astype(bool)
        counts=group.action_index.value_counts().to_numpy(dtype=float)/len(group)
        rows.append({'episode':int(episode),'tasks':len(group),
            'mean_reward':float(group.Task_Reward.mean()),
            'mean_latency':float(group.Task_Delay.mean()),
            'mean_safe_set_size':float(group.safe_set_size.mean()),
            'empty_safe_set_rate':float((~feasible).mean()),
            'feasibility_rate':float(feasible.mean()),
            'conditional_rsr':float(satisfied[feasible].mean()) if feasible.any() else np.nan,
            'overall_rsr':float(satisfied.mean()),
            'avoidable_violation_count':int((feasible & ~satisfied).sum()),
            'unavoidable_violation_count':int((~feasible & ~satisfied).sum()),
            'mean_masked_entropy':float(group.masked_entropy.mean()),
            'pair_78_frequency':float((group.action_index==params.num_actions-1).mean()),
            'pair_selection_hhi':float(np.sum(counts**2))})
    return pd.DataFrame(rows)


def train_one(agent,name,trial,episodes,tasks,output):
    torch.manual_seed(int(trial['Train_Action_Seed']))
    with scoped_environment_seeds(int(trial['Train_Arrival_Seed']),int(trial['Train_Spatial_Seed'])):
        with (output/f'{name}_training.log').open('w') as log,redirect_stdout(log):
            loop=ArrivalTraceLoop(agent,episodes,tasks,params.num_states,params.num_actions)
            loop.EP()
    assignments=persist_assignments(loop,output,f'{name}_train')
    if isinstance(agent,ReliabilityMaskedPairPPOAgent):
        decisions=persist_decisions(agent,assignments,output,f'{name}_train')
        episode_metrics(assignments,decisions).to_csv(output/f'{name}_training_episode_metrics.csv',index=False)
        pd.DataFrame(agent.update_diagnostics).to_csv(output/f'{name}_updates.csv',index=False)
    curve=pd.DataFrame({'episode':range(1,episodes+1),'episode_reward':loop.ep_reward_list,
                        'episode_total_delay':loop.ep_delay_list})
    curve.to_csv(output/f'{name}_training_curve.csv',index=False)
    return loop,assignments


def evaluate_one(trained,name,trial,episodes,tasks,pairs,output,mode='stochastic'):
    if isinstance(trained,ReliabilityMaskedPairPPOAgent):
        agent=copy.deepcopy(trained)
        agent.frozen=True
        agent.deployment_mode=mode
        agent.clear_rollout()
        agent.selection_archive=[]
    else:
        agent=BaselineGreedyAgent.from_trained(trained)
        agent.pairs=pairs
        agent.selection_archive=[]
    before=_state_dict_snapshot(agent)
    TrackingEnvironmentState.instances=[]
    TrackingEnvironmentState.active_agent=agent
    torch.manual_seed(int(trial.get('Eval_Action_Seed', 2026)))
    with patch.object(main_loop_module,'EnvironmentState',TrackingEnvironmentState):
        with scoped_environment_seeds(int(trial['Eval_Arrival_Seed']),int(trial['Eval_Spatial_Seed'])):
            with (output/f'{name}_evaluation.log').open('w') as log,redirect_stdout(log):
                loop=ArrivalTraceLoop(agent,episodes,tasks,params.num_states,params.num_actions)
                loop.EP()
    assert_state_dict_unchanged(before,agent)
    for state in TrackingEnvironmentState.instances:state.finalize()
    assignments=persist_assignments(loop,output,f'{name}_eval')
    decisions=persist_decisions(agent,assignments,output,f'{name}_eval')
    episode_metrics(assignments,decisions).to_csv(output/f'{name}_evaluation_episode_metrics.csv',index=False)
    load=summarize_server_load(TrackingEnvironmentState.instances,assignments,name)
    load.to_csv(output/f'{name}_server_load.csv',index=False)
    return loop,assignments,decisions,load


def summary(assignments,decisions,name):
    n=len(assignments)
    counts=assignments.action_index.value_counts().reindex(range(params.num_actions),fill_value=0).to_numpy(dtype=float)
    probabilities=counts/n
    positive=probabilities[probabilities>0]
    safe=~decisions.safe_set_empty.astype(bool)
    satisfied=assignments.Reliability_Satisfied.astype(bool)
    high=np.isclose(assignments.Reliability_Requirement,.9999)
    return {'policy':name,'tasks':n,
            'mean_reward':float(assignments.Task_Reward.mean()),
            'mean_latency':float(assignments.Task_Delay.mean()),
            'overall_rsr':float(satisfied.mean()),
            'highest_rsr':float(satisfied[high].mean()),
            'feasibility_rate':float(safe.mean()),
            'empty_safe_set_rate':float((~safe).mean()),
            'conditional_rsr':float(satisfied[safe].mean()) if safe.any() else np.nan,
            'avoidable_violation_count':int((safe & ~satisfied).sum()),
            'unavoidable_violation_count':int((~safe & ~satisfied).sum()),
            'mean_safe_set_size':float(decisions.safe_set_size.mean()),
            'pair_78_frequency':float(probabilities[-1]),
            'pair_selection_hhi':float(np.sum(probabilities**2)),
            'pair_selection_entropy':float(-np.sum(positive*np.log(positive))),
            'unique_pair_count':int(np.count_nonzero(counts))}


def compare_fixed_masks(decisions,minimum_states=1000):
    reference=pd.read_csv(FROZEN_MASKS).sort_values(['episode','task_id']).reset_index(drop=True)
    current=decisions.sort_values(['episode','task_id']).reset_index(drop=True)
    merged=current.merge(reference[['episode','task_id','safe_mask']],on=['episode','task_id'],
                         suffixes=('_new','_frozen'),validate='one_to_one')
    sample=merged.head(minimum_states)
    if len(sample)<minimum_states:raise RuntimeError('Not enough matched evaluation states for mask check')
    mismatch=sum(json.loads(a)!=json.loads(b) for a,b in zip(sample.safe_mask_new,sample.safe_mask_frozen))
    return {'states_compared':len(sample),'safe_mask_mismatch_count':int(mismatch)}


def run(train_episodes=MASKED_PAIR_PPO.smoke_train_episodes,
        eval_episodes=MASKED_PAIR_PPO.smoke_eval_episodes,
        tasks=MASKED_PAIR_PPO.tasks_per_episode,output=DEFAULT_OUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if train_episodes<100 or train_episodes>300:
        raise ValueError('Formal smoke requires 100–300 training episodes')
    if tasks!=200:raise ValueError('Formal smoke requires 200 tasks per episode')
    prior=json.loads((PRIOR/'run_metadata.json').read_text())
    trial=prior['formal_seed_row']
    pairs,rho=build_pair_correlations();pairs=list(pairs);rho=np.asarray(rho,dtype=float)
    if params.num_actions!=28 or params.num_states!=35 or str(params.model_summary).lower()!='ppo':
        raise RuntimeError('Formal 8-server Pair PPO configuration is required')
    kwargs=_agent_kwargs('pair_scoring',rho,int(trial['PPO_Minibatch_Seed']))
    torch.manual_seed(int(trial['Torch_Init_Seed']))
    legacy=DiagnosticPPOAgent(**kwargs)
    torch.manual_seed(int(trial['Torch_Init_Seed']))
    masked=ReliabilityMaskedPairPPOAgent(**kwargs)
    for module in ('policy_net','policy_old','value_net'):
        for key,value in getattr(legacy,module).state_dict().items():
            if not torch.equal(value,getattr(masked,module).state_dict()[key]):
                raise RuntimeError('Legacy and masked agents did not start from identical parameters')
    legacy_train,_=train_one(legacy,'legacy_pair',trial,train_episodes,tasks,output)
    legacy_curve=pd.read_csv(output/'legacy_pair_training_curve.csv')
    prior_curve=pd.read_csv(PRIOR/'training_curve.csv').head(train_episodes)
    if not np.array_equal(legacy_curve.episode_reward.to_numpy(),prior_curve.Episode_Reward.to_numpy()):
        raise RuntimeError('Legacy Pair PPO training differs from the saved 300-episode baseline')
    masked_train,_=train_one(masked,'masked_pair',trial,train_episodes,tasks,output)
    check_external_trace(trace(masked_train),trace(legacy_train))
    torch.save(masked.policy_old.state_dict(),output/'masked_pair_actor.pt')
    torch.save(masked.value_net.state_dict(),output/'masked_pair_critic.pt')
    torch.save(legacy.policy_old.state_dict(),output/'legacy_pair_actor.pt')
    evaluated=[];reference=None;mask_check=None
    for name,agent,mode in (('legacy_pair_greedy',legacy,'greedy'),
                             ('masked_pair_stochastic',masked,'stochastic'),
                             ('masked_pair_greedy',masked,'greedy')):
        loop,assignments,decisions,load=evaluate_one(agent,name,trial,eval_episodes,tasks,pairs,output,mode)
        if reference is None:reference=trace(loop)
        else:check_external_trace(trace(loop),reference)
        evaluated.append(summary(assignments,decisions,name))
        if name=='masked_pair_stochastic':mask_check=compare_fixed_masks(decisions)
    result=pd.DataFrame(evaluated)
    result.to_csv(output/'smoke_policy_summary.csv',index=False)
    if mask_check['safe_mask_mismatch_count']!=0:raise RuntimeError('New agent safe mask differs from frozen diagnostic')
    updates=pd.DataFrame(masked.update_diagnostics)
    if len(updates)!=train_episodes or not np.isfinite(updates.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError('Masked PPO update diagnostics missing or non-finite')
    metadata={'train_episodes':train_episodes,'eval_episodes':eval_episodes,'tasks_per_episode':tasks,
              'masked_deployment_default':'stochastic','masked_greedy_ablation_only':True,
              'external_arrival_spatial_streams_identical':True,
              'mask_comparison':mask_check,
              'initial_actor_critic_tensors_identical':True,
              'legacy_training_matches_prior_formal_prefix':True,
              'masked_actor_sha256':sha256_file(output/'masked_pair_actor.pt'),
              'seed_row':trial}
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-episodes',type=int,default=MASKED_PAIR_PPO.smoke_train_episodes)
    parser.add_argument('--eval-episodes',type=int,default=MASKED_PAIR_PPO.smoke_eval_episodes)
    parser.add_argument('--tasks-per-episode',type=int,default=MASKED_PAIR_PPO.tasks_per_episode)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    print(run(args.train_episodes,args.eval_episodes,args.tasks_per_episode,args.output_dir).to_string(index=False))
