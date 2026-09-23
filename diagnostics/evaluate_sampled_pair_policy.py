"""Frozen stochastic-vs-greedy evaluation with identical workload/risk seeds.

No optimizer step is performed. This isolates the deployment-policy-induced
queue distribution shift using the instrumented trial's final checkpoint.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from agents.ppo_agent import PPOAgent
from config.params import params
from core.main_loop import MainLoop
from Project_main import build_pair_correlations
from tools.paired_ppo_experiment import (
    FrozenEvaluationPPOAgent, _agent_kwargs, assert_state_dict_unchanged,
    _state_dict_snapshot, scoped_environment_seeds,
)
from tools.pair_policy_diagnostics import DiagnosticPPOAgent, TASK_ASSIGNMENT_COLUMNS
from diagnostics.requirement_conditioning_diagnostic import FORMAL, REQS

class FrozenSampledPairAgent(FrozenEvaluationPPOAgent):
    def select_action(self,state,epsilon=0.0,use_softmax=False,temperature=1.5):
        return PPOAgent.select_action(self,state,epsilon,use_softmax,temperature)


def run(run_dir:Path,seeds=(2026,2027,2028)):
    meta=json.loads((run_dir/'run_metadata.json').read_text())
    trial=pd.read_csv(FORMAL/'seed_plan.csv').set_index('Trial_ID').loc[int(meta['trial_id'])]
    _,rho=build_pair_correlations()
    trained=DiagnosticPPOAgent(**_agent_kwargs('pair_scoring',np.asarray(rho,dtype=float),int(trial.PPO_Minibatch_Seed)))
    trained.policy_old.load_state_dict(torch.load(run_dir/'actor_final.pt',map_location='cpu',weights_only=True))
    trained.policy_net.load_state_dict(trained.policy_old.state_dict())
    trained.value_net.load_state_dict(torch.load(run_dir/'critic_final.pt',map_location='cpu',weights_only=True))
    before=_state_dict_snapshot(trained)
    summary=[]
    greedy=pd.read_csv(run_dir/'evaluation_task_assignments.csv')
    sources=[('greedy',None,greedy)]
    for seed in seeds:
        frozen=FrozenSampledPairAgent.from_trained(trained)
        frozen.observation_archive=[]
        torch.manual_seed(int(seed))
        with scoped_environment_seeds(int(trial.Eval_Arrival_Seed),int(trial.Eval_Spatial_Seed)):
            loop=MainLoop(frozen,int(meta['eval_episodes']),int(meta['tasks_per_episode']),params.num_states,params.num_actions)
            loop.EP()
        assert_state_dict_unchanged(before,trained)
        assert_state_dict_unchanged(before,frozen)
        frame=pd.DataFrame(loop.task_Assignments_info,columns=TASK_ASSIGNMENT_COLUMNS)
        assert len(frame)==len(greedy)==4000
        frame.to_csv(run_dir/f'sampled_evaluation_seed{seed}.csv',index=False)
        sources.append(('sampled',seed,frame))
    for mode,seed,frame in sources:
        for req in (None,*REQS):
            subset=frame if req is None else frame[np.isclose(frame.Reliability_Requirement,req)]
            summary.append({'policy_mode':mode,'action_seed':seed,'R_req':'all' if req is None else str(req),
                'tasks':len(subset),'pair_78_fraction':float((subset.action_index==27).mean()),
                'mean_task_delay':float(subset.Task_Delay.mean()),
                'mean_task_reward':float(subset.Task_Reward.mean()),
                'reliability_satisfaction_rate':float(subset.Reliability_Satisfied.mean())})
    result=pd.DataFrame(summary);result.to_csv(run_dir/'frozen_policy_mode_comparison.csv',index=False)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,
        default=ROOT/'diagnostics/results/policy_oracle_alignment')
    args=parser.parse_args();print(run(args.run_dir).to_string(index=False))
