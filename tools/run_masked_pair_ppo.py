"""Dedicated train/evaluate entry point for Reliability-Masked Pair PPO.

Examples:
    python tools/run_masked_pair_ppo.py train-smoke
    python tools/run_masked_pair_ppo.py evaluate --checkpoint-dir diagnostics/results/masked_pair_ppo_smoke
    python tools/run_masked_pair_ppo.py evaluate --deployment greedy  # ablation only
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent
from config.masked_pair_ppo import MASKED_PAIR_PPO
from diagnostics.run_masked_pair_ppo import (
    DEFAULT_OUT, PRIOR, evaluate_one, run as run_smoke, summary,
)
from Project_main import build_pair_correlations
from tools.paired_ppo_experiment import _agent_kwargs


def evaluate(checkpoint_dir=DEFAULT_OUT, output_dir=None, deployment='stochastic',
             episodes=MASKED_PAIR_PPO.smoke_eval_episodes,
             tasks=MASKED_PAIR_PPO.tasks_per_episode):
    if deployment not in ('stochastic','greedy'):
        raise ValueError('deployment must be stochastic or greedy')
    checkpoint_dir=Path(checkpoint_dir)
    output_dir=Path(output_dir) if output_dir else checkpoint_dir/'deployment'
    output_dir.mkdir(parents=True,exist_ok=True)
    run_meta=json.loads((checkpoint_dir/'run_metadata.json').read_text())
    trial=run_meta['seed_row']
    pairs,rho=build_pair_correlations()
    torch.manual_seed(int(trial['Torch_Init_Seed']))
    agent=ReliabilityMaskedPairPPOAgent(**_agent_kwargs('pair_scoring',np.asarray(rho,dtype=float),
        int(trial['PPO_Minibatch_Seed'])))
    actor=torch.load(checkpoint_dir/'masked_pair_actor.pt',map_location='cpu',weights_only=True)
    critic=torch.load(checkpoint_dir/'masked_pair_critic.pt',map_location='cpu',weights_only=True)
    agent.policy_old.load_state_dict(actor)
    agent.policy_net.load_state_dict(actor)
    agent.value_net.load_state_dict(critic)
    mode_name='masked_pair_stochastic' if deployment=='stochastic' else 'masked_pair_greedy_ablation'
    _,assignments,decisions,_=evaluate_one(agent,mode_name,trial,episodes,tasks,list(pairs),output_dir,deployment)
    metrics=pd.DataFrame([summary(assignments,decisions,mode_name)])
    metrics.to_csv(output_dir/'deployment_summary.csv',index=False)
    (output_dir/'deployment_metadata.json').write_text(json.dumps({
        'agent_name':MASKED_PAIR_PPO.agent_name,
        'deployment_mode':deployment,
        'greedy_is_ablation_only':deployment=='greedy',
        'checkpoint_dir':str(checkpoint_dir),
        'episodes':episodes,'tasks_per_episode':tasks},indent=2),encoding='utf-8')
    return metrics


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train-smoke')
    train.add_argument('--train-episodes',type=int,default=MASKED_PAIR_PPO.smoke_train_episodes)
    train.add_argument('--eval-episodes',type=int,default=MASKED_PAIR_PPO.smoke_eval_episodes)
    train.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    deployment=sub.add_parser('evaluate')
    deployment.add_argument('--checkpoint-dir',type=Path,default=DEFAULT_OUT)
    deployment.add_argument('--output-dir',type=Path)
    deployment.add_argument('--deployment',choices=('stochastic','greedy'),default=MASKED_PAIR_PPO.deployment_mode)
    deployment.add_argument('--episodes',type=int,default=MASKED_PAIR_PPO.smoke_eval_episodes)
    args=parser.parse_args()
    if args.command=='train-smoke':
        result=run_smoke(args.train_episodes,args.eval_episodes,MASKED_PAIR_PPO.tasks_per_episode,args.output_dir)
    else:
        result=evaluate(args.checkpoint_dir,args.output_dir,args.deployment,args.episodes)
    print(result.to_string(index=False))

if __name__=='__main__':main()
