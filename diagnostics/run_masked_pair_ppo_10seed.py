"""Run one resumable trial of the formal paired Flat/Pair/Masked PPO study.

The formal seed plan and hyperparameters come from the prior 300-episode study.
No simulator, Actor, Critic, loss, or reward definition is changed here.
"""
from __future__ import annotations

import argparse
import atexit
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

import core.main_loop as main_loop_module
from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent
from config.params import params
from Project_main import build_pair_correlations
from diagnostics.evaluate_reliability_masked_policy import ArrivalTraceLoop, TrackingEnvironmentState, summarize_server_load
from diagnostics.run_masked_pair_ppo import (
    BaselineGreedyAgent, check_external_trace, episode_metrics, persist_assignments,
    persist_decisions, trace,
)
from tools.pair_policy_diagnostics import DiagnosticPPOAgent, TASK_ASSIGNMENT_COLUMNS
from tools.paired_ppo_experiment import (
    _agent_kwargs, _state_dict_snapshot, assert_state_dict_unchanged,
    scoped_environment_seeds, sha256_file,
)

PRIOR = ROOT / 'diagnostics/paired_ppo_experiment/formal_10seed_300ep'
OUT = ROOT / 'diagnostics/results/masked_pair_ppo_10seed'


def formal_spec():
    snapshot=OUT/'formal_reference_metadata.json'
    meta = json.loads((snapshot if snapshot.exists() else PRIOR/'metadata.json').read_text())
    if (meta['num_trials'], meta['train_episodes'], meta['eval_episodes'], meta['tasks_per_episode']) != (10, 300, 20, 200):
        raise RuntimeError('Unexpected prior formal trial configuration')
    for field, path in [('server_info_sha256', ROOT/'data/server_info.xlsx'),
                        ('task_parameters_sha256', ROOT/'data/task_parameters.xlsx')]:
        if sha256_file(path) != meta[field]:
            raise RuntimeError(f'Input data changed: {path}')
    expected = {'ppo_actor_lr':params.actor_lr_ppo, 'ppo_critic_lr':params.critic_lr_ppo,
                'ppo_gamma':params.gamma_ppo, 'ppo_gae_lambda':params.gae_lambda_ppo,
                'ppo_clip_eps':params.clip_eps_ppo, 'ppo_entropy_coef':params.entropy_coef_ppo,
                'ppo_epochs':params.k_epochs_ppo, 'ppo_batch_size':params.batch_size_ppo,
                'num_states':params.num_states, 'num_actions':params.num_actions,
                'lambda_ref':params.LAMBDA_REF, 'omega':params.FAILURE_RATE_OMEGA,
                'beta':params.SPATIAL_RISK_BETA_P, 'ell':params.SPATIAL_CORRELATION_LENGTH_KM,
                'task_arrival_rate':params.TASK_ARRIVAL_RATE}
    for field, value in expected.items():
        if meta[field] != value:
            raise RuntimeError(f'Formal configuration changed: {field}: {meta[field]} != {value}')
    snapshot_plan=OUT/'formal_seed_plan.csv'
    plan = pd.read_csv(snapshot_plan if snapshot_plan.exists() else PRIOR/'seed_plan.csv')
    if plan.Trial_ID.tolist() != list(range(10)):
        raise RuntimeError('Prior seed plan must contain exactly trials 0..9')
    return meta, plan


def action_seed(trial):
    # Independent Torch action stream; external arrival/hazard streams are untouched.
    return int(np.random.SeedSequence([int(trial['Trial_ID']), int(trial['Eval_Arrival_Seed']), 20260924]).generate_state(1)[0])


def numeric_summary(frame):
    values = frame.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(values).all():
        raise RuntimeError('Non-finite training/evaluation summary')


def train(agent, name, trial, run_dir, meta):
    torch.manual_seed(int(trial['Train_Action_Seed']))
    with scoped_environment_seeds(int(trial['Train_Arrival_Seed']), int(trial['Train_Spatial_Seed'])):
        with (run_dir/'training.log').open('w') as log, redirect_stdout(log):
            loop = ArrivalTraceLoop(agent, 300, 200, params.num_states, params.num_actions)
            loop.EP()
    if len(loop.task_Assignments_info) != 60000 or len(loop.ep_reward_list) != 300:
        raise RuntimeError(f'{name} training did not finish 300x200 tasks')
    curve = pd.DataFrame({'episode':range(1,301),'episode_reward':loop.ep_reward_list,
                          'episode_total_delay':loop.ep_delay_list})
    numeric_summary(curve)
    curve.to_csv(run_dir/'training_curve.csv',index=False)
    if name in ('flat','pair'):
        reference=PRIOR/'runs'/f"trial_{int(trial['Trial_ID']):03d}"/('flat' if name=='flat' else 'pair_scoring')/'training_curve.csv'
        if reference.exists():
            prior=pd.read_csv(reference)
            if not np.allclose(curve.episode_reward.to_numpy(),prior.Episode_Reward.to_numpy(),rtol=0,atol=1e-10):
                raise RuntimeError(f'{name} historical 300-episode training curve mismatch')
    if name == 'masked':
        assignments = pd.DataFrame(loop.task_Assignments_info,columns=TASK_ASSIGNMENT_COLUMNS).sort_values(['episode','task_id']).reset_index(drop=True)
        decisions = pd.DataFrame(agent.selection_archive).sort_values(['episode','task_id']).reset_index(drop=True)
        if len(decisions) != 60000 or not np.array_equal(decisions.action_index, assignments.action_index):
            raise RuntimeError('Masked training action archive mismatch')
        if not np.array_equal(decisions.selected_action_safe.astype(bool), assignments.Reliability_Satisfied.astype(bool)):
            raise RuntimeError('Masked training feasibility mismatch')
        metrics = episode_metrics(assignments,decisions)
        counts = decisions.groupby('episode').action_index.value_counts(normalize=True)
        metrics['top1_pair_frequency'] = metrics.episode.map(counts.groupby(level=0).max())
        metrics['pair_selection_entropy'] = metrics.episode.map(
            counts.groupby(level=0).apply(lambda x: float(-(x.to_numpy()*np.log(x.to_numpy())).sum())))
        numeric_summary(metrics)
        metrics.to_csv(run_dir/'training_episode_metrics.csv',index=False)
        updates = pd.DataFrame(agent.update_diagnostics)
        if len(updates)!=300 or not np.isfinite(updates.select_dtypes(include=[np.number]).to_numpy()).all():
            raise RuntimeError('Masked PPO updates missing or non-finite')
        if updates.first_minibatch_ratio_max_abs_error.max() > 1e-5:
            raise RuntimeError('PPO old/new masked ratio changed before first update')
        updates.to_csv(run_dir/'ppo_updates.csv',index=False)
        fallback = decisions[decisions.safe_set_empty.astype(bool)]
        pd.DataFrame([{'empty_safe_tasks':len(fallback),
                       'mean_deficit':float(fallback.reliability_deficit.mean()) if len(fallback) else 0.0,
                       'p95_deficit':float(fallback.reliability_deficit.quantile(.95)) if len(fallback) else 0.0,
                       'avoidable_violations':int(((~decisions.safe_set_empty.astype(bool)) &
                           (~assignments.Reliability_Satisfied.astype(bool))).sum())}]).to_csv(run_dir/'training_mask_diagnostics.csv',index=False)
        if not all(bool(row.effective_mask[row.action_index]) for row in decisions.itertuples()):
            raise RuntimeError('Masked training action outside effective support')
    torch.save(agent.policy_old.state_dict(),run_dir/'actor.pt')
    torch.save(agent.value_net.state_dict(),run_dir/'critic.pt')
    if hasattr(agent, 'observation_archive'):
        agent.observation_archive = []
    return loop,curve


def evaluate(trained, name, trial, run_dir, pairs, rho, mode):
    if isinstance(trained,ReliabilityMaskedPairPPOAgent):
        import copy
        frozen=copy.deepcopy(trained)
        frozen.frozen=True;frozen.deployment_mode=mode
        frozen.clear_rollout();frozen.selection_archive=[]
    else:
        trained.pair_correlations=np.asarray(rho,dtype=float)
        frozen=BaselineGreedyAgent.from_trained(trained)
        frozen.pairs=pairs;frozen.selection_archive=[]
    before=_state_dict_snapshot(frozen)
    TrackingEnvironmentState.instances=[]
    TrackingEnvironmentState.active_agent=frozen
    torch.manual_seed(action_seed(trial))
    with patch.object(main_loop_module,'EnvironmentState',TrackingEnvironmentState):
        with scoped_environment_seeds(int(trial['Eval_Arrival_Seed']),int(trial['Eval_Spatial_Seed'])):
            with (run_dir/'evaluation.log').open('w') as log,redirect_stdout(log):
                loop=ArrivalTraceLoop(frozen,20,200,params.num_states,params.num_actions)
                loop.EP()
    assert_state_dict_unchanged(before,frozen)
    for state in TrackingEnvironmentState.instances:state.finalize()
    assignments=persist_assignments(loop,run_dir,'evaluation')
    decisions=persist_decisions(frozen,assignments,run_dir,'evaluation')
    episode_metrics(assignments,decisions).to_csv(run_dir/'evaluation_episode_metrics.csv',index=False)
    load=summarize_server_load(TrackingEnvironmentState.instances,assignments,name)
    load.to_csv(run_dir/'server_load.csv',index=False)
    if len(assignments)!=4000 or len(decisions)!=4000:
        raise RuntimeError('Expected exactly 4000 evaluation decisions')
    if name in ('flat','pair'):
        reference=PRIOR/'runs'/f"trial_{int(trial['Trial_ID']):03d}"/('flat' if name=='flat' else 'pair_scoring')/'evaluation_task_assignments.csv'
        if reference.exists():
            old=pd.read_csv(reference).sort_values(['episode','task_id']).reset_index(drop=True)
            if not np.array_equal(assignments.action_index,old.action_index) or not np.allclose(
                assignments.Task_Reward,old.Task_Reward,rtol=0,atol=1e-10):
                raise RuntimeError(f'{name} evaluation differs from prior formal experiment')
    return loop,assignments,decisions,load


def run_trial(trial_id):
    # Independent seed processes use one CPU thread each; this changes no PPO hyperparameter.
    torch.set_num_threads(1)
    meta,plan=formal_spec()
    trial=plan.iloc[int(trial_id)].to_dict()
    run_dir=OUT/'runs'/f'trial_{trial_id:03d}'
    run_dir.mkdir(parents=True,exist_ok=True)
    if (run_dir/'completed.json').exists():
        print(f'trial {trial_id} already complete; skipping', flush=True)
        return
    lock=run_dir/'.running'
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        os.close(fd)
        atexit.register(lambda: lock.unlink(missing_ok=True))
    except FileExistsError:
        print(f'trial {trial_id} is running in another process; skipping',flush=True)
        return
    pairs,rho=build_pair_correlations();pairs=list(pairs);rho=np.asarray(rho,dtype=float)
    records=[];reference_train=None;reference_eval=None
    for name,actor_mode in [('flat','flat'),('pair','pair_scoring'),('masked','pair_scoring')]:
        sub=run_dir/name;sub.mkdir(exist_ok=True)
        torch.manual_seed(int(trial['Torch_Init_Seed']))
        kwargs=_agent_kwargs(actor_mode,rho,int(trial['PPO_Minibatch_Seed']))
        agent=(ReliabilityMaskedPairPPOAgent(**kwargs) if name=='masked' else DiagnosticPPOAgent(**kwargs))
        if all((sub/file).exists() for file in ('actor.pt','critic.pt','training_curve.csv')):
            print(f'trial {trial_id} {name} loading completed training checkpoint',flush=True)
            actor=torch.load(sub/'actor.pt',map_location='cpu',weights_only=True)
            agent.policy_net.load_state_dict(actor);agent.policy_old.load_state_dict(actor)
            agent.value_net.load_state_dict(torch.load(sub/'critic.pt',map_location='cpu',weights_only=True))
            curve=pd.read_csv(sub/'training_curve.csv')
            if len(curve)!=300:raise RuntimeError('Incomplete saved training curve')
        else:
            print(f'trial {trial_id} {name} training',flush=True)
            train_loop,curve=train(agent,name,trial,sub,meta)
            if reference_train is None:reference_train=trace(train_loop)
            else:check_external_trace(trace(train_loop),reference_train)
        print(f'trial {trial_id} {name} evaluating',flush=True)
        modes=[('masked_stochastic','stochastic'),('masked_greedy','greedy')] if name=='masked' else [(name,'greedy')]
        for label,mode in modes:
            dest=run_dir/label;dest.mkdir(exist_ok=True)
            loop,assignments,decisions,load=evaluate(agent,label,trial,dest,pairs,rho,mode)
            if reference_eval is None:reference_eval=trace(loop)
            else:check_external_trace(trace(loop),reference_eval)
            records.append({'trial_id':trial_id,'policy':label,'task_count':len(assignments),
                            'actor_sha256':sha256_file(sub/'actor.pt'),
                            'critic_sha256':sha256_file(sub/'critic.pt'),
                            'eval_action_seed':action_seed(trial)})
            print(f'trial {trial_id} {label} done',flush=True)
        (sub/'config.json').write_text(json.dumps({
            'policy':name,'training_episodes':300,'evaluation_episodes':20,
            'tasks_per_episode':200,'deployment_modes':[x[0] for x in modes],
            'formal_reference_commit':meta['git_commit'],
            'ppo_actor_lr':params.actor_lr_ppo,'ppo_critic_lr':params.critic_lr_ppo,
            'ppo_entropy_coef':params.entropy_coef_ppo,'ppo_gamma':params.gamma_ppo,
            'ppo_gae_lambda':params.gae_lambda_ppo,'ppo_clip_eps':params.clip_eps_ppo,
            'ppo_epochs':params.k_epochs_ppo,'ppo_batch_size':params.batch_size_ppo},indent=2))
    (run_dir/'seed_metadata.json').write_text(json.dumps({**trial,'Eval_Action_Seed':action_seed(trial)},indent=2))
    (run_dir/'completed.json').write_text(json.dumps({'trial_id':trial_id,'runs':records,
        'external_train_trace_paired':True,'external_eval_trace_paired':True,
        'legacy_training_evaluation_reproduced':(PRIOR/'runs'/f'trial_{trial_id:03d}'/'flat'/'training_curve.csv').exists()},indent=2))
    lock.unlink()
    print(f'trial {trial_id} complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--trial-id',type=int,choices=range(10),required=True)
    args=parser.parse_args()
    run_trial(args.trial_id)
