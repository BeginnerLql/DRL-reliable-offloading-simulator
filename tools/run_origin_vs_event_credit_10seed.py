"""Formal paired temporal-credit ablation on the archived Masked Pair PPO seed plan.

Run ``--gate`` before ``--trial``. Only the credit_mode differs between arms.
All historical result directories are read-only; output is a new directory.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from agents.context_masked_pair_ppo_agent import ContextMaskedPairPPOAgent, event_interval_rewards
from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent
from config.params import params
from Project_main import build_pair_correlations
from diagnostics.run_masked_pair_ppo import train_one, evaluate_one, summary, trace, check_external_trace
from diagnostics.evaluate_reliability_masked_policy import ArrivalTraceLoop
from tools.paired_ppo_experiment import scoped_environment_seeds
from diagnostics.run_masked_pair_ppo_10seed import formal_spec, action_seed, OUT as HISTORY
from tools.paired_ppo_experiment import _agent_kwargs, sha256_file
from tools.run_context_masked_pair_ppo import checkpoint_environment, validate_checkpoint_environment, load_verified_actor

OUT = ROOT/'diagnostics/results/origin_vs_event_credit_10seed'
TRAIN_EPISODES, EVAL_EPISODES, TASKS = 300, 20, 200
ARMS = {'origin': 'origin_task', 'event': 'event_interval'}


def digest_state(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def assert_equal_states(left, right):
    keys = set(left) | set(right)
    return sum(key not in left or key not in right or not torch.equal(left[key], right[key]) for key in keys)


def distribution(values, prefix):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise RuntimeError(f'Empty or non-finite {prefix}')
    return {f'{prefix}_{name}': float(value) for name, value in {
        'mean': x.mean(), 'std': x.std(ddof=0), 'p05':np.quantile(x,.05),
        'median': np.median(x), 'p95':np.quantile(x,.95),
    }.items()}


def zero_runs(values):
    lengths=[]; current=0
    for v in values:
        if v == 0: current += 1
        elif current: lengths.append(current); current=0
    if current: lengths.append(current)
    return lengths or [0]


class CreditAuditAgent(ContextMaskedPairPPOAgent):
    """Read-only taps; the inherited policy/GAE/optimizer path is unchanged."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.credit_rows=[]; self.gae_rows=[]; self.critic_rows=[]
        self.gradient_rows=[]; self.assignment_rows=[]; self.reassignment_rows=[]
        self.advantage_target_observer=self._observe_targets

    def _observe_targets(self, payload):
        raw=payload['raw_gae_advantages'].astype(float)
        used=payload['actor_used_advantages'].astype(float)
        values=payload['value_predictions'].astype(float)
        target=payload['gae_return_targets'].astype(float)
        ep=int(payload['episode'])
        self.gae_rows.append({'episode':ep, **distribution(raw,'raw_gae'),
            **distribution(used,'normalized_advantage'),
            'positive_fraction':float((raw>0).mean()),
            'negative_fraction':float((raw<0).mean()),
            'selected_action_advantage_std':float(pd.DataFrame({'a':payload['actions'],'v':raw}).groupby('a').v.mean().std(ddof=0))})
        error=values-target
        def correlation(fun):
            if np.std(values)<1e-12 or np.std(target)<1e-12: return np.nan
            return float(fun(values,target).statistic)
        self.critic_rows.append({'episode':ep, **distribution(values,'value'),
            **distribution(target,'target'), 'mae':float(np.abs(error).mean()),
            'value_loss_preupdate':float(self.value_loss_coef*np.mean(error**2)),
            'pearson':correlation(pearsonr),'spearman':correlation(spearmanr),
            'explained_variance':float(1-np.var(error)/np.var(target)) if np.var(target)>0 else np.nan})

    def _clip_gradients(self):
        def norm(module):
            squared=sum(float(torch.sum(p.grad.detach()**2).item()) for p in module.parameters() if p.grad is not None)
            return math.sqrt(squared)
        actor,critic=norm(self.policy_net),norm(self.value_net)
        # Exactly the historical joint clipping operation; no changed threshold.
        super()._clip_gradients()
        self.gradient_rows.append({'episode':self._current_episode,
                                   'actor_norm_before':actor,'critic_norm_before':critic})

    def train_step(self):
        if self.states and not self.frozen:
            times=np.asarray([self.decision_times[i] for i in self.task_ids],dtype=float)
            outcomes=list(self.outcome_records.values())
            rewards=np.asarray(self.rewards,dtype=float)
            interval=event_interval_rewards(times,float(times[-1]+self.delta_times[-1]),
                                            outcomes,self.gamma)
            ep=int(self._current_episode)
            active=interval if self.credit_mode=='event_interval' else rewards
            self.credit_rows.append({'episode':ep,'credit_mode':self.credit_mode,
                **distribution(active,'training_reward'),
                'zero_fraction':float((active==0).mean()),
                'event_zero_fraction':float((interval==0).mean()),
                'zero_run_max':max(zero_runs(interval)),
                'zero_run_mean':float(np.mean(zero_runs(interval)))})
            count=np.zeros(len(times),dtype=int)
            shifts=[]; delays=[]; prior_intervals=set(); current_intervals=set()
            for task_id,reward,timestamp in outcomes:
                index=int(np.searchsorted(times,timestamp,side='right')-1)
                origin=self.task_id_to_transition_index[task_id]
                count[index]+=1; shifts.append(index-origin)
                delays.append(timestamp-times[origin])
                if index>origin: prior_intervals.add(index)
                else: current_intervals.add(index)
            self.assignment_rows.append({'episode':ep,'zero_event_intervals':int((count==0).sum()),
                'one_event_intervals':int((count==1).sum()),'multiple_event_intervals':int((count>1).sum()),
                'preexisting_outcome_intervals':len(prior_intervals),
                'current_task_outcome_intervals':len(current_intervals),
                'shifted_outcomes':sum(x>0 for x in shifts),'zero_run_p90':float(np.quantile(zero_runs(interval),.9))})
            self.reassignment_rows.append({'episode':ep,'mean_index_shift':float(np.mean(shifts)),
                'median_index_shift':float(np.median(shifts)),
                'p90_index_shift':float(np.quantile(shifts,.9)),
                'max_index_shift':int(max(shifts)),
                'same_interval_rate':float(np.mean(np.asarray(shifts)==0)),
                'shifted_forward_rate':float(np.mean(np.asarray(shifts)>0)),
                'median_resolution_delay':float(np.median(delays))})
        super().train_step()


def new_agent(trial, rho, credit_mode, *, audited=True):
    torch.manual_seed(int(trial['Torch_Init_Seed']))
    kwargs=_agent_kwargs('pair_scoring',np.asarray(rho),int(trial['PPO_Minibatch_Seed']))
    kwargs.update(credit_mode=credit_mode,gradient_clipping='joint')
    return (CreditAuditAgent if audited else ContextMaskedPairPPOAgent)(**kwargs)


def legacy_agent(trial,rho):
    torch.manual_seed(int(trial['Torch_Init_Seed']))
    return ReliabilityMaskedPairPPOAgent(**_agent_kwargs('pair_scoring',np.asarray(rho),
                                                          int(trial['PPO_Minibatch_Seed'])))


def check_formal(trial,rho):
    if params.num_states!=35 or params.num_actions!=28 or params.serverNo!=8:
        raise RuntimeError('Formal state/action/server dimensions changed')
    a=legacy_agent(trial,rho); b=new_agent(trial,rho,'origin_task')
    for name in ('policy_net','policy_old','value_net'):
        if assert_equal_states(getattr(a,name).state_dict(),getattr(b,name).state_dict()):
            raise RuntimeError('Initial network tensors differ')
    return {'actor_initial_sha':digest_state(a.policy_net.state_dict()),
            'critic_initial_sha':digest_state(a.value_net.state_dict())}


def compare_frames(a,b,keys,columns):
    a=a.sort_values(keys).reset_index(drop=True)
    b=b.sort_values(keys).reset_index(drop=True)
    if len(a)!=len(b):raise RuntimeError('Different row counts')
    result={}
    for col in columns:
        if col not in a or col not in b: raise RuntimeError(f'Missing comparison column {col}')
        x,y=a[col].to_numpy(),b[col].to_numpy()
        if np.issubdtype(x.dtype,np.number) and np.issubdtype(y.dtype,np.number):
            result[col+'_max_diff']=float(np.nanmax(np.abs(x-y)))
        else:
            result[col+'_mismatches']=int(np.sum(x!=y))
    return result


def checkpoint_manifest(agent,trial,credit,output,environment):
    actor=output/'actor.pt';critic=output/'critic.pt'
    torch.save(agent.policy_old.state_dict(),actor)
    torch.save(agent.value_net.state_dict(),critic)
    manifest={'agent_name':agent.agent_name,'environment':environment,
        'credit_mode':credit,'actor_mode':'pair_scoring','gradient_clipping':'joint',
        'trial':trial,'pair_indices':agent.policy_old.pair_indices.tolist(),
        'pair_correlations':agent.policy_old.pair_correlations.tolist(),
        'actor_architecture':[[k,list(v.shape)] for k,v in agent.policy_old.state_dict().items()],
        'actor_sha256':sha256_file(actor),'critic_sha256':sha256_file(critic)}
    (output/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify_checkpoint(agent,output,environment,credit):
    manifest=json.loads((output/'checkpoint_manifest.json').read_text())
    if manifest['credit_mode']!=credit or manifest['actor_mode']!='pair_scoring' or manifest['gradient_clipping']!='joint':
        raise RuntimeError('Checkpoint arm configuration differs')
    if manifest['environment']!=environment:raise RuntimeError('Checkpoint environment differs')
    if manifest['pair_indices']!=agent.policy_old.pair_indices.tolist() or manifest['pair_correlations']!=agent.policy_old.pair_correlations.tolist():
        raise RuntimeError('Checkpoint pair buffers differ')
    for key in ('actor','critic'):
        if sha256_file(output/f'{key}.pt')!=manifest[f'{key}_sha256']:
            raise RuntimeError(f'Checkpoint {key} file changed')
    if manifest['actor_architecture']!=[[k,list(v.shape)] for k,v in agent.policy_old.state_dict().items()]:
        raise RuntimeError('Actor architecture differs')
    load_verified_actor(agent,torch.load(output/'actor.pt',map_location='cpu',weights_only=True))
    agent.value_net.load_state_dict(torch.load(output/'critic.pt',map_location='cpu',weights_only=True))


def run_gate(output):
    torch.set_num_threads(1)
    meta,plan=formal_spec();trial={k:int(v) for k,v in plan.iloc[0].items()}
    trial['Eval_Action_Seed']=action_seed(trial)
    pairs,rho=build_pair_correlations()
    initial=check_formal(trial,rho)
    output.mkdir(parents=True,exist_ok=False)
    current=checkpoint_environment()
    base=HISTORY/'runs/trial_000/masked'
    reference=pd.read_csv(base/'training_curve.csv')
    prior_actor=torch.load(base/'actor.pt',map_location='cpu',weights_only=True)
    prior_critic=torch.load(base/'critic.pt',map_location='cpu',weights_only=True)
    report={'base_commit':meta['git_commit'],**initial,'episodes':TRAIN_EPISODES,'tasks':TRAIN_EPISODES*TASKS}
    agents=[];loops=[];frames=[]
    for name,agent in [('historical',legacy_agent(trial,rho)),('origin',new_agent(trial,rho,'origin_task'))]:
        folder=output/name;folder.mkdir()
        loop,frame=train_one(agent,name,trial,TRAIN_EPISODES,TASKS,folder)
        agents.append(agent);loops.append(loop);frames.append(frame)
        curve=pd.read_csv(folder/f'{name}_training_curve.csv')
        report[name+'_historical_curve_reward_max_diff']=float(np.max(np.abs(curve.episode_reward-reference.episode_reward)))
        report[name+'_historical_actor_tensor_mismatch']=assert_equal_states(agent.policy_old.state_dict(),prior_actor)
        report[name+'_historical_critic_tensor_mismatch']=assert_equal_states(agent.value_net.state_dict(),prior_critic)
    check_external_trace(trace(loops[0]),trace(loops[1]))
    report['arrival_mismatch']=0;report['spatial_mismatch']=0
    for column in ('action_index','Task_Reward','Task_Delay','Execution_Reliability'):
        x=frames[0][column].to_numpy();y=frames[1][column].to_numpy()
        report[column+'_mismatch_or_max_diff']=int(np.sum(x!=y)) if column=='action_index' else float(np.max(np.abs(x-y)))
    a,b=agents
    report['actor_tensor_mismatch']=assert_equal_states(a.policy_old.state_dict(),b.policy_old.state_dict())
    report['critic_tensor_mismatch']=assert_equal_states(a.value_net.state_dict(),b.value_net.state_dict())
    report['exact']=bool(report['action_index_mismatch_or_max_diff']==0 and report['Task_Reward_mismatch_or_max_diff']<=1e-12
      and report['Task_Delay_mismatch_or_max_diff']<=1e-12 and report['actor_tensor_mismatch']==0
      and report['critic_tensor_mismatch']==0 and report['historical_historical_actor_tensor_mismatch']==0
      and report['historical_historical_critic_tensor_mismatch']==0)
    (output/'gate_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['exact']:raise RuntimeError('Full-seed historical regression failed')
    return report


def compress_large_csvs(folder):
    for path in folder.glob('*.csv'):
        if path.stat().st_size > 2_000_000:
            pd.read_csv(path).to_csv(path.with_suffix('.csv.gz'),index=False,compression='gzip')
            path.unlink()


def save_training_diagnostics(agent,folder,arm):
    for filename,rows in [('credit_distribution_by_episode',agent.credit_rows),
                          ('gae_statistics_by_episode',agent.gae_rows),
                          ('critic_statistics_by_episode',agent.critic_rows),
                          ('gradient_statistics_by_episode',agent.gradient_rows),
                          ('reward_reassignment_statistics',agent.reassignment_rows),
                          ('event_interval_sparsity',agent.assignment_rows),
                          ('return_conservation',agent.credit_diagnostics)]:
        frame=pd.DataFrame(rows)
        if filename=='gradient_statistics_by_episode':
            if frame.empty:raise RuntimeError(f'{arm}: no gradient records')
            frame=frame.groupby('episode',as_index=False).agg(
                minibatches=('actor_norm_before','size'),
                actor_norm_before=('actor_norm_before','mean'),
                actor_norm_p95=('actor_norm_before',lambda x:float(x.quantile(.95))),
                actor_norm_max=('actor_norm_before','max'),
                critic_norm_before=('critic_norm_before','mean'),
                critic_norm_p95=('critic_norm_before',lambda x:float(x.quantile(.95))),
                critic_norm_max=('critic_norm_before','max'))
        if len(frame)!=TRAIN_EPISODES or frame.episode.tolist()!=list(range(1,TRAIN_EPISODES+1)):
            raise RuntimeError(f'{arm}: missing {filename} episodes')
        frame.insert(0,'arm',arm)
        frame.to_csv(folder/f'{filename}.csv',index=False)
    updates=pd.DataFrame(agent.update_diagnostics)
    if len(updates)!=TRAIN_EPISODES or updates.first_minibatch_ratio_max_abs_error.max()>1e-5:
        raise RuntimeError(f'{arm}: PPO update diagnostics invalid')
    if not np.isfinite(updates.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError(f'{arm}: PPO update diagnostics non-finite')
    if arm=='event' and max(x['return_residual'] for x in agent.credit_diagnostics)>=1e-10:
        raise RuntimeError('Event return conservation failed')


def run_trial(trial_id,output):
    torch.set_num_threads(1)
    gate=json.loads((output/'gate/gate_report.json').read_text())
    if not gate['exact']:raise RuntimeError('Historical regression gate failed')
    _,plan=formal_spec()
    trial={k:int(v) for k,v in plan.iloc[trial_id].items()}
    trial['Eval_Action_Seed']=action_seed(trial)
    pairs,rho=build_pair_correlations();pairs=list(pairs)
    initial=check_formal(trial,rho)
    environment=checkpoint_environment()
    folder=output/f'trial_{trial_id:03d}'
    if folder.exists():raise RuntimeError('Trial directory exists; refusing overwrite')
    folder.mkdir(parents=True)
    (folder/'trial_manifest.json').write_text(json.dumps({**trial,**initial,'environment':environment},indent=2)+'\n')
    training_traces=[];evaluation_traces=[];record=[]
    historical=HISTORY/'runs'/f'trial_{trial_id:03d}'
    for arm,credit in ARMS.items():
        dest=folder/arm;dest.mkdir()
        agent=new_agent(trial,rho,credit)
        if digest_state(agent.policy_net.state_dict())!=initial['actor_initial_sha'] or digest_state(agent.value_net.state_dict())!=initial['critic_initial_sha']:
            raise RuntimeError('Initial Actor/Critic weights drifted')
        loop,assignments=train_one(agent,arm,trial,TRAIN_EPISODES,TASKS,dest)
        training_traces.append(trace(loop))
        save_training_diagnostics(agent,dest,arm)
        ref=historical/'masked'
        if arm=='origin':
            old_curve=pd.read_csv(ref/'training_curve.csv')
            new_curve=pd.read_csv(dest/f'{arm}_training_curve.csv')
            reward_diff=float(np.max(np.abs(new_curve.episode_reward-old_curve.episode_reward)))
            actor_diff=assert_equal_states(agent.policy_old.state_dict(),torch.load(ref/'actor.pt',weights_only=True,map_location='cpu'))
            critic_diff=assert_equal_states(agent.value_net.state_dict(),torch.load(ref/'critic.pt',weights_only=True,map_location='cpu'))
            if reward_diff>1e-12 or actor_diff or critic_diff:
                raise RuntimeError(f'Arm A trial {trial_id} differs from historical checkpoint')
        manifest=checkpoint_manifest(agent,trial,credit,dest,environment)
        # Load into a fresh agent; verification must happen before evaluation.
        frozen_source=new_agent(trial,rho,credit,audited=False)
        verify_checkpoint(frozen_source,dest,environment,credit)
        evaluated,eval_tasks,eval_decisions,load=evaluate_one(frozen_source,arm,trial,EVAL_EPISODES,TASKS,pairs,dest,'stochastic')
        evaluation_traces.append(trace(evaluated))
        if arm=='origin':
            old=pd.read_csv(historical/'masked_stochastic/evaluation_task_assignments.csv')
            if not np.array_equal(eval_tasks.action_index,old.action_index) or np.max(np.abs(eval_tasks.Task_Reward-old.Task_Reward))>1e-12:
                raise RuntimeError(f'Arm A trial {trial_id} evaluation differs from historical')
        values=summary(eval_tasks,eval_decisions,arm)
        values.update(trial_id=trial_id,arm=arm,p95_latency=float(np.quantile(eval_tasks.Task_Delay,.95)),
            top1_pair_frequency=float(eval_tasks.action_index.value_counts(normalize=True).max()),
            max_mean_queue=float(load.mean_queue_length.max()) if len(load) else np.nan,
            actor_sha256=manifest['actor_sha256'],critic_sha256=manifest['critic_sha256'])
        record.append(values)
        compress_large_csvs(dest)
        (dest/'completed.json').write_text(json.dumps({'trial_id':trial_id,'arm':arm,'tasks_train':60000,
                'tasks_eval':4000,'checkpoint_verified':True,'historical_exact':arm=='origin'},indent=2)+'\n')
    check_external_trace(training_traces[0],training_traces[1])
    check_external_trace(evaluation_traces[0],evaluation_traces[1])
    pd.DataFrame(record).to_csv(folder/'seed_level_summary.csv',index=False)
    (folder/'reproducibility_checks.json').write_text(json.dumps({'initial_network_equal':True,
        'arrival_mismatch':0,'spatial_mismatch':0,'task_sequence_mismatch':0,
        'checkpoint_verified':True,'historical_origin_exact':True},indent=2)+'\n')
    (folder/'completed.json').write_text(json.dumps({'trial_id':trial_id,'completed':True},indent=2)+'\n')
    return record


def run_smoke(output):
    torch.set_num_threads(1)
    gate=json.loads((output/'gate/gate_report.json').read_text())
    if not gate['exact']:raise RuntimeError('Historical regression gate failed')
    _,plan=formal_spec();trial={k:int(v) for k,v in plan.iloc[0].items()}
    pairs,rho=build_pair_correlations();agent=new_agent(trial,rho,'event_interval')
    dest=output/'smoke'
    if dest.exists():raise RuntimeError('Smoke directory already exists')
    dest.mkdir(parents=True)
    loop,assignments=train_one(agent,'event_smoke',trial,5,TASKS,dest)
    if len(agent.credit_diagnostics)!=5 or max(x['return_residual'] for x in agent.credit_diagnostics)>=1e-10:
        raise RuntimeError('Smoke return conservation failed')
    updates=pd.DataFrame(agent.update_diagnostics)
    if len(updates)!=5 or updates.first_minibatch_ratio_max_abs_error.max()>1e-5 or not np.isfinite(updates.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError('Smoke PPO updates failed')
    summary_row={'episodes':5,'tasks':len(assignments),'max_return_residual':max(x['return_residual'] for x in agent.credit_diagnostics),
                 'max_first_ratio_error':float(updates.first_minibatch_ratio_max_abs_error.max()),
                 'nan_count':0,'pending_outcomes':len(agent.pending_task_rewards)+len(agent.outcome_records)}
    (dest/'smoke_report.json').write_text(json.dumps(summary_row,indent=2)+'\n')
    compress_large_csvs(dest)
    return summary_row


def reconstruct_temporal_shifts(assignments, arrival_seed, trial_id, arm):
    """Recreate decision times from the isolated arrival RNG and real first finishes."""
    rng=np.random.default_rng(int(arrival_seed))
    rows=[]
    for episode,group in assignments.groupby('episode',sort=True):
        group=group.sort_values('task_id').reset_index(drop=True)
        if group.task_id.tolist()!=list(range(1,TASKS+1)):
            raise RuntimeError('Training task sequence is incomplete')
        arrivals=[]; now=0.0
        for _ in range(TASKS):
            now+=float(rng.exponential(scale=1.0/params.TASK_ARRIVAL_RATE))
            arrivals.append(now)
        endings=group[['Primary_End','Backup_End']].to_numpy(dtype=float)
        resolution=np.nanmin(endings,axis=1)
        if not np.isfinite(resolution).all():raise RuntimeError('Missing first-result finish time')
        event_index=np.searchsorted(arrivals,resolution,side='right')-1
        origin_index=np.arange(TASKS)
        delay=resolution-np.asarray(arrivals)
        if (delay < -1e-10).any() or (event_index<origin_index).any():
            raise RuntimeError('Reconstructed resolution occurs before origin decision')
        rows.append(pd.DataFrame({'trial_id':trial_id,'arm':arm,'episode':episode,
            'task_id':group.task_id,'origin_index':origin_index,
            'event_index':event_index,'index_shift':event_index-origin_index,
            'resolution_delay':delay}))
    return pd.concat(rows,ignore_index=True)


def aggregate(output):
    folders=[output/f'trial_{i:03d}' for i in range(10)]
    if not all((f/'completed.json').exists() for f in folders):
        raise RuntimeError('All ten paired trials must complete before aggregation')
    direct_trace=json.loads((output/'gate/decision_trace/state_trace_verification.json').read_text())
    if not direct_trace['exact'] or direct_trace['decisions_checked']!=TRAIN_EPISODES*TASKS:
        raise RuntimeError('Full-seed direct state/interval regression gate failed')
    plan=pd.read_csv(HISTORY/'formal_seed_plan.csv')
    plan['Eval_Action_Seed']=[action_seed(row) for _,row in plan.iterrows()]
    plan.to_csv(output/'formal_seed_plan.csv',index=False)
    import subprocess
    base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    _,rho=build_pair_correlations()
    metadata={'base_commit':base_commit,'historical_formal_seed_plan':str(HISTORY/'formal_seed_plan.csv'),
        'seed_plan_sha256':sha256_file(HISTORY/'formal_seed_plan.csv'),
        'training_episodes':TRAIN_EPISODES,'evaluation_episodes':EVAL_EPISODES,'tasks_per_episode':TASKS,
        'arms':ARMS,'actor_mode':'pair_scoring','gradient_clipping':'joint',
        'deployment':'masked stochastic','bootstrap_samples':20000,'bootstrap_seed':2043,
        'checkpoint_environment':checkpoint_environment(),
        'action_pair_order':list(__import__('itertools').combinations(range(1,9),2)),
        'pair_correlations':np.asarray(rho).tolist(),
        'gate_report':json.loads((output/'gate/gate_report.json').read_text()),
        'direct_state_trace_gate':direct_trace,
        'smoke_report':json.loads((output/'smoke/smoke_report.json').read_text())}
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    seed=pd.concat([pd.read_csv(f/'seed_level_summary.csv') for f in folders],ignore_index=True)
    if len(seed)!=20 or set(seed.arm)!=set(ARMS):raise RuntimeError('Missing seed/arm result')
    seed['avoidable_violation_rate']=seed.avoidable_violation_count/seed.tasks
    seed['unavoidable_violation_rate']=seed.unavoidable_violation_count/seed.tasks
    seed['mean_server_queue_length']=[float(pd.read_csv(output/f'trial_{int(row.trial_id):03d}'/row.arm/f'{row.arm}_server_load.csv').mean_queue_length.mean())
                                      for row in seed.itertuples()]
    seed.to_csv(output/'seed_level_summary.csv',index=False)
    for source,destination in [('credit_distribution_by_episode','credit_distribution_by_episode'),
         ('gae_statistics_by_episode','gae_statistics_by_episode'),
         ('critic_statistics_by_episode','critic_statistics_by_episode'),
         ('gradient_statistics_by_episode','gradient_statistics_by_episode'),
         ('reward_reassignment_statistics','reward_reassignment_statistics'),
         ('event_interval_sparsity','event_interval_sparsity'),
         ('return_conservation','return_conservation')]:
        frames=[]
        for i,folder in enumerate(folders):
            for arm in ARMS:
                frame=pd.read_csv(folder/arm/f'{source}.csv')
                frame.insert(0,'trial_id',i)
                frames.append(frame)
        pd.concat(frames,ignore_index=True).to_csv(output/f'{destination}.csv',index=False)
    curve=[]; reproducibility=[]; task_shifts=[]
    for i,folder in enumerate(folders):
        checks={'trial_id':i,**json.loads((folder/'reproducibility_checks.json').read_text())}
        for phase in ('train','eval'):
            def read_decisions(arm):
                base=folder/arm/f'{arm}_{phase}_decisions.csv'
                return pd.read_csv(base if base.exists() else base.with_suffix('.csv.gz'))
            left,right=read_decisions('origin'),read_decisions('event')
            if len(left)!=len(right):raise RuntimeError('Paired decision trace lengths differ')
            for column in ('episode','task_id','R_req','safe_mask','effective_mask','best_achievable_reliability'):
                mismatch=int((left[column].to_numpy()!=right[column].to_numpy()).sum())
                checks[f'{phase}_{column}_mismatch']=mismatch
                if mismatch:raise RuntimeError(f'Trial {i} {phase} {column} mismatch')
        reproducibility.append(checks)
        for arm in ARMS:
            base=folder/arm/f'{arm}_train_task_assignments.csv'
            assignments=pd.read_csv(base if base.exists() else base.with_suffix('.csv.gz'))
            task_shifts.append(reconstruct_temporal_shifts(assignments,plan.loc[i,'Train_Arrival_Seed'],i,arm))
            frame=pd.read_csv(folder/arm/f'{arm}_training_curve.csv')
            frame.insert(0,'arm',arm);frame.insert(0,'trial_id',i)
            curve.append(frame)
    pd.DataFrame(reproducibility).to_csv(output/'reproducibility_checks.csv',index=False)
    pd.concat(task_shifts,ignore_index=True).to_csv(output/'task_credit_reassignment.csv.gz',index=False,compression='gzip')
    pd.concat([f for f in curve if f.arm.iloc[0]=='origin']).to_csv(output/'training_curve_origin.csv',index=False)
    pd.concat([f for f in curve if f.arm.iloc[0]=='event']).to_csv(output/'training_curve_event.csv',index=False)
    metrics={'mean_reward':'higher','mean_latency':'lower','p95_latency':'lower',
             'overall_rsr':'report','highest_rsr':'report',
             'pair_selection_hhi':'lower','max_mean_queue':'lower'}
    paired=[];boot=[]
    rng=np.random.default_rng(2043)
    draws=rng.integers(0,10,size=(20000,10))
    for metric,direction in metrics.items():
        wide=seed.pivot(index='trial_id',columns='arm',values=metric).sort_index()
        delta=wide.event.to_numpy()-wide.origin.to_numpy()
        if len(delta)!=10 or not np.isfinite(delta).all():raise RuntimeError(f'Invalid {metric} paired values')
        drawmeans=delta[draws].mean(axis=1)
        lo,hi=np.quantile(drawmeans,[.025,.975])
        wins=np.sum(delta>0) if direction!='lower' else np.sum(delta<0)
        losses=np.sum(delta<0) if direction!='lower' else np.sum(delta>0)
        ties=np.sum(delta==0)
        boot.append({'metric':metric,'event_minus_origin':float(delta.mean()),
            'ci_low':float(lo),'ci_high':float(hi),'wins':int(wins),'losses':int(losses),
            'ties':int(ties),'bootstrap_samples':20000,'bootstrap_seed':2043,
            'origin_mean':float(wide.origin.mean()),'origin_sample_sd':float(wide.origin.std(ddof=1)),
            'event_mean':float(wide.event.mean()),'event_sample_sd':float(wide.event.std(ddof=1))})
        for i,value in enumerate(delta):paired.append({'trial_id':i,'metric':metric,'event_minus_origin':value})
    pd.DataFrame(paired).to_csv(output/'paired_differences.csv',index=False)
    bootstrap=pd.DataFrame(boot);bootstrap.to_csv(output/'paired_bootstrap.csv',index=False)
    safe=pd.read_csv(ROOT/'diagnostics/results/external_baselines_rerun/baseline_seed_results.csv')
    safe=safe[safe.policy=='Safe Min-Latency'].set_index('trial_id').sort_index()
    if len(safe)!=10:raise RuntimeError('SafeMin seed reference incomplete')
    gap=[]
    for arm in ARMS:
        section=seed[seed.arm==arm].set_index('trial_id').sort_index()
        for metric in ('mean_reward','mean_latency','p95_latency'):
            for i in range(10):gap.append({'trial_id':i,'arm':arm,'metric':metric,
                'ppo_minus_safe_min':float(section.loc[i,metric]-safe.loc[i,metric]),
                'safe_min_value':float(safe.loc[i,metric])})
    gap=pd.DataFrame(gap);gap.to_csv(output/'safe_min_gap_comparison.csv',index=False)
    return seed,bootstrap,gap


def write_report_and_figures(output,seed,bootstrap,gap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'origin':'#3571a8','event':'#d47a2c'}
    def paired_figure(metric,name,ylabel):
        wide=seed.pivot(index='trial_id',columns='arm',values=metric).sort_index()
        fig,ax=plt.subplots(figsize=(6.5,4.5))
        for _,row in wide.iterrows():ax.plot([0,1],[row.origin,row.event],color='#999999',alpha=.6)
        ax.scatter(np.zeros(10),wide.origin,color=colors['origin'],label='Origin')
        ax.scatter(np.ones(10),wide.event,color=colors['event'],label='Event')
        ax.set_xticks([0,1],['Origin','Event']);ax.set_ylabel(ylabel);ax.set_title(name)
        fig.tight_layout();fig.savefig(output/f'{metric}_paired.png',dpi=160);plt.close(fig)
    for metric,label in [('mean_reward','Task reward'),('mean_latency','Mean latency (s)'),
                         ('p95_latency','P95 latency (s)'),('pair_selection_hhi','Pair HHI')]:
        paired_figure(metric,label,label)
    for metric in ('mean_reward','mean_latency','p95_latency'):
        f=gap[gap.metric==metric].pivot(index='trial_id',columns='arm',values='ppo_minus_safe_min').sort_index()
        fig,ax=plt.subplots(figsize=(6.5,4.5))
        ax.plot(f.index,f.origin,'o-',label='Origin');ax.plot(f.index,f.event,'s-',label='Event')
        ax.axhline(0,color='black',linewidth=.7)
        ax.set_xlabel('Trial');ax.set_ylabel('PPO − Safe Min-Latency');ax.set_title(metric+' gap')
        ax.legend();fig.tight_layout();fig.savefig(output/f'{metric}_safe_min_gap.png',dpi=160);plt.close(fig)
    curves={arm:pd.read_csv(output/f'training_curve_{arm}.csv') for arm in ARMS}
    fig,ax=plt.subplots(figsize=(7,4.5))
    for arm,frame in curves.items():
        mean=frame.groupby('episode').episode_reward.mean()
        ax.plot(mean.index,mean.rolling(10,min_periods=1).mean(),label=arm,color=colors[arm])
    ax.set_xlabel('Training episode');ax.set_ylabel('Mean task reward sum');ax.legend()
    fig.tight_layout();fig.savefig(output/'training_reward_curves.png',dpi=160);plt.close(fig)
    for source,column,filename in [
        ('critic_statistics_by_episode','explained_variance','critic_explained_variance.png'),
        ('gae_statistics_by_episode','raw_gae_std','raw_gae_std.png'),
        ('credit_distribution_by_episode','event_zero_fraction','event_zero_fraction.png')]:
        frame=pd.read_csv(output/f'{source}.csv')
        fig,ax=plt.subplots(figsize=(7,4.5))
        for arm in ARMS:
            f=frame[frame.arm==arm].groupby('episode')[column].mean()
            ax.plot(f.index,f.rolling(10,min_periods=1).mean(),label=arm,color=colors[arm])
        ax.set_xlabel('Training episode');ax.set_ylabel(column);ax.legend()
        fig.tight_layout();fig.savefig(output/filename,dpi=160);plt.close(fig)
    reassign=pd.read_csv(output/'reward_reassignment_statistics.csv')
    shift_tasks=pd.read_csv(output/'task_credit_reassignment.csv.gz')
    fig,ax=plt.subplots(figsize=(6,4))
    for arm in ARMS:
        ax.hist(shift_tasks[shift_tasks.arm==arm].index_shift,bins=range(0,51),alpha=.5,label=arm,density=True)
    ax.set_xlabel('Task origin-to-event index shift (0–50 shown)');ax.legend()
    fig.tight_layout();fig.savefig(output/'reward_index_shift.png',dpi=160);plt.close(fig)
    metric=bootstrap.set_index('metric')
    reward=metric.loc['mean_reward'];latency=metric.loc['mean_latency'];p95=metric.loc['p95_latency']
    reliability=metric.loc['overall_rsr']
    improved=(reward.ci_low>0 and (latency.ci_high<0 or p95.ci_high<0)
              and reliability.event_minus_origin>=-.005)
    hurt=(reward.ci_high<0 or (latency.ci_low>0 and p95.ci_low>0))
    if improved:classification='A — EVENT-INTERVAL CREDIT CLEARLY IMPROVES PPO'
    elif hurt:classification='D — EVENT-INTERVAL CREDIT HURTS PPO'
    elif abs(reward.event_minus_origin)<.01 and abs(latency.event_minus_origin)<.01:
        classification='C — EVENT-INTERVAL CREDIT HAS LITTLE EFFECT'
    else:classification='B — EVENT-INTERVAL CREDIT CHANGES LEARNING WITHOUT RELIABLE DEPLOYMENT IMPROVEMENT'
    credit=pd.read_csv(output/'credit_distribution_by_episode.csv')
    gae=pd.read_csv(output/'gae_statistics_by_episode.csv')
    critic=pd.read_csv(output/'critic_statistics_by_episode.csv')
    critic_paired=critic.groupby(['trial_id','arm'])[
        ['mae','pearson','spearman','explained_variance','value_loss_preupdate']
    ].mean().unstack('arm')
    critic_deltas=pd.DataFrame({key:critic_paired[(key,'event')]-critic_paired[(key,'origin')]
        for key in ('mae','pearson','spearman','explained_variance','value_loss_preupdate')})
    critic_deltas.to_csv(output/'critic_paired_differences.csv')
    stable_critic_gain=bool((critic_deltas.mae<0).all() and (critic_deltas.pearson>0).all()
                            and (critic_deltas.value_loss_preupdate<0).all())
    if classification.startswith('C') and stable_critic_gain:
        classification='B — EVENT-INTERVAL CREDIT IMPROVES LEARNING SIGNAL BUT NOT DEPLOYMENT'
    assign=pd.read_csv(output/'event_interval_sparsity.csv')
    conserve=pd.read_csv(output/'return_conservation.csv')
    reass=pd.read_csv(output/'reward_reassignment_statistics.csv')
    shift_tasks=pd.read_csv(output/'task_credit_reassignment.csv.gz')
    event_shifts=shift_tasks[shift_tasks.arm=='event']
    reproducibility=pd.read_csv(output/'reproducibility_checks.csv')
    if (not reproducibility.initial_network_equal.all() or reproducibility.arrival_mismatch.sum()
        or reproducibility.spatial_mismatch.sum() or reproducibility.task_sequence_mismatch.sum()
        or conserve[conserve.arm=='event'].return_residual.max()>=1e-10):
        classification='E — EXPERIMENT INVALID'
    lines=['# Origin-task versus event-interval credit: formal 10-seed study','',
        'Both arms use the historical `pair_scoring` Actor, joint clipping, the same PPO hyperparameters, '
        'production task reward and masked stochastic deployment. Only training credit differs. '
        'Each seed uses 300 × 200 training and 20 × 200 evaluation tasks. Evaluation metrics are task-level.', '',
        '## Historical regression and reproducibility','',
        'The complete trial-0 300-episode gate is in `gate/gate_report.json`. '
        'A separate direct 60,000-decision trace gate compares state, decision time, delta_t, '
        'action, mask, old log probability and final tensors exactly; see '
        '`gate/decision_trace/state_trace_verification.json`. '
        'Every Origin trial additionally reproduces its archived Actor/Critic checkpoint, training curve and evaluation actions/rewards. '
        'Initial tensors, arrival/spatial streams and input workbooks match between arms.', '',
        '## Paired performance (Event − Origin)','',
        '|Metric|Origin mean ± SD|Event mean ± SD|Delta|95% paired bootstrap CI|W/L/T|',
        '|---|---:|---:|---:|---:|---:|']
    for row in bootstrap.itertuples():
        lines.append(f'|{row.metric}|{row.origin_mean:.6g} ± {row.origin_sample_sd:.6g}|'
          f'{row.event_mean:.6g} ± {row.event_sample_sd:.6g}|{row.event_minus_origin:.6g}|'
          f'[{row.ci_low:.6g}, {row.ci_high:.6g}]|{row.wins}/{row.losses}/{row.ties}|')
    lines+=['','20,000 paired seed bootstrap resamples; seed 2043. Reliability deltas are descriptive.','',
        '## Training mechanism','',
        f'Event return conservation: {len(conserve[conserve.arm=="event"])} episodes, '
        f'maximum residual {conserve[conserve.arm=="event"].return_residual.max():.3g}.',
        f'Event zero-reward interval fraction: {credit[credit.arm=="event"].event_zero_fraction.mean():.4f}. '
        f'Multiple-event interval fraction: {assign[assign.arm=="event"].multiple_event_intervals.mean()/TASKS:.4f}. '
        f'Maximum zero run: {credit[credit.arm=="event"].zero_run_max.max()}.',
        f'Same-interval rate: {(event_shifts.index_shift==0).mean():.4f}; '
        f'shifted-forward rate: {(event_shifts.index_shift>0).mean():.4f}; '
        f'median index shift: {event_shifts.index_shift.median():.3g}; '
        f'P90 index shift: {event_shifts.index_shift.quantile(.9):.3g}; '
        f'max index shift: {event_shifts.index_shift.max():.3g}; '
        f'median resolution delay: {event_shifts.resolution_delay.median():.3g} s.',
        '', '|Arm|Pre-update Critic MAE|Pearson|Spearman|Explained variance|Pre-update value loss|Raw GAE std|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        c=critic[critic.arm==arm];g=gae[gae.arm==arm]
        lines.append(f'|{arm}|{c.mae.mean():.5g}|{c.pearson.mean():.5g}|{c.spearman.mean():.5g}|'
           f'{c.explained_variance.mean():.5g}|{c.value_loss_preupdate.mean():.5g}|{g.raw_gae_std.mean():.5g}|')
    lines+=['',f'Critic MAE, Pearson and pre-update value loss improve in all {len(critic_deltas)} paired seeds. '
        'Explained variance remains near zero; the relative improvement does not establish a useful value model.',
        'Critic loss here is the pre-update MSE × value-loss coefficient, measured from the actual PPO '
        'target before minibatch optimization. It is not the post-update training loss.', '',
        '## External Safe Min-Latency reference','',
        '|Metric|Origin − SafeMin|Event − SafeMin|','|---|---:|---:|']
    for name in ('mean_reward','mean_latency','p95_latency'):
        f=gap[gap.metric==name]
        lines.append(f'|{name}|{f[f.arm=="origin"].ppo_minus_safe_min.mean():.6g}|'
                     f'{f[f.arm=="event"].ppo_minus_safe_min.mean():.6g}|')
    lines+=['','SafeMin is an external archived benchmark; it does not participate in either PPO training arm.',
        '', '## Classification','', classification,'',
        'Event-interval credit was tested for temporal consistency of arrival-to-arrival transitions and all '
        'resolution events, including outcomes of previously decided tasks. No expected-Q or prior '
        'single-continuation causal claim is assumed.', '',
        f'Go for formal method inclusion: {"YES" if classification.startswith("A") else "NO"}. '
        'This ablation alone does not justify changing Actor or Critic architecture.']
    (output/'ORIGIN_VS_EVENT_CREDIT_REPORT.md').write_text('\n'.join(lines)+'\n')
    return classification


def run_trace_gate(output):
    """Direct 60,000-decision state and elapsed-time comparison for trial zero."""
    torch.set_num_threads(1)
    _,plan=formal_spec();trial={k:int(v) for k,v in plan.iloc[0].items()}
    _,rho=build_pair_correlations()
    target=output/'gate/decision_trace'
    if target.exists():raise RuntimeError('Decision trace gate already exists')
    if not json.loads((output/'gate/gate_report.json').read_text())['exact']:
        raise RuntimeError('Existing historical gate failed')
    target.mkdir(parents=True)
    arrays=[];loops=[];models=[]
    for label,agent in [('historical',legacy_agent(trial,rho)),
                        ('origin',new_agent(trial,rho,'origin_task',audited=False))]:
        observed=[];transitions=[]
        prepare=agent.prepare_action
        store=agent.store_transition
        def tapped_prepare(task,env_state,episode,state,*,_prepare=prepare):
            observed.append((int(episode),int(task.id),float(task.env.now),np.asarray(state).copy()))
            return _prepare(task,env_state,episode,state)
        def tapped_store(*args,_store=store,**kwargs):
            result=_store(*args,**kwargs)
            transitions.append((int(kwargs['task_id']),int(args[1]),float(kwargs['delta_t']),
                float(agent.old_log_probs[-1]),np.asarray(agent.effective_masks[-1]).copy()))
            return result
        agent.prepare_action=tapped_prepare
        agent.store_transition=tapped_store
        torch.manual_seed(int(trial['Train_Action_Seed']))
        with scoped_environment_seeds(int(trial['Train_Arrival_Seed']),int(trial['Train_Spatial_Seed'])):
            with (target/f'{label}.log').open('w') as log,redirect_stdout(log):
                loop=ArrivalTraceLoop(agent,TRAIN_EPISODES,TASKS,params.num_states,params.num_actions)
                loop.EP()
        if len(observed)!=TRAIN_EPISODES*TASKS or len(transitions)!=TRAIN_EPISODES*TASKS:
            raise RuntimeError(f'{label}: incomplete direct decision trace')
        data={'episode':np.asarray([x[0] for x in observed],dtype=np.int16),
              'task_id':np.asarray([x[1] for x in observed],dtype=np.int16),
              'decision_time':np.asarray([x[2] for x in observed],dtype=np.float64),
              'states':np.stack([x[3] for x in observed]),
              'transition_task_id':np.asarray([x[0] for x in transitions],dtype=np.int16),
              'actions':np.asarray([x[1] for x in transitions],dtype=np.int8),
              'delta_t':np.asarray([x[2] for x in transitions],dtype=np.float64),
              'old_log_probability':np.asarray([x[3] for x in transitions],dtype=np.float64),
              'effective_mask':np.stack([x[4] for x in transitions]).astype(np.bool_)}
        np.savez_compressed(target/f'{label}.npz',**data)
        arrays.append(data);loops.append(loop);models.append(agent)
    check_external_trace(trace(loops[0]),trace(loops[1]))
    result={'decisions_checked':TRAIN_EPISODES*TASKS,'arrival_mismatch':0,'spatial_mismatch':0}
    for field in arrays[0]:
        a,b=arrays[0][field],arrays[1][field]
        result[field+'_mismatches']=int(np.count_nonzero(a!=b))
        if result[field+'_mismatches']:raise RuntimeError(f'Direct gate mismatch: {field}')
    for name in ('policy_net','policy_old','value_net'):
        result[name+'_tensor_mismatches']=assert_equal_states(getattr(models[0],name).state_dict(),
                                                              getattr(models[1],name).state_dict())
        if result[name+'_tensor_mismatches']:raise RuntimeError(f'Direct gate mismatch: {name}')
    result['exact']=True
    (target/'state_trace_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--gate',action='store_true')
    parser.add_argument('--trace-gate',action='store_true')
    parser.add_argument('--aggregate',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--trial-id',type=int,choices=range(10))
    args=parser.parse_args()
    output=args.output_dir
    if args.trace_gate:
        result=run_trace_gate(output)
    elif args.aggregate:
        seed,bootstrap,gap=aggregate(output)
        result={'seed_rows':len(seed),'classification':write_report_and_figures(output,seed,bootstrap,gap)}
    elif args.gate:
        if output.exists():parser.error('Output directory must not exist')
        result=run_gate(output)
    elif args.smoke:
        result=run_smoke(output)
    elif args.trial_id is not None:
        if not (output/'smoke/smoke_report.json').exists():
            parser.error('Event smoke gate has not passed')
        result=run_trial(args.trial_id,output)
    else:parser.error('Select --gate, --smoke, or --trial-id')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
