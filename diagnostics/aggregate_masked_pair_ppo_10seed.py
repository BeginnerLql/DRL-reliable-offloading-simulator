"""Aggregate the paired formal 10-seed masked Pair PPO experiment.

The seed, not the task, is the statistical unit. No model is trained here.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.params import params
from diagnostics.run_masked_pair_ppo_10seed import OUT, PRIOR, formal_spec

POLICIES=('flat','pair','masked_stochastic','masked_greedy')
TIERS=(.9,.99,.999,.9999)
BOOTSTRAP_SEED=2043
BOOTSTRAP_SAMPLES=20000
TIE_TOLERANCE=1e-10


def read_trial(trial):
    base=OUT/'runs'/f'trial_{trial:03d}'
    if not (base/'completed.json').exists():
        raise RuntimeError(f'Trial {trial} is incomplete')
    completed=json.loads((base/'completed.json').read_text())
    runs={row['policy']:row for row in completed['runs']}
    if set(runs)!=set(POLICIES):raise RuntimeError(f'Trial {trial} missing deployment')
    for key in ('actor_sha256','critic_sha256'):
        if runs['masked_stochastic'][key]!=runs['masked_greedy'][key]:
            raise RuntimeError(f'Trial {trial} masked ablations use different checkpoint')
    return base


def metrics(assignments,decisions):
    if len(assignments)!=len(decisions) or len(assignments)==0:
        raise RuntimeError('Evaluation rows missing or not paired')
    n=len(assignments)
    a=assignments.sort_values(['episode','task_id']).reset_index(drop=True)
    d=decisions.sort_values(['episode','task_id']).reset_index(drop=True)
    if not np.array_equal(a.action_index,d.action_index):
        raise RuntimeError('Decision and outcome actions differ')
    satisfied=a.Reliability_Satisfied.astype(bool).to_numpy()
    safe=~d.safe_set_empty.astype(bool).to_numpy()
    selected_safe=d.selected_action_safe.astype(bool).to_numpy()
    if not np.array_equal(satisfied,selected_safe):
        raise RuntimeError('Selected pair reliability disagrees with simulator')
    probabilities=a.action_index.value_counts().reindex(range(params.num_actions),fill_value=0).to_numpy(dtype=float)/n
    nonzero=probabilities[probabilities>0]
    deficits=d.loc[~safe,'reliability_deficit'].to_numpy(dtype=float)
    return {'tasks':n,'mean_reward':float(a.Task_Reward.mean()),
            'mean_latency':float(a.Task_Delay.mean()),
            'p50_latency':float(a.Task_Delay.quantile(.5)),
            'p90_latency':float(a.Task_Delay.quantile(.9)),
            'p95_latency':float(a.Task_Delay.quantile(.95)),
            'overall_rsr':float(satisfied.mean()),
            'feasibility_rate':float(safe.mean()),
            'conditional_rsr':float(satisfied[safe].mean()) if safe.any() else np.nan,
            'avoidable_violation_rate':float((safe & ~satisfied).mean()),
            'unavoidable_violation_rate':float((~safe & ~satisfied).mean()),
            'empty_safe_set_rate':float((~safe).mean()),
            'mean_empty_deficit':float(deficits.mean()) if len(deficits) else 0.0,
            'p95_empty_deficit':float(np.quantile(deficits,.95)) if len(deficits) else 0.0,
            'avoidable_violation_count':int((safe & ~satisfied).sum()),
            'unavoidable_violation_count':int((~safe & ~satisfied).sum()),
            'mean_safe_set_size':float(d.safe_set_size.mean()),
            'pair_selection_hhi':float(np.sum(probabilities**2)),
            'pair_selection_entropy':float(-(nonzero*np.log(nonzero)).sum()),
            'top1_pair_frequency':float(probabilities.max()),
            'unique_selected_pairs':int(np.count_nonzero(probabilities)),
            'pair_78_frequency':float(probabilities[-1]),
            'mean_masked_entropy':float(d.masked_entropy.mean())}


def bootstrap_ci(deltas,seed=BOOTSTRAP_SEED,samples=BOOTSTRAP_SAMPLES):
    values=np.asarray(deltas,dtype=float)
    if len(values)!=10 or not np.isfinite(values).all():
        raise RuntimeError('Bootstrap requires 10 complete finite seed deltas')
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,10,size=(samples,10))
    means=values[indices].mean(axis=1)
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def compile_tables():
    formal_spec()
    rows=[];tiers=[];loads=[];pair_rows=[];training=[];episode_training=[]
    for trial in range(10):
        base=read_trial(trial)
        stochastic_masks=None
        for policy in POLICIES:
            sub=base/policy
            a=pd.read_csv(sub/'evaluation_task_assignments.csv')
            d=pd.read_csv(sub/'evaluation_decisions.csv')
            load=pd.read_csv(sub/'server_load.csv')
            if len(a)!=4000 or len(d)!=4000 or len(load)!=params.serverNo:
                raise RuntimeError(f'Trial {trial} {policy} evaluation incomplete')
            if policy=='masked_stochastic':
                stochastic_masks=d.sort_values(['episode','task_id']).safe_mask.tolist()
            if policy=='masked_greedy':
                if stochastic_masks!=d.sort_values(['episode','task_id']).safe_mask.tolist():
                    raise RuntimeError(f'Trial {trial}: stochastic/greedy safe masks differ')
            row={'trial_id':trial,'policy':policy,**metrics(a,d)}
            row['maximum_server_selection_share']=float(load.selection_count.max()/(2*len(a)))
            row['maximum_server_utilization']=float(load.utilization.max())
            row['maximum_mean_queue_length']=float(load.mean_queue_length.max())
            row['maximum_p95_queue_length']=float(load.p95_queue_length.max())
            row['mean_server_queue_length']=float(load.mean_queue_length.mean())
            row['mean_server_p95_queue_length']=float(load.p95_queue_length.mean())
            row['mean_server_waiting_time']=float(load.mean_waiting_time.mean())
            rows.append(row)
            counts=a.action_index.value_counts().reindex(range(params.num_actions),fill_value=0)
            for action_index,count in counts.items():
                pair_rows.append({'trial_id':trial,'policy':policy,'action_index':int(action_index),
                                  'selection_count':int(count),'selection_frequency':float(count/len(a))})
            load.insert(0,'trial_id',trial);loads.append(load)
            for requirement in TIERS:
                which=np.isclose(a.Reliability_Requirement.to_numpy(dtype=float),requirement)
                if which.sum()!=1000:
                    raise RuntimeError(f'Trial {trial}: requirement tier not balanced')
                tiers.append({'trial_id':trial,'policy':policy,'R_req':requirement,
                              **metrics(a.loc[which].reset_index(drop=True),d.loc[which].reset_index(drop=True))})
            if policy.startswith('masked'):
                if row['avoidable_violation_count'] or row['conditional_rsr'] < 1-1e-10:
                    raise RuntimeError(f'Trial {trial}: masked action feasibility violation')
        for train_name in ('flat','pair','masked'):
            sub=base/train_name
            curve=pd.read_csv(sub/'training_curve.csv')
            if len(curve)!=300 or not np.isfinite(curve.select_dtypes(include=[np.number]).to_numpy()).all():
                raise RuntimeError(f'Trial {trial} {train_name} training curve invalid')
            training.append({'trial_id':trial,'policy':train_name,
                             'first30_mean_episode_reward':float(curve.episode_reward.head(30).mean()),
                             'last30_mean_episode_reward':float(curve.episode_reward.tail(30).mean()),
                             'final_episode_reward':float(curve.episode_reward.iloc[-1])})
            curve.insert(0,'policy',train_name);curve.insert(0,'trial_id',trial)
            if train_name=='masked':
                ep=pd.read_csv(sub/'training_episode_metrics.csv')
                updates=pd.read_csv(sub/'ppo_updates.csv')
                if len(ep)!=300 or len(updates)!=300:
                    raise RuntimeError(f'Trial {trial}: masked training diagnostics invalid')
                if not np.isfinite(ep.select_dtypes(include=[np.number]).to_numpy()).all():
                    raise RuntimeError('Masked training contains NaN/Inf')
                training[-1].update({'first30_masked_entropy':float(ep.mean_masked_entropy.head(30).mean()),
                    'last30_masked_entropy':float(ep.mean_masked_entropy.tail(30).mean()),
                    'first30_pair_hhi':float(ep.pair_selection_hhi.head(30).mean()),
                    'last30_pair_hhi':float(ep.pair_selection_hhi.tail(30).mean()),
                    'final30_top1_frequency':float(ep.top1_pair_frequency.tail(30).mean()),
                    'mean_safe_set_size':float(ep.mean_safe_set_size.mean()),
                    'empty_safe_set_rate':float(ep.empty_safe_set_rate.mean()),
                    'ratio_min':float(updates.ratio_min.min()),
                    'ratio_max':float(updates.ratio_max.max()),
                    'max_first_ratio_error':float(updates.first_minibatch_ratio_max_abs_error.max())})
                ep.insert(0,'trial_id',trial);episode_training.append(ep)
    result=pd.DataFrame(rows).sort_values(['trial_id','policy']).reset_index(drop=True)
    tier=pd.DataFrame(tiers).sort_values(['trial_id','policy','R_req']).reset_index(drop=True)
    return result,tier,pd.concat(loads,ignore_index=True),pd.DataFrame(pair_rows),pd.DataFrame(training),pd.concat(episode_training,ignore_index=True)


def paired_tables(result):
    metrics_to_compare=('mean_reward','mean_latency','overall_rsr','highest_rsr',
                        'pair_selection_hhi','pair_selection_entropy','top1_pair_frequency',
                        'maximum_server_selection_share','maximum_server_utilization',
                        'maximum_mean_queue_length','maximum_p95_queue_length',
                        'mean_server_queue_length','mean_server_p95_queue_length',
                        'avoidable_violation_rate','conditional_rsr')
    comparisons=[]
    for label,left,right in [('masked_minus_pair','masked_stochastic','pair'),
                             ('stochastic_minus_greedy','masked_stochastic','masked_greedy')]:
        for trial in range(10):
            a=result[(result.trial_id==trial)&(result.policy==left)].iloc[0]
            b=result[(result.trial_id==trial)&(result.policy==right)].iloc[0]
            row={'comparison':label,'trial_id':trial}
            for metric in metrics_to_compare:
                row[f'delta_{metric}']=float(a[metric]-b[metric])
            comparisons.append(row)
    frame=pd.DataFrame(comparisons)
    cis=[];wlt=[]
    for label,group in frame.groupby('comparison',sort=False):
        for metric in metrics_to_compare:
            column=f'delta_{metric}'
            values=group.sort_values('trial_id')[column].to_numpy(dtype=float)
            low,high=bootstrap_ci(values)
            cis.append({'comparison':label,'metric':metric,'mean_delta':float(values.mean()),
                        'std_delta':float(values.std(ddof=1)),
                        'ci95_low':low,'ci95_high':high,
                        'ci_excludes_zero':bool(low>0 or high<0),
                        'bootstrap_samples':BOOTSTRAP_SAMPLES,'bootstrap_seed':BOOTSTRAP_SEED})
            smaller_is_better=metric in ('mean_latency','pair_selection_hhi','top1_pair_frequency',
                'maximum_server_selection_share','maximum_server_utilization',
                'maximum_mean_queue_length','maximum_p95_queue_length',
                'mean_server_queue_length','mean_server_p95_queue_length',
                'avoidable_violation_rate')
            beneficial=-values if smaller_is_better else values
            wlt.append({'comparison':label,'metric':metric,
                        'positive_seeds':int((beneficial>TIE_TOLERANCE).sum()),
                        'negative_seeds':int((beneficial<-TIE_TOLERANCE).sum()),
                        'ties':int((np.abs(beneficial)<=TIE_TOLERANCE).sum()),
                        'tie_tolerance':TIE_TOLERANCE,
                        'win_direction':'lower' if smaller_is_better else 'higher'})
    return frame,pd.DataFrame(cis),pd.DataFrame(wlt)


def plot_all(result,tier,comparisons):
    def series(policy,metric):
        return result[result.policy.eq(policy)].sort_values('trial_id')[metric].to_numpy(dtype=float)
    def plot_series(metric,filename,ylabel,policies=POLICIES):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for policy in policies:
            ax.plot(range(10),series(policy,metric),marker='o',label=policy)
        ax.set(xlabel='Paired seed / trial ID',ylabel=ylabel,xticks=range(10));ax.grid(alpha=.25)
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/filename,dpi=160);plt.close(fig)
    for metric,file,label in [('mean_reward','reward_by_seed.png','Mean task reward'),
                              ('mean_latency','latency_by_seed.png','Mean task latency (s)'),
                              ('overall_rsr','overall_rsr_by_seed.png','Overall RSR'),
                              ('highest_rsr','highest_rsr_by_seed.png','Highest R_req RSR'),
                              ('pair_selection_hhi','pair_hhi_by_seed.png','Pair selection HHI'),
                              ('maximum_mean_queue_length','max_server_load_by_seed.png','Maximum server mean queue length'),
                              ('mean_latency','stochastic_vs_greedy_latency.png','Mean task latency (s)'),
                              ('pair_selection_hhi','stochastic_vs_greedy_hhi.png','Pair selection HHI')]:
        plot_series(metric,file,label, policies=('masked_stochastic','masked_greedy') if file.startswith('stochastic_vs_greedy') else POLICIES)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for metric in ('feasibility_rate','conditional_rsr','overall_rsr'):
        ax.plot(range(10),series('masked_stochastic',metric),marker='o',label=metric)
    ax.set(xlabel='Paired seed / trial ID',ylabel='Rate',xticks=range(10),ylim=(0,1.02));ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'feasibility_and_conditional_rsr.png',dpi=160);plt.close(fig)
    paired=comparisons[comparisons.comparison.eq('masked_minus_pair')].sort_values('trial_id')
    for metric,file,label in [('mean_reward','paired_reward_delta.png','Masked − Pair reward'),
                              ('overall_rsr','paired_rsr_delta.png','Masked − Pair Overall RSR')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        ax.bar(range(10),paired[f'delta_{metric}']);ax.axhline(0,color='black',lw=.8)
        ax.set(xlabel='Paired seed / trial ID',ylabel=label,xticks=range(10));ax.grid(alpha=.25,axis='y')
        fig.tight_layout();fig.savefig(OUT/file,dpi=160);plt.close(fig)


def main():
    result,tier,loads,pair_rows,training,episode_training=compile_tables()
    highest=tier[np.isclose(tier.R_req,.9999)][['trial_id','policy','overall_rsr']].rename(columns={'overall_rsr':'highest_rsr'})
    result=result.merge(highest,on=['trial_id','policy'],validate='one_to_one')
    comparisons,cis,wlt=paired_tables(result)
    # Keep checkpoint, config, seed and both summaries together for every trained agent.
    for trial in range(10):
        base=OUT/'runs'/f'trial_{trial:03d}'
        seed_text=(base/'seed_metadata.json').read_text()
        for train_name,deployments in (('flat',('flat',)),('pair',('pair',)),('masked',('masked_stochastic','masked_greedy'))):
            sub=base/train_name
            (sub/'seed_metadata.json').write_text(seed_text)
            training[(training.trial_id==trial)&training.policy.eq(train_name)].to_csv(sub/'training_summary.csv',index=False)
            result[(result.trial_id==trial)&result.policy.isin(deployments)].to_csv(sub/'evaluation_summary.csv',index=False)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in [('raw_seed_results',result),('evaluation_seed_summary',result),
        ('requirement_level_results',tier),('server_load_results',loads),
        ('pair_selection_results',pair_rows),('training_seed_summary',training),
        ('training_episode_results',episode_training),('paired_comparisons',comparisons),
        ('bootstrap_ci',cis),('win_loss_tie',wlt)]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    plot_all(result,tier,comparisons)
    (OUT/'aggregation_metadata.json').write_text(json.dumps({
        'trials':10,'training_episodes':300,'evaluation_episodes':20,'tasks_per_episode':200,
        'bootstrap_resamples':BOOTSTRAP_SAMPLES,'bootstrap_seed':BOOTSTRAP_SEED,
        'tie_tolerance':TIE_TOLERANCE,'statistics_unit':'training seed',
        'legacy_reference':str(PRIOR.relative_to(ROOT)),
        'masked_stochastic_is_formal_deployment':True,'masked_greedy_ablation_uses_same_checkpoint':True},indent=2))
    print(result.groupby('policy')[['mean_reward','mean_latency','overall_rsr','highest_rsr','pair_selection_hhi']].agg(['mean','std']).to_string(),flush=True)
    return result,tier,cis,wlt

if __name__=='__main__':main()
