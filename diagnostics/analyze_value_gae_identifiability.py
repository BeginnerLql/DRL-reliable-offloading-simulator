"""Read-only analysis of frozen value/GAE audit artifacts; Q_H never trains V."""
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
import torch
from scipy.spatial import cKDTree
from diagnostics.value_gae_identifiability import *
from diagnostics.run_value_gae_identifiability import load_dataset,verify_protected
from agents.policy_centered_action_advantage import PolicyCenteredActionAdvantage,policy_centered_advantage


def stats(v):
    x=np.asarray(v,float)
    return dict(mean=float(x.mean()),std=float(x.std()),min=float(x.min()),max=float(x.max()),
                **{f'p{q:02d}':float(np.percentile(x,q)) for q in [1,5,25,50,75,95,99]})


def scale_and_noise(data,meta):
    rows=[];y=data['mc_return'];v=data['original_value'];ep=data['episode']
    groups={s:np.isin(ep,ids) for s,ids in meta['split'].items()}
    groups.update(early_position=data['task_ids']<=67,middle_position=(data['task_ids']>67)&(data['task_ids']<=134),late_position=data['task_ids']>134,all=np.ones(len(y),bool))
    for group,mask in groups.items():
        for label,x in [('MC_return',y),('PPO_V',v)]:
            rows.append(dict(group=group,quantity=label,n=int(mask.sum()),**stats(x[mask]),
               std_v_over_g=float(v[mask].std()/y[mask].std()),mean_abs_v_over_g=float(np.abs(v[mask]).mean()/np.abs(y[mask]).mean())))
    pd.DataFrame(rows).to_csv(OUT/'value_target_statistics.csv',index=False)
    train,test=groups['train'],groups['test'];x=data['states'].astype(float)
    mu=x[train].mean(0);sd=x[train].std(0);active=sd>1e-8;z=(x[:,active]-mu[active])/sd[active]
    tree=cKDTree(z[train]);distance,indices=tree.query(z[test],k=20,workers=1)
    local=y[train][indices];variance=np.var(y[test]);ratio=float(np.var(local,axis=1).mean()/variance)
    noise=[dict(method='train-neighbor k20 for test states',n=int(test.sum()),conditional_variance_ratio=ratio,
                mean_neighbor_distance=float(distance.mean()),
                normalized_pair_difference_variance=float(np.mean((y[test,None]-local)**2)/(2*variance)),
                note='Approximate-state grouping, not an irreducible-noise estimate; train/test episodes disjoint')]
    # Coarse conditioning is descriptive; all cut points come from TRAIN.
    b=data['backlog'].max(1);c=data['task_parameters'][:,1]
    bc=np.quantile(b[train],[1/3,2/3]);cc=np.quantile(c[train],[1/3,2/3])
    frame=pd.DataFrame(dict(g=y[test],load=np.digitize(b[test],bc),demand=np.digitize(c[test],cc),req=data['requirement'][test]))
    conditional=frame.groupby(['load','demand','req']).g.agg(['var','count'])
    coarse=float((conditional['var'].fillna(0)*conditional['count']).sum()/conditional['count'].sum()/variance)
    noise.append(dict(method='test load/demand/R_req coarse bins',n=int(test.sum()),conditional_variance_ratio=coarse,
                      mean_neighbor_distance=np.nan,normalized_pair_difference_variance=np.nan,note='TRAIN tercile cutpoints; bins are approximate states'))
    pd.DataFrame(noise).to_csv(OUT/'return_noise_diagnostics.csv',index=False)


def agreement(signal,reference):
    x,y=np.asarray(signal,float),np.asarray(reference,float);nonzero=np.abs(y)>1e-8
    n=int(nonzero.sum())
    # Unit-free normalized MAE; diagnostic normalization is not used by learners.
    zx=(x-x.mean())/(x.std()+1e-8);zy=(y-y.mean())/(y.std()+1e-8)
    quartile=lambda a:np.digitize(a,np.quantile(a,[.25,.5,.75]))
    return dict(n=len(x),nonzero_reference_count=n,pearson=correlation(x,y),spearman=correlation(x,y,True),
                sign_agreement=float(np.mean(np.sign(x[nonzero])==np.sign(y[nonzero]))) if n else np.nan,
                normalized_mae=float(np.abs(zx-zy).mean()),quartile_agreement=float(np.mean(quartile(x)==quartile(y))),
                sign_agreement_including_zeros=float(np.mean(np.sign(x)==np.where(nonzero,np.sign(y),0.))))


def matched_analysis(data):
    q=pd.read_csv(OUT/'matched_counterfactual_qh.csv')
    manifest=pd.read_csv(OUT/'matched_state_manifest.csv')
    if q.state_id.nunique()!=1000:raise RuntimeError('Require 1000 matched states')
    if not np.isfinite(q[[f'{prefix}q_h{h}' for prefix in ('','legacy_') for h in HORIZONS]].to_numpy()).all():raise RuntimeError('Nonfinite counterfactual return')
    if q.duplicated(['state_id','action_index']).any():raise RuntimeError('Duplicate counterfactual branch')
    references=[];selected=[];selected_return_checks=[]
    with np.load(OUT/'offline_gae.npz') as archive:gaes={k:archive[k] for k in archive.files}
    pairs,rho=build_pair_correlations();learner=PolicyCenteredActionAdvantage(params.num_states,params.num_actions,params.hidden_layers_ppo,params.serverNo,rho,activation=params.af_ppo)
    weight=ROOT/'diagnostics/results/advantage_side_learner/policy_centered_advantage.pt'
    learner.network.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));learner.eval();before=model_hash(learner.network)
    ranking=[]
    for sid,frame in q.groupby('state_id',sort=True):
        frame=frame.sort_values('action_index');ids=frame.action_index.to_numpy(int);index=int(frame.dataset_index.iloc[0]);chosen=int(data['actions'][index])
        if len(frame)!=int(data['effective_mask'][index].sum()) or set(ids)!=set(np.flatnonzero(data['effective_mask'][index])):raise RuntimeError('Candidate action support changed')
        if not np.array_equal(frame.probability.to_numpy(),data['probabilities'][index,ids].astype(float)):
            if not np.allclose(frame.probability.to_numpy(),data['probabilities'][index,ids],rtol=0,atol=1e-15):raise RuntimeError('Saved probabilities changed')
        p=frame.probability.to_numpy();p=p/p.sum()
        selected_local=int(np.flatnonzero(ids==chosen)[0])
        # The branch using the recorded action must recover its original rollout return.
        for h in HORIZONS:
            end=min(index+h,(index//200+1)*200)
            weights=np.r_[1.,np.cumprod(params.gamma_ppo**data['delta_t'][index:end-1])]
            expected=float(weights@data['rewards'][index:end])
            residual=abs(float(frame.iloc[selected_local][f'q_h{h}'])-expected)
            if residual>1e-9:raise RuntimeError('Selected branch differs from original trajectory return')
            selected_return_checks.append(dict(state_id=sid,horizon=h,selected_branch_return_residual=residual))
        myopic_local=int(np.flatnonzero(ids==int(frame.myopic_action.iloc[0]))[0])
        common={k:frame.iloc[0][k] for k in ['state_id','episode','task_id','dataset_index','load','requirement','safe_set_bin','safe_set_size','effective_set_size']}
        with torch.no_grad():
            raw=learner.network(torch.tensor(data['states'][index:index+1],dtype=torch.float32))
            centered=policy_centered_advantage(raw,torch.tensor(data['probabilities'][index:index+1]),torch.tensor(data['effective_mask'][index:index+1])).numpy()[0,ids]
        for lens,prefix in [('production_smdp',''),('legacy_interval_step','legacy_')]:
            h20=frame[prefix+'q_h20'].to_numpy();regret=float(h20.max()-h20[myopic_local]);coupling='coupled' if regret>.01 else 'easy'
            for h in HORIZONS:
                values=frame[f'{prefix}q_h{h}'].to_numpy();ref=values-float(p@values)
                residual=float(abs(p@ref))
                if residual>1e-9:raise RuntimeError('Counterfactual centering error')
                for a,prob,val,relative in zip(ids,p,values,ref):
                    references.append({**common,'lens':lens,'horizon':h,'action_index':int(a),'probability':prob,
                                       'q_return':val,'reference_advantage':relative,'centering_residual':residual,
                                       'coupling':coupling,'myopic_h20_regret':regret})
                selected.append({**common,'lens':lens,'horizon':h,'coupling':coupling,'reference_advantage':ref[selected_local],
                                 'frozen_centered_advantage':float(centered[selected_local]),
                                 **{k:float(v[index]) for k,v in gaes.items()}})
                ranking.append({**common,'lens':lens,'horizon':h,'coupling':coupling,
                                'frozen_advantage_spearman':correlation(centered,ref,True),
                                'frozen_advantage_regret':float(values.max()-values[np.argmax(centered)])})
    if before!=model_hash(learner.network):raise RuntimeError('Frozen side learner changed')
    pd.DataFrame(selected_return_checks).to_csv(OUT/'selected_branch_return_checks.csv',index=False)
    pd.DataFrame(references).to_csv(OUT/'matched_centered_advantage_reference.csv',index=False)
    selected=pd.DataFrame(selected);selected.to_csv(OUT/'matched_selected_action_signals.csv',index=False)
    pd.DataFrame(ranking).to_csv(OUT/'frozen_centered_advantage_alignment.csv',index=False)
    learner_agreement=[]
    for (lens,h),frame in selected.groupby(['lens','horizon']):
        learner_agreement.append(dict(lens=lens,horizon=h,**agreement(frame.frozen_centered_advantage,frame.reference_advantage)))
    pd.DataFrame(learner_agreement).to_csv(OUT/'frozen_centered_advantage_selected_alignment.csv',index=False)
    for filename,extra in [('gae_vs_counterfactual_advantage.csv',[]),('gae_vs_counterfactual_by_coupling.csv',['coupling']),
        ('gae_vs_counterfactual_by_load.csv',['load']),('gae_vs_counterfactual_by_requirement.csv',['requirement']),
        ('gae_vs_counterfactual_by_safe_set.csv',['safe_set_bin'])]:
        rows=[]
        for key,frame in selected.groupby(['lens','horizon']+extra,dropna=False):
            context=dict(zip(['lens','horizon']+extra,key))
            for arm in ['V0','V1','V2','V3']:
                for kind,suffix in [('raw',''),('normalized','_normalized')]:
                    rows.append({**context,'value_source':arm,'advantage_type':kind,
                                 **agreement(frame[arm+suffix],frame.reference_advantage)})
        pd.DataFrame(rows).to_csv(OUT/filename,index=False)
    return selected


def make_plots(data,selected,best):
    with np.load(OUT/'value_predictions.npz') as z:preds={k:z[k] for k in z.files}
    meta=json.loads((OUT/'frozen_policy_dataset_manifest.json').read_text());test=np.isin(data['episode'],meta['split']['test'])
    y=data['mc_return'][test];sample=np.arange(0,len(y),5)
    def save(fig,name):fig.tight_layout();fig.savefig(OUT/name,dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.scatter(y[sample],preds['V0'][test][sample],s=5,alpha=.25)
    ax.set(xlabel='Realized natural-episode MC return',ylabel='Original PPO V',title='Frozen PPO value scale');save(fig,'ppo_v_vs_mc_return.png')
    validation=np.isin(data['episode'],meta['split']['validation'])
    calibration_y=data['mc_return'][validation];calibration_sample=np.arange(0,len(calibration_y),5)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,arm in zip(axes,['V1','V2','V3']):
        p=preds[arm][validation];ax.scatter(calibration_y[calibration_sample],p[calibration_sample],s=4,alpha=.2);ax.plot([calibration_y.min(),calibration_y.max()],[calibration_y.min(),calibration_y.max()],'k--',lw=1)
        ax.set(title=arm+' validation',xlabel='MC return',ylabel='Predicted return')
    save(fig,'value_calibration.png')
    curve=pd.read_csv(OUT/'value_fit_training_curves.csv');fig,ax=plt.subplots(figsize=(8,4))
    for (arm,split),g in curve[curve.split.isin(['train','validation'])].groupby(['arm','split']):ax.plot(g.gradient_updates,g.rmse,label=f'{arm} {split}',ls='-' if split=='train' else '--')
    ax.set(xlabel='Equal gradient-update budget',ylabel='Raw-unit MC RMSE');ax.legend(ncol=2,fontsize=8);save(fig,'value_loss_curves.png')
    sat=pd.read_csv(OUT/'huber_saturation_diagnostics.csv');fig,ax=plt.subplots(figsize=(7,4))
    for arm,g in sat.groupby('arm'):ax.plot(g.unit,g.hypothetical_huber_saturation_fraction,label=arm)
    ax.set(xlabel='Episode update',ylabel='P(|residual| > 1)',title='Hypothetical Huber saturation (actual loss: MSE)');ax.legend();save(fig,'huber_saturation.png')
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(y,bins=50,alpha=.35,density=True,label='MC return')
    for arm in ['V0','V1','V2','V3']:ax.hist(preds[arm][test],bins=50,histtype='step',density=True,label=arm)
    ax.set(xlabel='Raw return units',ylabel='Density');ax.legend();save(fig,'prediction_return_distributions.png')
    table=pd.read_csv(OUT/'value_fit_main_results.csv');t=table[table.split=='test'];fig,ax=plt.subplots(figsize=(8,4))
    pos=np.arange(len(t));ax.bar(pos-.18,t.r2,.36,label='R²');ax.bar(pos+.18,t.explained_variance,.36,label='EV');ax.set_xticks(pos,t.arm);ax.legend();save(fig,'test_r2_ev.png')
    primary=selected[(selected.lens=='production_smdp')&(selected.horizon==20)]
    for arm,name in [('V0','original_gae_vs_centered_qh.png'),(best,'best_value_gae_vs_centered_qh.png')]:
        fig,ax=plt.subplots(figsize=(6,4));ax.scatter(primary.reference_advantage,primary[arm+'_normalized'],s=9,alpha=.4)
        ax.set(xlabel='Matched centered Q_H20 (production SMDP)',ylabel=f'{arm} normalized GAE');save(fig,name)
    by=pd.read_csv(OUT/'gae_vs_counterfactual_by_coupling.csv');by=by[(by.lens=='production_smdp')&(by.horizon==20)&(by.advantage_type=='normalized')]
    fig,ax=plt.subplots(figsize=(7,4))
    for offset,label in [(-.18,'coupled'),(.18,'easy')]:
        g=by[by.coupling==label].set_index('value_source').reindex(['V0','V1','V2','V3']);ax.bar(np.arange(4)+offset,g.spearman,.36,label=label)
    ax.set_xticks(np.arange(4),['V0','V1','V2','V3']);ax.set_ylabel('GAE/reference Spearman');ax.legend();save(fig,'coupled_easy_alignment.png')
    overall=pd.read_csv(OUT/'gae_vs_counterfactual_advantage.csv');overall=overall[(overall.lens=='production_smdp')&(overall.advantage_type=='normalized')]
    fig,ax=plt.subplots(figsize=(7,4))
    for arm,g in overall.groupby('value_source'):ax.plot(g.horizon,g.spearman,'o-',label=arm)
    ax.set(xlabel='Evaluation-only H',ylabel='GAE/reference Spearman');ax.legend();save(fig,'horizon_alignment.png')


def decision_and_report(data,selected):
    meta=json.loads((OUT/'frozen_policy_dataset_manifest.json').read_text());fitmeta=json.loads((OUT/'value_fit_metadata.json').read_text());best=fitmeta['best_validation_arm']
    fits=pd.read_csv(OUT/'value_fit_main_results.csv');test=fits[fits.split=='test'].set_index('arm');val=fits[fits.split=='validation'].set_index('arm')
    a=pd.read_csv(OUT/'gae_vs_counterfactual_advantage.csv');c=pd.read_csv(OUT/'gae_vs_counterfactual_by_coupling.csv')
    a=a[(a.lens=='production_smdp')&(a.advantage_type=='normalized')&(a.horizon==20)].set_index('value_source')
    c=c[(c.lens=='production_smdp')&(c.advantage_type=='normalized')&(c.horizon==20)&(c.coupling=='coupled')].set_index('value_source')
    def improved(t,arm,base='V0'):
        return bool(t.loc[arm,'mae']<=.9*t.loc[base,'mae'] and t.loc[arm,'r2']>=t.loc[base,'r2']+.05
                    and t.loc[arm,'explained_variance']>=t.loc[base,'explained_variance']+.05
                    and t.loc[arm,'pearson']>=t.loc[base,'pearson']+.10)
    good_value=improved(test,best) and improved(val,best)
    align=bool(a.loc[best,'spearman']-a.loc['V0','spearman']>=.05 or a.loc[best,'sign_agreement']-a.loc['V0','sign_agreement']>=.05)
    coupled=bool(c.loc[best,'spearman']-c.loc['V0','spearman']>=.02 or c.loc[best,'sign_agreement']-c.loc['V0','sign_agreement']>=.02)
    go=bool(good_value and align and coupled)
    scale=bool(test.loc['V3','mae']<.9*test.loc['V2','mae'] and test.loc['V3','r2']>test.loc['V2','r2']+.05)
    limited=bool(test.loc[['V3','Ridge','SmallMLP'],'r2'].max()<.10)
    if go:primary='D';secondary='B' if scale else None
    elif limited:primary='C';secondary='B' if scale else None
    elif good_value:primary='E';secondary='B' if scale else None
    elif scale:primary='B';secondary=None
    elif improved(test,'V1') and improved(val,'V1'):primary='A';secondary=None
    else:primary='C';secondary=None
    result=dict(primary=primary,secondary=secondary,best_validation_arm=best,value_gate=good_value,
                overall_gae_gate=align,coupled_gae_gate=coupled,modify_formal_v=go,
                integrate_centered_advantage=False,standardization_scale_effect=scale,
                current_state_low_predictability=limited,one_seed_only=True,
                primary_reference='production task-credit reward + gamma**delta_t',
                historical_reference='interval completion reward + gamma**step, sensitivity only',
                original_huber_hypothesis_rejected='Formal value loss is MSE; delta1 saturation is hypothetical',
                v1_inference_limit='Fresh offline replay and critic-only gradient clipping also differ; improvement alone does not isolate policy nonstationarity causally')
    (OUT/'root_cause_decision.json').write_text(json.dumps(result,indent=2))
    integrity=json.loads((OUT/'frozen_policy_integrity.json').read_text());mc=pd.read_csv(OUT/'mc_return_integrity_checks.csv')
    checks=pd.read_csv(OUT/'matched_state_replay_checks.csv');mismatch=int(checks.filter(like='mismatch').to_numpy().sum())
    noise=pd.read_csv(OUT/'return_noise_diagnostics.csv');sat=pd.read_csv(OUT/'huber_saturation_diagnostics.csv')
    grads=pd.read_csv(OUT/'gradient_update_diagnostics.csv')
    grad_summary=grads.assign(clipped=grads.gradient_norm_before_clip>params.max_grad_norm_ppo).groupby('arm',as_index=False)[['gradient_norm_before_clip','parameter_update_norm','prediction_shift','clipped']].mean()
    grad_summary.to_csv(OUT/'gradient_update_summary.csv',index=False)
    overall=pd.read_csv(OUT/'gae_vs_counterfactual_advantage.csv')
    def table(frame,columns):
        lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
        for _,r in frame.iterrows():lines.append('| '+' | '.join(f'{r[k]:.5g}' if isinstance(r[k],(float,np.floating)) else str(r[k]) for k in columns)+' |')
        return '\n'.join(lines)
    report=[ '# Frozen-Policy Value Critic / GAE Target Identifiability Audit','',
        f"Decision: Primary {primary}, Secondary {secondary}; formal-V GO={go}; Centered Advantage integration=NO. Validation-selected {best} reduces test MAE from {test.loc['V0','mae']:.3f} to {test.loc[best,'mae']:.3f}, while production-consistent normalized H20 GAE Spearman changes {a.loc['V0','spearman']:.4f} → {a.loc[best,'spearman']:.4f}, and sign agreement {a.loc['V0','sign_agreement']:.2%} → {a.loc[best,'sign_agreement']:.2%}.",'',
        f"Source main: {meta['source_commit']}. Trial-0 frozen stochastic Actor, 300 natural episodes / 60,000 decisions. No PPO retraining. One-seed descriptive evidence, not a multi-seed causal conclusion.",'',
        '## Two implementation facts established before fitting','',
        '1. Production masked PPO uses **0.5 × MSE**, with a joint Actor/V gradient-norm cap of 0.5. It does not use Huber. All reported delta=1 Huber fractions are hypothetical; they cannot explain production gradients as Huber saturation.',
        'Source anchors: agents/masked_pair_ppo_agent.py::train_step; core/main_loop.py::_get_ppo_terminal_time; diagnostics/run_long_horizon_coupling.py::_interval_returns; core/env_state.py::get_state.',
        '2. Historical long-horizon Q_H accumulates completion-event interval rewards with gamma**step. Production PPO assigns final task reward back to its originating task transition and discounts by gamma**delta_t. Primary matched comparisons here use the production reward/discount. Legacy interval/step Q_H is retained separately. Historical 21.6% coupled rate and prior ranking numbers therefore do not describe exactly the production objective. No historical artifact was changed.','',
        '## Frozen data and terminal integrity','',
        f"Actor checkpoint SHA256: {integrity['actor_file_hash_before']}. Before/after collection and after all side work match. Fixed-state action-distribution hashes match; protected file mismatches: {integrity.get('protected_file_mismatch_count',0)}.",
        f"Split: 180 train / 60 validation / 60 test episodes. Matched indices were frozen before collection and Q_H: 1,000 decisions across the test episodes. Seeds, split IDs, input hashes, index list and optimization gates are persisted. MC maximum recursion residual: {mc.max_recursion_residual.max():.3g}.",
        'After the last arrival, the environment waits for each pending task resolution (first replica result), backfills its own final reward, then stores exactly one terminal transition. Its delta_t ends at the latest task outcome. No terminal bootstrap is used. SimPy subsequently drains remaining non-cancelled replicas. Natural MC uses only that episode’s recorded task-origin rewards; there is no rollout truncation or cross-episode target. The diagnostic original normalized GAE is the value the frozen rollout **would** feed the production actor loss; no Actor update actually occurs.',
        'Observation normalization is the fixed production get_state mapping. Raw state fields and policy-normalized observations, masks, probabilities/logits, physical diagnostics, V/TD/GAE, rewards/delta_t/done and terminal type are saved in per-episode NPZ files. Episode seeds are independent deterministic substreams of one formal seed.','',
        '## Horizon-free return scale',
        table(pd.read_csv(OUT/'value_target_statistics.csv').query("quantity == 'MC_return'"),['group','mean','std','p05','p50','p95']),
        'All reported value errors/calibration are in original return units after inverse transformation; target normalization is TRAIN-only.',
        '## Equal-budget offline value fitting','',
        'V1 uses the existing bootstrapped GAE-return target, refreshed once per episode update, not plain TD(0). V2 fits raw horizon-free MC return; V3 fits TRAIN-only standardized MC return and inverse-transforms predictions. Each uses fresh identical initialization, [64,32] tanh, Adam 5e-4, batch64, 2 epochs ×300 train-episode updates =2,400 gradients. The train episodes cycle deterministically. Critic-only clipping uses the same 0.5 limit; this does not reproduce the original combined Actor/V gradient norm. No tuning or test-set checkpoint selection. V4 would be identical to V2 because the production loss is already MSE and is omitted.',
        'Small MLP is [128,64] ReLU with the same budget and standardized target. Ridge alpha=1 uses train-only standardized state columns. They are identifiability diagnostics, not production changes.',
        'MC return removes bootstrap-target dependence and provides a horizon-free **realized** return-to-go fitting target under the frozen policy. It is a stochastic sample, not the true policy value.','',
        '### Validation',table(fits[fits.split=='validation'],['arm','mae','rmse','pearson','spearman','r2','explained_variance']),
        '### Test',table(fits[fits.split=='test'],['arm','mae','rmse','pearson','spearman','r2','explained_variance']),'',
        '## Hypothetical Huber thresholds / actual optimization',
        'V1/V2 residual thresholds are in raw return units; V3/SmallMLP are in standardized training units. The CSV also includes raw-unit fractions. Prediction-shift magnitudes follow the same native-unit distinction; clipped is the fraction of gradient updates exceeding the shared norm cap.',
        table(sat[sat.unit.isin([0,30,150,300])],['arm','unit','hypothetical_huber_saturation_fraction','residual_mean','residual_std']),
        table(grad_summary,['arm','gradient_norm_before_clip','parameter_update_norm','prediction_shift','clipped']),'',
        '## State identifiability and realized-return noise',
        table(noise,['method','n','conditional_variance_ratio','mean_neighbor_distance']),
        'Nearest neighbors and coarse bins approximate states; their residual variance is not an irreducible-noise bound. MC return also depends on remaining natural-episode length; the current observation has no explicit decision index or remaining-episode length. It also excludes the episode-specific spatial hazard realization and effective mask, although those affect the deployed masked distribution. These are possible information limitations, not demonstrated causal explanations; no additional state features are supplied to these fitting arms. Position-stratified return statistics are supplied without adding position to any model.','',
        '## Matched-state replay and common randomness',
        f"Reconstructed {checks.state_id.nunique()} states, {len(checks)} candidate branches. Aggregate state/mask/probability/backlog/task/snapshot/prefix/arrival/Torch-stream mismatch count: {mismatch}. Branches simulate all 200 tasks through natural drain; only the target action is overridden, after consuming its normal categorical draw. Later arrivals, spatial realization and task-index action RNG streams match. Production reliability vectors from the baseline are reused only after hazard/task identity checks; the unforced replay re-evaluates production feasibility.",
        'Each action has one common-random-number continuation, not a many-replicate expectation estimate. H=5/10/20/50 is evaluation only; neither MC/GAE fitting nor arm selection reads Q_H. Centering uses saved masked probabilities. Sign agreement excludes near-zero reference advantages (|A_ref|≤1e-8); denominator and zero-inclusive sensitivity are saved.','',
        '### Primary production-consistent normalized GAE alignment',
        table(overall[(overall.lens=='production_smdp')&(overall.advantage_type=='normalized')],['value_source','horizon','n','nonzero_reference_count','pearson','spearman','sign_agreement','normalized_mae','quartile_agreement']),
        '### Historical interval/step normalized GAE sensitivity',
        table(overall[(overall.lens=='legacy_interval_step')&(overall.advantage_type=='normalized')],['value_source','horizon','spearman','sign_agreement']),
        '### Previously trained centered-advantage learner (frozen)',
        table(pd.read_csv(OUT/'frozen_centered_advantage_selected_alignment.csv').query("lens == 'production_smdp'"),['horizon','pearson','spearman','sign_agreement']),
        'This table compares the learner output for the originally selected action to exactly the same reference used for GAE. Per-state within-action ranking and selected greedy regret are separate metrics in frozen_centered_advantage_alignment.csv; they are not directly interchangeable with cross-state GAE correlation.',
        'Raw GAE and all coupling/load/requirement/safe-set strata are in the companion CSVs. Coupled retains myopic H20 regret >0.01 separately for each reward lens. The frozen previous Advantage network is evaluated without parameter updates.','',
        '## Required answers',
        'Q1. Actor and original V checkpoints remain frozen; parameter and distribution probes match.',
        f'Q2. MC uses gamma**delta_t and a terminal reset; max residual {mc.max_recursion_residual.max():.3g}.',
        f"Q3. Original V test MAE={test.loc['V0','mae']:.5g}, R²={test.loc['V0','r2']:.5g}, EV={test.loc['V0','explained_variance']:.5g}.",
        f"Q4. Fresh frozen V1 test MAE={test.loc['V1','mae']:.5g}, R²={test.loc['V1','r2']:.5g}. A change cannot isolate policy tracking from fresh initialization/offline updates/critic-only clipping.",
        f"Q5. Raw MC V2 test MAE={test.loc['V2','mae']:.5g}, R²={test.loc['V2','r2']:.5g}.",
        f"Q6. Standardized V3 test MAE={test.loc['V3','mae']:.5g}, R²={test.loc['V3','r2']:.5g}; raw-versus-standardized scale gate={scale}.",
        'Q7. Most raw residuals may exceed 1, but active MSE does not saturate like Huber. Gradient clipping/scale is the applicable mechanism to consider.',
        'Q8. MSE-only V4 is redundant with V2, so it is not a distinct experiment.',
        f"Q9. Best test R² among V3/Ridge/SmallMLP={test.loc[['V3','Ridge','SmallMLP'],'r2'].max():.5g}; low-predictability (<0.10) flag={limited}.",
        f"Q10. k20 conditional/global variance ratio={noise.iloc[0].conditional_variance_ratio:.5g}; coarse-bin ratio={noise.iloc[1].conditional_variance_ratio:.5g} (descriptive).",
        f'Q11. Exactly 1,000 test decisions matched to their original rollout GAE; mismatch count={mismatch}.',
        f"Q12. Original normalized GAE H20 Spearman={a.loc['V0','spearman']:.5g}, sign={a.loc['V0','sign_agreement']:.5g}.",
        f"Q13. Validation-selected {best}: H20 Spearman={a.loc[best,'spearman']:.5g}, sign={a.loc[best,'sign_agreement']:.5g}; value gate={good_value}, alignment gate={align}.",
        f"Q14. Coupled H20 original/best Spearman={c.loc['V0','spearman']:.5g}/{c.loc[best,'spearman']:.5g}; sign={c.loc['V0','sign_agreement']:.5g}/{c.loc[best,'sign_agreement']:.5g}; coupled gate={coupled}.",
        'Q15. The following H20 tables report high-load, all requirements and safe-set strata explicitly.',
        f'Q16. Primary={primary}; Secondary={secondary}. Formal-V modification GO={go}; Centered Advantage/Actor integration remains NO.','',
        '## H20 key strata']
    for file,extra in [('gae_vs_counterfactual_by_coupling.csv','coupling'),('gae_vs_counterfactual_by_load.csv','load'),('gae_vs_counterfactual_by_requirement.csv','requirement'),('gae_vs_counterfactual_by_safe_set.csv','safe_set_bin')]:
        frame=pd.read_csv(OUT/file);frame=frame[(frame.lens=='production_smdp')&(frame.horizon==20)&(frame.advantage_type=='normalized')&(frame.value_source.isin(['V0',best]))]
        report.extend(['',table(frame,[extra,'value_source','n','nonzero_reference_count','spearman','sign_agreement'])])
    report.extend(['','## Decision gates and limitation',
                   'The fitted models explain less than 10% of held-out realized-return variance at the fixed budget, supporting limited observed-state predictability (C), not impossibility. Standardization clearly improves optimization (B). However, better absolute-value fit did not improve overall or coupled GAE alignment: the pattern relevant to E is present, and there is no evidence here for D or for immediately changing production PPO. This does not establish the cause of the deployed performance gap to Safe Min-Latency.',
                   '```json\n'+json.dumps(result,indent=2)+'\n```',
                   'Classification labels: A = moving-policy value tracking; B = value scale/loss optimization; C = limited state-return identifiability; D = better value fitting improves GAE credit; E = better value fitting does not fix GAE action credit. The C flag means limited prediction in these fixed-budget models, not zero information or proof of irreducible noise.',
                   'Multiple diagnostic arms share one formal seed. Neither 60,000 transitions nor 1,000 matched states are independent training seeds. Improvements are descriptive, not statistical proof of a causal mechanism or proof of better deployed PPO.'])
    report.extend(['','## Reproduction and regression',
        'Run with the project Conda interpreter from the repository root. The stages are: prepare, collect, fit, branch --workers 4, verify via diagnostics/run_value_gae_identifiability.py; then diagnostics/analyze_value_gae_identifiability.py. Existing collection/branch files are resumed, not silently replaced. Fitting is deterministic for the persisted seeds and software environment.',
        'Only new diagnostic files and this result directory belong to this change. Production sources, formal checkpoints, data spreadsheets and protected historical artifacts are hash-checked. Test results and software versions are recorded in validation_summary.json.'])
    (OUT/'VALUE_GAE_IDENTIFIABILITY_REPORT.md').write_text('\n\n'.join(report)+'\n')
    return best


def main():
    torch.set_num_threads(1);meta=json.loads((OUT/'frozen_policy_dataset_manifest.json').read_text());data=load_dataset()
    scale_and_noise(data,meta);selected=matched_analysis(data);verify_protected()
    best=decision_and_report(data,selected);make_plots(data,selected,best)
    print((OUT/'root_cause_decision.json').read_text(),flush=True)


if __name__=='__main__':main()
