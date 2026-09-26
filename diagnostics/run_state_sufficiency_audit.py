"""Frozen-rollout collection, augmented-state fit and audit report pipeline."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from agents.ppo_agent import PPOValueNetwork
from diagnostics.state_sufficiency_audit import *


def old_episode(ep):
    with np.load(VALUE_OUT/'dataset'/f'episode_{int(ep):03d}.npz') as z:
        return {k:z[k] for k in z.files}


def seed_frame():
    return pd.read_csv(VALUE_OUT/'episode_seed_plan.csv')


def protected_hashes():
    manifest=json.loads((VALUE_OUT/'protected_hashes.json').read_text())
    return manifest


def collection_manifest():
    old=json.loads(VALUE_MANIFEST.read_text())
    return {
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'parent_value_audit_commit':old['source_commit'],
        'formal_checkpoint_sha256':sha256_file(CHECKPOINT/'actor.pt'),
        'previous_critic_sha256':sha256_file(CHECKPOINT/'critic.pt'),
        'episodes':300,'tasks_per_episode':200,'decisions':60000,
        'episode_split':old['split'],'matched_state_ids':pd.read_csv(VALUE_OUT/'matched_state_manifest.csv').state_id.tolist(),
        'matched_state_selection':'Reused unchanged from frozen-policy value audit; 1,000 test states; no reselection',
        'new_upload_instrumentation':'Diagnostic-only wrapper around Task.calc_input_output_delay and EnvironmentState.register_waiting_replica; passes original durations/events through unchanged.',
        'trajectory_regression_fields':['action','observation','probability','effective_mask','safe_mask','reliability','reward','delta_t','task_delay','arrival_trace','Torch RNG trace'],
        'input_dimension':35,'server_count':8,
        'state_dimensions':{'S0':35,'S1':126,'S2':75,'S3':166,'S4':223},
        'variant_definitions':{
            'S0':'Production 35-D observation.',
            'S1':'S0 + pair reliability[28] + effective mask[28] + safe mask[28] + summary[7].',
            'S2':'S0 + per-server in-flight count[8], future CPU service workload[8], and remaining-upload mean/min/max[24].',
            'S3':'S0 + S1 reliability/mask context + S2 in-flight context.',
            'S4':'S3 + per-server running remaining service, waiting service/count, running indicator, pending replica count, episode-effective hazards and current spatial field (diagnostic upper-bound arm).',
        },
        'no_future_information':'Each feature is captured at the current decision time. Future event traces/outcomes are used only as targets or diagnostics, never as model inputs.',
        'feature_normalization':'S0 retains production [0,1] observation. Appended augmented features use TRAIN-only column mean/std; zero-variance columns scale by 1. MC target uses prior audit train-only mean/std.',
        'prediction_models':'Same [64,32] tanh PPOValueNetwork for all MLP variants; all Ridge arms use train-only standardized input, alpha=1.',
        'optimization_budget':{'gradient_updates_per_MLP':2400,'batch_size':64,'epochs_per_episode_update':2,'train_episode_updates':300,'optimizer':'Adam','lr':params.critic_lr_ppo,'target':'train-only standardized SMDP MC return','init_seed':VARIANT_SEED},
        'nearest_neighbor':'Same k=20 train-neighbor variance estimator and train-only standardization for every variant.',
        'alias_thresholds':{'close_S0_neighbors':'test nearest-neighbor distance at or below the test-set 10th percentile; include all exact ties, so selected share may exceed 10%; descriptive only',
            'flight_distinct':'Euclidean distance > 1 in train-standardized in-flight vectors',
            'backlog_similar':'Euclidean distance <= 0.10 over the eight S0 normalized backlog coordinates'},
        'preregistered_decision_rules':{
            'A':'S1/S2/S3 have negligible test R2/EV gains, no material conditional-variance decrease, and no coupled GAE alignment gain.',
            'B':'S1 reliability/mask context gives the clearest predictive gains, especially high requirement, small safe set, and mask aliases.',
            'C':'S2 in-flight features give the clearest predictive gains, especially high load and queue-alias states.',
            'D':'S3 test R2 gain >= 0.05, positive R2 interaction, >= 0.10 relative conditional-variance reduction, and coupled H20 Spearman or sign improvement >= 0.02 over S0.',
            'E':'An augmented variant improves held-out prediction by test R2 >= 0.02 (or MAE >= 10%) while no variant improves coupled H20 Spearman or sign by >= 0.02.'},
        'classification_effect_size_thresholds':{'material_test_r2_gain':0.02,'strong_both_state_r2_gain':0.05,'conditional_variance_relative_reduction':0.10,'coupled_h20_alignment_gain':0.02,'backlog_similarity_l2':0.10,'flight_standardized_l2':1.0},
        'protected_input_hash_manifest':str((VALUE_OUT/'protected_hashes.json').relative_to(ROOT)),
        'all_splits_episodes_disjoint':True,
    }


def set_up_agent():
    agent,rho=make_agent()
    agent.__class__=StateSufficiencyAgent
    return agent,rho


def flatten_row(ep_data,episode,i):
    row={'episode':int(episode),'decision_index':int(i+1),'dataset_index':int((episode-1)*200+i),
         'task_id':int(ep_data['task_ids'][i]),'simulation_time':float(ep_data['time'][i]),
         'pending_task_count':int(ep_data['pending_task_count'][i]),'active_upload_count':int(ep_data['active_uploads'][i]),
         'action_index':int(ep_data['actions'][i]),'reward':float(ep_data['rewards'][i]),
         'delta_t':float(ep_data['delta_t'][i]),'done':bool(ep_data['done'][i]),
         'mc_return':float(ep_data['mc_return'][i]),'task_delay':float(ep_data['task_delay'][i]),
         'safe_set_size':int(ep_data['safe_mask'][i].sum()),
         'safe_set_empty':bool(ep_data.get('safe_set_empty', np.sum(ep_data['safe_mask'],axis=1)==0)[i]),
         'reliability_requirement':float(ep_data['requirement'][i])}
    groups={
        's0':ep_data['states'][i], 'pair_reliability':ep_data['pair_reliability'][i],
        'safe_mask':ep_data['safe_mask'][i].astype(float), 'effective_mask':ep_data['effective_mask'][i].astype(float),
        'reliability_summary':ep_data['reliability_summary'][i], 'inflight_count':ep_data['inflight_count'][i],
        'inflight_workload':ep_data['inflight_workload'][i], 'upload_remaining_mean':ep_data['upload_remaining_mean'][i],
        'upload_remaining_min':ep_data['upload_remaining_min'][i], 'upload_remaining_max':ep_data['upload_remaining_max'][i],
        'running_remaining':ep_data['running_remaining'][i], 'waiting_service':ep_data['waiting_service'][i],
        'waiting_count':ep_data['waiting_count'][i], 'running_count':ep_data['running_count'][i],
        'pending_replica_count':ep_data['pending_replica_count'][i], 'cpu_backlog_seconds':ep_data['cpu_backlog_seconds'][i],
        'effective_failure_rates':ep_data['effective_failure_rates'][i], 'spatial_risk_field':ep_data['spatial_risk_field'][i],
        'probabilities':ep_data['probabilities'][i], 'pair_correlation':ep_data['pair_correlation'],
    }
    for name,vec in groups.items():
        for j,value in enumerate(np.asarray(vec).ravel()):row[f'{name}_{j:02d}']=float(value)
    row['inflight_replica_records_json']=str(ep_data['flight_records_json'][i])
    return row


def trajectory_regression(old,new,episode):
    check={}
    for name in ['states','actions','probabilities','effective_mask','safe_mask','reliabilities','rewards','delta_t','arrival_trace','rng_trace']:
        a,b=np.asarray(old[name]),np.asarray(new[name])
        check[name+'_mismatch']=int(not np.array_equal(a,b))
    delay=np.asarray(old['task_delay'],float)-np.asarray(new['task_delay'],float)
    check.update(episode=int(episode),action_mismatch=check['actions_mismatch'],
        state_mismatch=check['states_mismatch'],mask_mismatch=int(check['effective_mask_mismatch'] or check['safe_mask_mismatch']),
        probability_mismatch=check['probabilities_mismatch'],reliability_mismatch=check['reliabilities_mismatch'],
        reward_max_abs_diff=float(np.max(np.abs(np.asarray(old['rewards'])-np.asarray(new['rewards'])))),
        delay_max_abs_diff=float(np.max(np.abs(delay))),
        upload_events=int(2*len(old['actions'])))
    mismatches=sum(check[k] for k in check if k.endswith('_mismatch'))
    if mismatches or check['reward_max_abs_diff']!=0.0 or check['delay_max_abs_diff']!=0.0:
        raise RuntimeError(f'Frozen rollout changed while collecting telemetry in episode {episode}: {check}')
    return check


def collect():
    OUT.mkdir(parents=True,exist_ok=True)
    inv=OUT/'STATE_INFORMATION_INVENTORY.md'
    if not inv.exists():raise RuntimeError('Write the state inventory before collecting/fitting')
    meta_path=OUT/'state_variant_manifest.json'
    if not meta_path.exists():meta_path.write_text(json.dumps(collection_manifest(),indent=2))
    manifest=json.loads(meta_path.read_text())
    if manifest['parent_value_audit_commit']!='5fc305361404c7fe0f1a8b5f6efb9105007e753c':
        raise RuntimeError('Unexpected frozen-policy data source')
    scratch=OUT/'scratch';scratch.mkdir(exist_ok=True)
    seedrows=seed_frame().to_dict('records')
    actor_file_before=sha256_file(CHECKPOINT/'actor.pt')
    critic_file_before=sha256_file(CHECKPOINT/'critic.pt')
    agent,rho=set_up_agent();actor_param_before=model_hash(agent.policy_old)
    critic_param_before=model_hash(agent.value_net)
    regression=[];history_counts=[];telemetry_path=OUT/'latent_state_telemetry.csv.gz'
    # Per-episode scratch chunks make long data capture restartable.
    for seedrow in seedrows:
        episode=int(seedrow['episode']);epfile=scratch/f'episode_{episode:03d}.npz'
        old=old_episode(episode)
        if not epfile.exists():
            data,loop=collect_replay_episode(agent,seedrow)
            check=trajectory_regression(old,data,episode);regression.append(check)
            data['episode']=np.full(200,episode,dtype=np.int16)
            data['decision_index']=np.arange(1,201,dtype=np.int16)
            data['pair_correlation']=np.broadcast_to(rho,(200,len(rho))).copy()
            data['flight_features']=np.asarray([rec['flight_features'] for rec in agent.records])
            data['reliability_features']=np.asarray([rec['reliability_features'] for rec in agent.records])
            # Capture fixed current traces from diagnostic state, never any future event.
            for key in ['inflight_count','inflight_workload','upload_remaining_mean','upload_remaining_min','upload_remaining_max',
                        'running_remaining','waiting_service','waiting_count','running_count','pending_replica_count','cpu_backlog_seconds',
                        'effective_failure_rates','spatial_risk_field','pair_reliability','reliability_summary','state1','state2','state3','state4',
                        'pending_task_count','active_uploads','flight_records_json']:
                data[key]=np.asarray([rec[key] for rec in agent.records])
            if len(loop.env_state.audit_upload_history)!=400 or loop.env_state.audit_inflight:
                raise RuntimeError(f'Upload lifecycle incomplete in episode {episode}')
            history_counts.append(dict(episode=episode,upload_start_and_queue_entry_count=len(loop.env_state.audit_upload_history),
                                       active_upload_at_terminal=len(loop.env_state.audit_inflight)))
            np.savez_compressed(epfile,**data)
        else:
            with np.load(epfile) as z:data={k:z[k] for k in z.files}
            # A resumptive chunk must still reproduce its source episode.
            check=trajectory_regression(old,data,episode);regression.append(check)
        if episode%20==0:print(f'collected and regression-checked {episode}/300 episodes',flush=True)

    chunks=[np.load(scratch/f'episode_{ep:03d}.npz',allow_pickle=False) for ep in range(1,301)]
    keys=[k for k in chunks[0].files if chunks[0][k].ndim>0 and chunks[0][k].shape[0]==200]
    all_data={k:np.concatenate([z[k] for z in chunks],axis=0) for k in keys}
    for z in chunks:z.close()
    if len(all_data['states'])!=60000:raise RuntimeError('Expected 60,000 frozen decisions')
    # Stream the wide table once; a complete gzip CSV is reused after a
    # later-stage integrity check fails so restart does not rewrite 60k rows.
    telemetry_valid=False
    if telemetry_path.exists():
        with gzip.open(telemetry_path,'rt',encoding='utf-8',newline='') as stream:
            telemetry_valid=sum(1 for _ in stream)==60001
    if not telemetry_valid:
        with gzip.open(telemetry_path,'wt',encoding='utf-8',newline='') as stream:
            for episode in range(1,301):
                with np.load(scratch/f'episode_{episode:03d}.npz',allow_pickle=False) as z:
                    epdata={k:z[k] for k in z.files}
                if 'safe_set_empty' not in epdata:
                    epdata['safe_set_empty']=(np.sum(epdata['safe_mask'],axis=1)==0)
                rows=[flatten_row(epdata,episode,i) for i in range(200)]
                pd.DataFrame(rows).to_csv(stream,index=False,header=episode==1)
    else:
        print('reusing complete 60,000-row telemetry table',flush=True)
    pd.DataFrame(regression).to_csv(OUT/'frozen_trajectory_regression.csv',index=False)
    if len(regression)!=300 or sum(row['action_mismatch'] for row in regression):raise RuntimeError('Frozen action trace mismatch')
    actor_file_after=sha256_file(CHECKPOINT/'actor.pt');critic_file_after=sha256_file(CHECKPOINT/'critic.pt')
    actor_param_after=model_hash(agent.policy_old);critic_param_after=model_hash(agent.value_net)
    if (actor_file_before!=actor_file_after or actor_param_before!=actor_param_after or
        critic_file_before!=critic_file_after or critic_param_before!=critic_param_after or
        actor_file_before!=manifest['formal_checkpoint_sha256'] or
        critic_file_before!=manifest['previous_critic_sha256']):
        raise RuntimeError('Formal frozen policy/critic checkpoint changed')
    protected=protected_hashes();bad=[f for f,h in protected.items() if sha256_file(ROOT/f)!=h]
    if bad:raise RuntimeError(f'Protected production/historical files changed: {bad}')
    regression_summary={k: (int(sum(row[k] for row in regression)) if k.endswith('_mismatch') else max(float(row[k]) for row in regression))
                        for k in ['action_mismatch','state_mismatch','mask_mismatch','probability_mismatch','reliability_mismatch','reward_max_abs_diff','delay_max_abs_diff']}
    integrity={'actor_file_sha256_before':actor_file_before,'actor_file_sha256_after':actor_file_after,
               'actor_parameter_sha256_before':actor_param_before,'actor_parameter_sha256_after':actor_param_after,
               'critic_file_sha256_before':critic_file_before,'critic_file_sha256_after':critic_file_after,
               'formal_policy_parameter_mismatch_count':int(actor_param_before!=actor_param_after),
               'formal_critic_parameter_mismatch_count':int(critic_param_before!=critic_param_after),
               'protected_source_hash_mismatch_count':len(bad),**regression_summary,
               'episodes':300,'decision_count':60000,'upload_lifecycle_count':int(2*60000)}
    (OUT/'trajectory_regression_summary.json').write_text(json.dumps(integrity,indent=2))
    # Fixed row IDs are imported unchanged; every row stays in held-out test.
    oldmatched=pd.read_csv(VALUE_OUT/'matched_state_manifest.csv')
    if oldmatched.state_id.nunique()!=1000 or set(oldmatched.state_id)!=set(manifest['matched_state_ids']):raise RuntimeError('Matched-state IDs changed')
    oldmatched['dataset_index']=(oldmatched.episode.to_numpy()-1)*200+oldmatched.task_id.to_numpy()-1
    idx=oldmatched.dataset_index.to_numpy(dtype=int)
    matched={'state_id':oldmatched.state_id,'episode':oldmatched.episode,'task_id':oldmatched.task_id,
             'dataset_index':idx,'load':oldmatched.load,'requirement':oldmatched.requirement,
             'safe_set_bin':oldmatched.safe_set_bin,'safe_set_size':oldmatched.safe_set_size,
             'effective_set_size':oldmatched.effective_set_size,'mc_return':all_data['mc_return'][idx],
             'selected_action':all_data['actions'][idx], 'task_delay':all_data['task_delay'][idx]}
    feat_cols={}
    # Explicit variant features are regenerated from the captured semantics.
    variants=make_variants(all_data)
    for name,x in variants.items():
        for j in range(x.shape[1]):feat_cols[f'{name}_{j:03d}']=x[idx,j]
    pd.DataFrame({**matched,**feat_cols}).to_csv(OUT/'matched_state_augmented_features.csv',index=False)
    manifest['telemetry_decisions']=60000;manifest['upload_lifecycle_records']=120000
    manifest['input_trajectory_regression']=integrity
    manifest['formal_policy_changed']=False;manifest['formal_simulator_changed']=False
    manifest['feature_dimensions']={k:int(v.shape[1]) for k,v in variants.items()}
    (OUT/'state_variant_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Feature and target integrity digest for all fitted rows.
    with open(OUT/'latent_state_telemetry.csv.gz','rb') as f:manifest['latent_telemetry_sha256']=hashlib.sha256(f.read()).hexdigest()
    (OUT/'state_variant_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Scratch contains only reproducible per-episode copies; final aggregate files are committed.
    shutil.rmtree(scratch)
    np.savez_compressed(OUT/'variant_features_for_fit.npz',**{**variants,**{
        'episode':all_data['episode'],'task_id':all_data['task_ids'],'mc_return':all_data['mc_return'],
        'rewards':all_data['rewards'],'delta_t':all_data['delta_t'],'done':all_data['done'],
        'original_gae':all_data['original_gae'],'original_normalized_gae':all_data['original_normalized_gae'],
        'task_delay':all_data['task_delay'],'s0_backlog':all_data['backlog'],
        'selected_action':all_data['actions'],'probabilities':all_data['probabilities'],
        'effective_mask':all_data['effective_mask'],'safe_mask':all_data['safe_mask'],
        'pair_reliability':all_data['pair_reliability'],'inflight_count':all_data['inflight_count'],
        'inflight_workload':all_data['inflight_workload'],'upload_remaining_mean':all_data['upload_remaining_mean'],
    }})
    print(json.dumps(integrity,indent=2),flush=True)


def _predict(model,x,batch_size=2048):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(x[i:i+batch_size],dtype=torch.float32)).cpu().numpy()
                               for i in range(0,len(x),batch_size)])


def train_mlp(name,x,episode,y,split,target_mean,target_std):
    x=np.asarray(x,dtype=np.float32).copy();trainmask=np.isin(episode,split['train'])
    # Keep production S0 coordinates as-is; scale only newly appended telemetry/context.
    feature_mean=np.zeros(x.shape[1],float);feature_scale=np.ones(x.shape[1],float)
    if name!='S0':
        feature_mean[35:]=x[trainmask,35:].mean(0)
        feature_scale[35:]=x[trainmask,35:].std(0)
        feature_scale[feature_scale<1e-8]=1.0
        x[:,35:]=(x[:,35:]-feature_mean[35:])/feature_scale[35:]
    yt=normalize_target(y,target_mean,target_std).astype(np.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(VARIANT_SEED)
        model=PPOValueNetwork(x.shape[1],params.hidden_layers_ppo,activation=params.af_ppo)
    optimizer=torch.optim.Adam(model.parameters(),lr=params.critic_lr_ppo)
    rng=np.random.default_rng(2026092703)
    train_eps=np.random.default_rng(2026092702).permutation(split['train'])
    units=np.resize(train_eps,300)
    records=[];step=0
    for unit,ep in enumerate(units,1):
        ix=np.flatnonzero(episode==ep);rngids=rng.permutation(len(ix))
        states=torch.as_tensor(x[ix],dtype=torch.float32);targets=torch.as_tensor(yt[ix],dtype=torch.float32)
        for _ in range(params.k_epochs_ppo):
            order=rngids if _==0 else rng.permutation(len(ix))
            for start in range(0,len(ix),params.batch_size_ppo):
                batch=order[start:start+params.batch_size_ppo]
                pred=model(states[batch]);loss=params.value_loss_coef_ppo*nn.functional.mse_loss(pred,targets[batch])
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=nn.utils.clip_grad_norm_(model.parameters(),params.max_grad_norm_ppo)
                optimizer.step();step+=1
                if step%400==0:records.append({'state_variant':name,'gradient_updates':step,'unit':unit,'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm)})
    if step!=2400:raise RuntimeError(f'{name} received {step} updates, expected 2400')
    (OUT/'side_models').mkdir(exist_ok=True)
    torch.save(model.state_dict(),OUT/'side_models'/f'{name}_MLP.pt')
    pred=_predict(model,x)
    pred_raw=inverse_target(pred,target_mean,target_std)
    return model,pred_raw,feature_mean,feature_scale,records


def train_ridge(name,x,episode,y,split,target_mean,target_std):
    x=np.asarray(x,dtype=np.float64);train=np.isin(episode,split['train'])
    mu=x[train].mean(0);sd=x[train].std(0);sd[sd<1e-8]=1.0
    z=(x-mu)/sd;zt=normalize_target(y[train],target_mean,target_std)
    coef=np.linalg.solve(z[train].T@z[train]+np.eye(z.shape[1]),z[train].T@zt)
    pred=inverse_target(z@coef,target_mean,target_std)
    (OUT/'side_models').mkdir(exist_ok=True)
    np.savez_compressed(OUT/'side_models'/f'{name}_Ridge.npz',feature_mean=mu,feature_std=sd,coefficients=coef,target_mean=target_mean,target_std=target_std,alpha=1.)
    return pred


def fit():
    meta=json.loads((OUT/'state_variant_manifest.json').read_text())
    z=np.load(OUT/'variant_features_for_fit.npz',allow_pickle=False)
    data={k:z[k] for k in z.files};z.close()
    feature_names=['S0','S1','S2','S3']
    split=meta['episode_split'];episode=data['episode'];y=data['mc_return']
    oldmeta=json.loads((VALUE_OUT/'value_fit_metadata.json').read_text());mu=float(oldmeta['train_mean']);sd=float(oldmeta['train_std'])
    if abs(mu-float(y[np.isin(episode,split['train'])].mean()))>1e-8 or abs(sd-float(y[np.isin(episode,split['train'])].std()))>1e-8:
        raise RuntimeError('Reused training target statistics do not match the frozen return dataset')
    predictions={};rows=[];curves=[];models={}
    for name in feature_names:
        model,pred,feature_mean,feature_scale,curve=train_mlp(name,data[name],episode,y,split,mu,sd)
        predictions[name+'_MLP']=pred;models[name]=model;curves.extend(curve)
        if name=='S0':
            with np.load(VALUE_OUT/'value_predictions.npz') as ref:previous=ref['V3']
            s0max=float(np.max(np.abs(pred-previous)))
            if s0max>2e-5:raise RuntimeError(f'S0 MLP failed to reproduce previous V3 baseline: {s0max}')
            meta['S0_MLP_previous_V3_max_prediction_difference']=s0max
        ridge=train_ridge(name,data[name],episode,y,split,mu,sd);predictions[name+'_Ridge']=ridge
        for arm,p in [(name+'_MLP',pred),(name+'_Ridge',ridge)]:
            for part,epids in split.items():
                mask=np.isin(episode,epids)
                rows.append({'state_variant':name,'model':arm.split('_')[-1],'split':part,'n':int(mask.sum()),**value_metrics(p[mask],y[mask])})
        print(f'fit {name} MLP + Ridge done',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'state_value_prediction_results.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'state_value_training_diagnostics.csv',index=False)
    np.savez_compressed(OUT/'state_value_predictions.npz',**predictions)
    meta['target_train_mean']=mu;meta['target_train_std']=sd;meta['models_trained']=True
    (OUT/'state_variant_manifest.json').write_text(json.dumps(meta,indent=2))
    add_subgroup_predictions(data,predictions,split,meta)
    compute_gae_alignment(data,predictions,meta)
    aliasing_analysis(data,meta)
    make_plots_and_decision(data,predictions,meta)
    z.close()


def add_subgroup_predictions(data,predictions,split,meta):
    ep=data['episode'];test=np.isin(ep,split['test']);m=pd.read_csv(VALUE_OUT/'matched_state_manifest.csv')
    rows=[]
    common={'high_load':test & (np.max(data['S0'][:,16:24],axis=1)>0.0)}
    # Reuse exact historical matched-state load/Rreq/safe strata for action-credit groups later.
    for group,condition in [('load_low',m.load.eq('low')),('load_medium',m.load.eq('medium')),('load_high',m.load.eq('high')),
                            *[(f'req_{r}',np.isclose(m.requirement,r)) for r in (.9,.99,.999,.9999)],
                            ('safe_1_5',m.safe_set_bin.eq('1-5')),('safe_6_15',m.safe_set_bin.eq('6-15')),
                            ('safe_16_28',m.safe_set_bin.eq('16-28')),('safe_empty',m.safe_set_bin.eq('empty'))]:
        ix=m.loc[condition,'dataset_index'].to_numpy(dtype=int)
        for name in ['S0','S1','S2','S3']:
            for model in ['MLP','Ridge']:
                p=predictions[name+'_'+model]
                rows.append({'group':group,'state_variant':name,'model':model,'n':len(ix),**value_metrics(p[ix],data['mc_return'][ix])})
    pd.DataFrame([r for r in rows if r['group'].startswith('load_')]).to_csv(OUT/'state_value_prediction_by_load.csv',index=False)
    pd.DataFrame([r for r in rows if r['group'].startswith('req_')]).to_csv(OUT/'state_value_prediction_by_requirement.csv',index=False)
    pd.DataFrame([r for r in rows if r['group'].startswith('safe_')]).to_csv(OUT/'state_value_prediction_by_safe_set.csv',index=False)
    # k=20 train-neighbor conditional variance uses the same estimator for S0..S3.
    train=np.isin(ep,meta['episode_split']['train']);te=np.isin(ep,meta['episode_split']['test']);variance=np.var(data['mc_return'][te]);out=[]
    for name in ['S0','S1','S2','S3']:
        distance,indices=nearest_indices(data[name],train,te,k=20)
        local=data['mc_return'][train][indices]
        ratio=float(np.var(local,axis=1).mean()/variance)
        out.append({'state_variant':name,'k':20,'n_test':int(te.sum()),'conditional_global_variance_ratio':ratio,
                    'mean_neighbor_distance':float(distance.mean()),'estimator':'Mean within 20 TRAIN-neighbor MC-return variance / global TEST variance.'})
    pd.DataFrame(out).to_csv(OUT/'conditional_return_variance.csv',index=False)


def compute_gae_alignment(data,predictions,meta):
    episode=data['episode'];gaes={};
    for name in ['S0','S1','S2','S3']:
        value=predictions[name+'_MLP'];normalized=np.empty(len(episode),dtype=float)
        raw=np.empty(len(episode),dtype=float)
        for ep in range(1,301):
            ix=np.flatnonzero(episode==ep);nv=np.zeros(len(ix),dtype=float);nv[:-1]=value[ix[1:]]
            a=production_gae(data['rewards'][ix]*params.reward_scale_ppo,data['delta_t'][ix],data['done'][ix],
                             value[ix],nv,params.gamma_ppo,params.gae_lambda_ppo)
            raw[ix]=a[1];normalized[ix]=a[2]
        gaes[name]=raw;gaes[name+'_normalized']=normalized
    np.savez_compressed(OUT/'augmented_offline_gae.npz',**gaes)
    prev=pd.read_csv(VALUE_OUT/'matched_selected_action_signals.csv')
    prev=prev[(prev.lens=='production_smdp')&(prev.horizon.isin(HORIZONS))]
    manifest=pd.read_csv(VALUE_OUT/'matched_state_manifest.csv');index=(manifest.episode.to_numpy()-1)*200+manifest.task_id.to_numpy()-1
    if prev.state_id.nunique()!=1000:raise RuntimeError('Prior matched action-credit references missing')
    split_rows=[]
    q=prev[['state_id','episode','task_id','lens','horizon','coupling','reference_advantage']].drop_duplicates()
    for r in q.itertuples(index=False):
        ix=(int(r.episode)-1)*200+int(r.task_id)-1
        row={'state_id':r.state_id,'episode':r.episode,'task_id':r.task_id,'horizon':r.horizon,'coupling':r.coupling,
             'reference_advantage':r.reference_advantage,'load':manifest.loc[manifest.state_id==r.state_id,'load'].iloc[0],
             'requirement':float(manifest.loc[manifest.state_id==r.state_id,'requirement'].iloc[0]),
             'safe_set_bin':manifest.loc[manifest.state_id==r.state_id,'safe_set_bin'].iloc[0]}
        for name in ['S0','S1','S2','S3']:
            row[name+'_raw']=gaes[name][ix];row[name+'_normalized']=gaes[name+'_normalized'][ix]
        split_rows.append(row)
    selected=pd.DataFrame(split_rows);selected.to_csv(OUT/'augmented_state_gae_alignment.csv',index=False)
    for fname,groups in [('augmented_state_gae_by_coupling.csv',['coupling']),('augmented_state_gae_by_load.csv',['load']),
                         ('augmented_state_gae_by_requirement.csv',['requirement']),('augmented_state_gae_by_safe_set.csv',['safe_set_bin'])]:
        out=[]
        for key,frame in selected.groupby(['horizon']+groups,dropna=False):
            context=dict(zip(['horizon']+groups,key if isinstance(key,tuple) else (key,)))
            for name in ['S0','S1','S2','S3']:
                for norm in ['normalized','raw']:
                    p=frame[name+'_'+norm].to_numpy();y=frame.reference_advantage.to_numpy();nz=np.abs(y)>1e-8
                    corr=float(pd.Series(p).corr(pd.Series(y),method='spearman')) if len(p)>2 else float('nan')
                    pearson=float(np.corrcoef(p,y)[0,1]) if np.std(p)>1e-12 and np.std(y)>1e-12 else float('nan')
                    out.append({**context,'state_variant':name,'advantage_type':norm,'n':len(frame),
                                'nonzero_reference_count':int(nz.sum()),'pearson':pearson,'spearman':corr,
                                'sign_agreement':float(np.mean(np.sign(p[nz])==np.sign(y[nz]))) if nz.any() else np.nan})
        pd.DataFrame(out).to_csv(OUT/fname,index=False)


def aliasing_analysis(data,meta):
    ep=data['episode'];test=np.isin(ep,meta['episode_split']['test']);train=np.isin(ep,meta['episode_split']['train'])
    idx=np.flatnonzero(test);s0=np.asarray(data['S0'][test],dtype=np.float64)
    tree=cKDTree(s0);dist,near=tree.query(s0,k=2,workers=1)
    local=np.where(near[:,0]==np.arange(len(near)),1,0);partner=np.where(local==0,near[:,0],near[:,1]);d0=np.where(local==0,dist[:,0],dist[:,1])
    global_i=idx;global_j=idx[partner];effective=data['effective_mask'];safe=data['safe_mask'];reliability=data['pair_reliability']
    flight=np.asarray(data.get('flight_features',data['S2'][:,35:]),dtype=float)
    trainflight=flight[train].astype(float);fscale=trainflight.std(0);fscale[fscale<1e-8]=1.
    fdelta=(flight[global_i].astype(float)-flight[global_j].astype(float))/fscale
    fd=np.linalg.norm(fdelta,axis=1)
    workload=np.asarray(data['inflight_workload'],dtype=float)
    wscale=workload[train].std(0);wscale[wscale<1e-8]=1.
    workload_delta=(workload[global_i]-workload[global_j])/wscale
    workload_distance=np.linalg.norm(workload_delta,axis=1)
    mham=np.abs(effective[global_i].astype(int)-effective[global_j].astype(int)).sum(1)
    sham=np.abs(safe[global_i].astype(int)-safe[global_j].astype(int)).sum(1)
    tv=.5*np.abs(data['probabilities'][global_i]-data['probabilities'][global_j]).sum(1)
    gap=np.abs(data['mc_return'][global_i]-data['mc_return'][global_j])
    backlog_obs=data['S0'][:,16:24];bd=np.linalg.norm(backlog_obs[global_i]-backlog_obs[global_j],axis=1)
    close_cut=float(np.quantile(d0,.10));close=d0<=close_cut;flightdiff=fd>1.;maskdiff=mham>0
    pairs=pd.DataFrame({'dataset_index':global_i,'neighbor_index':global_j,'episode':ep[global_i],'neighbor_episode':ep[global_j],
        'task_id':data['task_id'][global_i],'neighbor_task_id':data['task_id'][global_j],
        's0_distance':d0,'pair_reliability_distance':np.linalg.norm(reliability[global_i]-reliability[global_j],axis=1),
        'effective_mask_hamming':mham,'safe_mask_hamming':sham,'policy_probability_tv':tv,
        'flight_standardized_distance':fd,'inflight_workload_standardized_distance':workload_distance,
        'backlog_observation_distance':bd,'return_gap':gap,
        'mask_alias':maskdiff,'flight_alias':flightdiff,'close_s0_neighbor':close})
    # Future queue/delay are labels for mechanism analysis only; never used by a model.
    queue_change=np.full(len(ep),np.nan);nextq=np.full(len(ep),np.nan)
    for i in range(len(ep)):
        t=int(data['task_id'][i]);e=int(ep[i])
        if t<200:
            nxt=i+1
            queue_change[i]=float(data['s0_backlog'][nxt].sum()-data['s0_backlog'][i].sum())
            nextq[i]=float(data['s0_backlog'][nxt].sum())
        else:
            queue_change[i]=float(-data['s0_backlog'][i].sum())
    pairs['flight_queue_change_gap']=np.abs(queue_change[global_i]-queue_change[global_j])
    pairs['next_task_backlog_gap']=np.abs(nextq[global_i]-nextq[global_j])
    pairs['realized_latency_gap']=np.abs(data['task_delay'][global_i]-data['task_delay'][global_j])
    pairs.to_csv(OUT/'nearest_neighbor_aliasing.csv',index=False)
    nearclose=pairs[close].copy()
    def alias_row(label,frame):
        return {'group':label,'n_pairs':len(frame),'mean_s0_distance':frame.s0_distance.mean(),
            'mean_effective_mask_hamming':frame.effective_mask_hamming.mean(),'effective_mask_alias_rate':frame.mask_alias.mean(),
            'mean_safe_mask_hamming':frame.safe_mask_hamming.mean(),
            'safe_mask_alias_rate':float((frame.safe_mask_hamming>0).mean()),
            'mean_flight_standardized_distance':frame.flight_standardized_distance.mean(),
            'flight_alias_rate':frame.flight_alias.mean(),'mean_return_gap':frame.return_gap.mean(),
            'median_return_gap':frame.return_gap.median(),'p90_return_gap':frame.return_gap.quantile(.9),
            'mean_policy_probability_tv':frame.policy_probability_tv.mean()}
    mask_summary=alias_row('S0 nearest-neighbor cohort; percentile ties included',nearclose)
    mask_summary.update(
        nearest_neighbor_return_gap=float(nearclose.return_gap.mean()),
        effective_mask_alias_return_gap=float(nearclose.loc[nearclose.mask_alias,'return_gap'].mean()),
        safe_mask_alias_return_gap=float(nearclose.loc[nearclose.safe_mask_hamming>0,'return_gap'].mean()),
        in_flight_alias_return_gap=float(nearclose.loc[nearclose.flight_alias,'return_gap'].mean()),
        neither_alias_return_gap=float(nearclose.loc[(~nearclose.mask_alias)&(~nearclose.flight_alias),'return_gap'].mean()))
    pd.DataFrame([mask_summary]).to_csv(OUT/'mask_aliasing_analysis.csv',index=False)
    group_summary=[
        {'group':'all S0-near pairs',**alias_row('all S0-near pairs',nearclose)},
        {'group':'effective-mask alias',**alias_row('effective-mask alias',nearclose[nearclose.mask_alias])},
        {'group':'in-flight alias',**alias_row('in-flight alias',nearclose[nearclose.flight_alias])},
        {'group':'neither alias',**alias_row('neither alias',nearclose[(~nearclose.mask_alias)&(~nearclose.flight_alias)])},
        {'group':'both aliases',**alias_row('both aliases',nearclose[nearclose.mask_alias&nearclose.flight_alias])},
    ]
    pd.DataFrame(group_summary).to_csv(OUT/'aliasing_mechanism_summary.csv',index=False)
    # Overall and per-server hidden upload workload, measured at each of the
    # 60,000 decision instants. Future queue/delay fields remain labels only.
    counts=np.asarray(data['inflight_count'],dtype=float)
    workloads=np.asarray(data['inflight_workload'],dtype=float)
    summary=[{'scope':'aggregate_per_decision','server_id':'all',
        'decisions':int(len(counts)),'decisions_with_inflight':int((counts.sum(axis=1)>0).sum()),
        'inflight_decision_rate':float(np.mean(counts.sum(axis=1)>0)),
        'mean_inflight_replicas':float(counts.sum(axis=1).mean()),
        'max_inflight_replicas':int(counts.sum(axis=1).max()),
        'mean_inflight_future_service_workload':float(workloads.sum(axis=1).mean()),
        'max_inflight_future_service_workload':float(workloads.sum(axis=1).max())}]
    for j in range(counts.shape[1]):
        summary.append({'scope':'per_server_per_decision','server_id':j+1,
            'decisions':int(len(counts)),'decisions_with_inflight':int((counts[:,j]>0).sum()),
            'inflight_decision_rate':float(np.mean(counts[:,j]>0)),
            'mean_inflight_replicas':float(counts[:,j].mean()),
            'max_inflight_replicas':int(counts[:,j].max()),
            'mean_inflight_future_service_workload':float(workloads[:,j].mean()),
            'max_inflight_future_service_workload':float(workloads[:,j].max())})
    pd.DataFrame(summary).to_csv(OUT/'hidden_workload_summary.csv',index=False)
    # Four-way attribution among close S0 neighbors. Mask D is the neither-different reference group.
    groups=[]
    for label,md,fdiff in [('A_mask_only',True,False),('B_flight_only',False,True),('C_both',True,True),('D_neither',False,False)]:
        frame=nearclose[(nearclose.mask_alias==md)&(nearclose.flight_alias==fdiff)]
        groups.append(alias_row(label,frame))
    pd.DataFrame(groups).to_csv(OUT/'return_gap_attribution.csv',index=False)
    backlog_similar=nearclose[nearclose.backlog_observation_distance<=.10]
    inflight_groups=[]
    for label,flag in [('distinct_inflight',True),('similar_inflight',False)]:
        frame=backlog_similar[backlog_similar.flight_alias==flag]
        inflight_groups.append({'group':label,'n_pairs':len(frame),'mean_flight_distance':frame.flight_standardized_distance.mean(),
            'mean_return_gap':frame.return_gap.mean(),'median_return_gap':frame.return_gap.median(),
            'mean_future_queue_change_gap':frame.flight_queue_change_gap.mean(),
            'mean_next_task_backlog_gap':frame.next_task_backlog_gap.mean(),
            'mean_realized_latency_gap':frame.realized_latency_gap.mean()})
    pd.DataFrame(inflight_groups).to_csv(OUT/'inflight_aliasing_analysis.csv',index=False)
    # Choose one representative per unordered pair, all from the pre-defined closest neighborhood.
    cases=nearclose.sort_values('return_gap',ascending=False).copy()
    cases['pair_key']=cases.apply(lambda r:tuple(sorted((int(r.dataset_index),int(r.neighbor_index)))),axis=1)
    cases=cases.drop_duplicates('pair_key').head(10)
    case_rows=[]
    for r in cases.itertuples(index=False):
        i=int(r.dataset_index);j=int(r.neighbor_index)
        def js(v):return json.dumps(np.asarray(v).tolist(),separators=(',',':'))
        a={'case_id':len(case_rows)+1,'dataset_index_i':i,'dataset_index_j':j,'episode_i':int(ep[i]),'episode_j':int(ep[j]),
           'task_id_i':int(data['task_id'][i]),'task_id_j':int(data['task_id'][j]),
           's0_distance':float(r.s0_distance),'return_i':float(data['mc_return'][i]),'return_j':float(data['mc_return'][j]),
           'return_gap':float(r.return_gap),'effective_mask_hamming':int(r.effective_mask_hamming),
           'policy_tv':float(r.policy_probability_tv),'safe_mask_i':js(safe[i]),'safe_mask_j':js(safe[j]),
           'effective_mask_i':js(effective[i]),'effective_mask_j':js(effective[j]),
           'pair_reliability_i':js(reliability[i]),'pair_reliability_j':js(reliability[j]),
           'backlog_i':js(data['s0_backlog'][i]),'backlog_j':js(data['s0_backlog'][j]),
           'inflight_count_i':js(data['inflight_count'][i]),'inflight_count_j':js(data['inflight_count'][j]),
           'inflight_workload_i':js(data['inflight_workload'][i]),'inflight_workload_j':js(data['inflight_workload'][j]),
           'upload_remaining_min_i':js(data['upload_remaining_min'][i]),'upload_remaining_mean_i':js(data['upload_remaining_mean'][i]),
           'upload_remaining_max_i':js(data['upload_remaining_max'][i]),'upload_remaining_min_j':js(data['upload_remaining_min'][j]),
           'upload_remaining_mean_j':js(data['upload_remaining_mean'][j]),'upload_remaining_max_j':js(data['upload_remaining_max'][j]),
           'task_delay_i':float(data['task_delay'][i]),'task_delay_j':float(data['task_delay'][j])}
        case_rows.append(a)
    pd.DataFrame(case_rows).to_csv(OUT/'nominal_state_aliasing_case_studies.csv',index=False)
    return pairs,groups,inflight_groups


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['collect','fit','analyze'])
    args=parser.parse_args();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='collect':collect()
    elif args.stage=='fit':fit()
    else:
        z=np.load(OUT/'variant_features_for_fit.npz',allow_pickle=False);data={k:z[k] for k in z.files};z.close()
        predfile=np.load(OUT/'state_value_predictions.npz',allow_pickle=False);pred={k:predfile[k] for k in predfile.files};predfile.close()
        meta=json.loads((OUT/'state_variant_manifest.json').read_text())
        aliasing_analysis(data,meta);make_plots_and_decision(data,pred,meta)


def _alignment_tables():
    overall=pd.read_csv(OUT/'augmented_state_gae_alignment.csv')
    def summarize(frame,group_name,group_value):
        rows=[]
        for name in ['S0','S1','S2','S3']:
            for horizon,part in frame.groupby('horizon'):
                x=part[name+'_normalized'].to_numpy();y=part.reference_advantage.to_numpy();nz=np.abs(y)>1e-8
                rows.append({'group':group_value,'state_variant':name,'horizon':int(horizon),'n':len(part),
                    'nonzero_reference_count':int(nz.sum()),
                    'pearson':float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-12 and np.std(y)>1e-12 else np.nan,
                    'spearman':float(pd.Series(x).corr(pd.Series(y),method='spearman')) if len(x)>2 else np.nan,
                    'sign_agreement':float(np.mean(np.sign(x[nz])==np.sign(y[nz]))) if nz.any() else np.nan})
        return rows
    rows=summarize(overall,'all','all')
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'gae_alignment_summary.csv',index=False)
    return result


def _markdown_table(frame):
    def cell(value):
        if pd.isna(value):
            return ''
        if isinstance(value,(float,np.floating)):
            return f'{float(value):.4g}'
        return str(value).replace('|','\\|')
    columns=[str(c) for c in frame.columns]
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    rows.extend('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)


def make_plots_and_decision(data,predictions,meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=pd.read_csv(OUT/'state_value_prediction_results.csv')
    test=results[results.split=='test']
    var=test[test.model=='MLP'].set_index('state_variant')
    ridge=test[test.model=='Ridge'].set_index('state_variant')
    contrib=[]
    for model,table in [('MLP',var),('Ridge',ridge)]:
        s0,s1,s2,s3=[float(table.loc[x,'r2']) for x in ['S0','S1','S2','S3']]
        contrib.append({'model':model,'r2_s0':s0,'r2_s1':s1,'r2_s2':s2,'r2_s3':s3,
            'delta_r2_reliability':s1-s0,'delta_r2_inflight':s2-s0,'delta_r2_both':s3-s0,
            'interaction':s3-s1-s2+s0})
    pd.DataFrame(contrib).to_csv(OUT/'incremental_state_information.csv',index=False)
    cond=pd.read_csv(OUT/'conditional_return_variance.csv').set_index('state_variant')
    align=_alignment_tables()
    h20=align[align.horizon==20].set_index('state_variant')
    coupled=pd.read_csv(OUT/'augmented_state_gae_by_coupling.csv')
    c20=coupled[(coupled.horizon==20)&(coupled.advantage_type=='normalized')&(coupled.coupling=='coupled')].set_index('state_variant')
    easy20=coupled[(coupled.horizon==20)&(coupled.advantage_type=='normalized')&(coupled.coupling=='easy')].set_index('state_variant')
    strong=float(var.loc['S3','r2']-var.loc['S0','r2'])>=.05
    interaction=float(var.loc['S3','r2']-var.loc['S1','r2']-var.loc['S2','r2']+var.loc['S0','r2'])>0
    variance_drop=1-float(cond.loc['S3','conditional_global_variance_ratio'])/float(cond.loc['S0','conditional_global_variance_ratio'])
    c_gain=max(float(c20.loc['S3','spearman']-c20.loc['S0','spearman']),
               float(c20.loc['S3','sign_agreement']-c20.loc['S0','sign_agreement']))
    strong_both=strong and interaction and variance_drop>=.10 and c_gain>=.02
    material_any=any(float(var.loc[n,'r2']-var.loc['S0','r2'])>=.02 or float(var.loc[n,'mae'])<=.9*float(var.loc['S0','mae']) for n in ['S1','S2','S3'])
    gae_any=max(float(c20.loc[n,'spearman']-c20.loc['S0','spearman']) for n in ['S1','S2','S3'])
    gae_any=max(gae_any,max(float(c20.loc[n,'sign_agreement']-c20.loc['S0','sign_agreement']) for n in ['S1','S2','S3']))
    e_condition=material_any and gae_any<.02
    del_r=float(var.loc['S1','r2']-var.loc['S0','r2']);del_f=float(var.loc['S2','r2']-var.loc['S0','r2'])
    high=pd.read_csv(OUT/'state_value_prediction_by_load.csv').query("group == 'load_high' and model == 'MLP'").set_index('state_variant')
    highest=pd.read_csv(OUT/'state_value_prediction_by_requirement.csv').query("group == 'req_0.9999' and model == 'MLP'").set_index('state_variant')
    smallsafe=pd.read_csv(OUT/'state_value_prediction_by_safe_set.csv').query("group == 'safe_6_15' and model == 'MLP'").set_index('state_variant')
    mask_alias=pd.read_csv(OUT/'mask_aliasing_analysis.csv').iloc[0]
    inflight_alias=pd.read_csv(OUT/'inflight_aliasing_analysis.csv')
    strongest_reliability=del_r>del_f and (highest.loc['S1','r2']>highest.loc['S0','r2'] or mask_alias.effective_mask_alias_rate>.10)
    strongest_flight=del_f>del_r and high.loc['S2','r2']>high.loc['S0','r2']
    if strong_both: root='D'
    elif strongest_reliability: root='B'
    elif strongest_flight: root='C'
    elif e_condition: root='E'
    elif all(float(var.loc[n,'r2']-var.loc['S0','r2'])<.02 for n in ['S1','S2','S3']) and variance_drop<.05 and max(gae_any,0)<.02:root='A'
    else:root='E' if material_any else 'A'
    min_significant_predictive=(strongest_reliability or strongest_flight or strong_both)
    coupled_improved=c_gain>=.02
    recommend_state=bool(strong_both and min_significant_predictive and coupled_improved)
    result={'primary_classification':root,'classification_name':{'A':'CURRENT STATE IS SUFFICIENT ENOUGH','B':'RELIABILITY CONTEXT IS MISSING STATE',
        'C':'IN-FLIGHT WORKLOAD IS MISSING STATE','D':'BOTH ARE NEEDED','E':'VALUE PREDICTION IMPROVES BUT ACTION CREDIT STILL DOES NOT'}[root],
        'secondary_classification':None,'test_r2_delta':{n:float(var.loc[n,'r2']-var.loc['S0','r2']) for n in ['S1','S2','S3']},
        'conditional_variance_relative_reduction_s3':variance_drop,'coupled_h20_alignment_gain_s3':c_gain,
        'mask_alias_rate_top_nearest_close_s0':float(mask_alias.effective_mask_alias_rate),
        'safe_mask_alias_rate_top_nearest_close_s0':float(mask_alias.safe_mask_alias_rate),
        'mean_mask_alias_return_gap':float(mask_alias.effective_mask_alias_return_gap),
        'mean_inflight_alias_return_gap':float(mask_alias.in_flight_alias_return_gap),
        'mean_neither_alias_return_gap':float(mask_alias.neither_alias_return_gap),
        's0_neighbor_pairs':int(mask_alias.n_pairs),
        's0_neighbor_pair_share_of_test':float(mask_alias.n_pairs/12000),
        'formal_state_expansion_recommended':recommend_state,'formal_ppo_retraining_recommended':False,
        'formal_actor_or_simulator_modified':False,'single_formal_seed_descriptive_only':True,
        'decision_thresholds':meta['classification_effect_size_thresholds'],
        'strong_both_state_gate':bool(strong_both),'predictive_gain_without_coupled_gae_gate':bool(e_condition),
        'conclusion_limit':'A single frozen evaluation seed and supervised diagnostics do not establish that state aliasing causes the Safe Min-Latency performance gap.'}
    (OUT/'state_sufficiency_decision.json').write_text(json.dumps(result,indent=2))
    # Compact main diagnostic plots, all held-out/test results.
    def save(fig,name):fig.tight_layout();fig.savefig(OUT/name,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));names=['S0','S1','S2','S3'];ax.bar(names,[var.loc[n,'r2'] for n in names]);ax.set(ylabel='Test R²',title='State variant return prediction (MLP)');save(fig,'test_r2_by_state.png')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(names,[var.loc[n,'explained_variance'] for n in names]);ax.set(ylabel='Test explained variance',title='State variant explained variance (MLP)');save(fig,'test_ev_by_state.png')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(names,[cond.loc[n,'conditional_global_variance_ratio'] for n in names]);ax.axhline(1.,color='k',ls='--',lw=1);ax.set(ylabel='Conditional / global variance',title='k=20 train-neighbor return variance');save(fig,'conditional_return_variance.png')
    pairs=pd.read_csv(OUT/'nearest_neighbor_aliasing.csv');close=pairs[pairs.close_s0_neighbor]
    fig,ax=plt.subplots(figsize=(7,4));sc=ax.scatter(close.s0_distance,close.return_gap,c=close.effective_mask_hamming,s=10,cmap='viridis');fig.colorbar(sc,ax=ax,label='Effective-mask Hamming');ax.set(xlabel='S0 nearest-neighbor distance',ylabel='MC return gap',title='Nominal-state nearest-neighbor aliasing');save(fig,'s0_distance_vs_return_gap.png')
    fig,ax=plt.subplots(figsize=(7,4));sc=ax.scatter(close.effective_mask_hamming,close.return_gap,c=close.policy_probability_tv,s=10,cmap='plasma');fig.colorbar(sc,ax=ax,label='Policy distribution TV');ax.set(xlabel='Effective-mask Hamming distance',ylabel='MC return gap');save(fig,'mask_distance_vs_return_gap.png')
    fig,ax=plt.subplots(figsize=(7,4));sc=ax.scatter(close.inflight_workload_standardized_distance,close.return_gap,c=close.backlog_observation_distance,s=10,cmap='viridis');fig.colorbar(sc,ax=ax,label='S0 backlog distance');ax.set(xlabel='Standardized in-flight CPU workload distance',ylabel='MC return gap');save(fig,'flight_distance_vs_return_gap.png')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(names,[high.loc[n,'r2'] for n in names]);ax.set(ylabel='High-load test R²',title='High-load value prediction (MLP)');save(fig,'high_load_prediction.png')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(names,[c20.loc[n,'spearman'] for n in names]);ax.set(ylabel='Spearman',title='H20 coupled-state GAE alignment');save(fig,'coupled_h20_gae_alignment.png')
    cases=pd.read_csv(OUT/'nominal_state_aliasing_case_studies.csv')
    if len(cases):
        case=cases.iloc[0];i=int(case.dataset_index_i);j=int(case.dataset_index_j)
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True);sid=np.arange(1,9)
        for ax,back,flight,label in [(axes[0],data['s0_backlog'][i],data['inflight_workload'][i],f"State i, G={case.return_i:.1f}"),
                                     (axes[1],data['s0_backlog'][j],data['inflight_workload'][j],f"Neighbor j, G={case.return_j:.1f}")]:
            ax.bar(sid-.18,back,.36,label='CPU backlog');ax.bar(sid+.18,flight,.36,label='In-flight CPU work');ax.set_xticks(sid);ax.set_xlabel('Server');ax.set_title(label);ax.legend(fontsize=8)
        fig.suptitle(f"S0 distance={case.s0_distance:.4f}, return gap={case.return_gap:.2f}, mask Hamming={case.effective_mask_hamming}")
        save(fig,'representative_state_aliasing_case.png')
    # Add a concise, reproducible report. The full tables remain machine-readable CSVs.
    pred=pd.read_csv(OUT/'state_value_prediction_results.csv').query("split == 'test'")
    cond_table=pd.read_csv(OUT/'conditional_return_variance.csv')
    reg=json.loads((OUT/'trajectory_regression_summary.json').read_text())
    alias=pd.read_csv(OUT/'return_gap_attribution.csv');rtable=pd.DataFrame(contrib)
    alignsummary=align.query('horizon == 20').merge(pd.read_csv(OUT/'augmented_state_gae_by_coupling.csv').query("horizon == 20 and coupling == 'coupled' and advantage_type == 'normalized'"),on='state_variant',suffixes=('_overall','_coupled'))
    report=['# State Sufficiency Audit','',f"Parent frozen-policy value audit commit: `{meta['parent_value_audit_commit']}`. This diagnostic uses one frozen formal evaluation seed and 60,000 task decisions; all fit and state-aliasing comparisons are descriptive.",'',
        '## Decision','',f"Primary classification: **{root} — {result['classification_name']}**. Formal observation expansion: **{'YES' if recommend_state else 'NO'}**. PPO retraining: **NO**.",
        'An augmented predictor alone does not show that the state representation causes the performance gap to Safe Min-Latency. The coupled-state GAE H20 alignment gate is required for a state expansion recommendation.',
        '## Frozen trajectory regression','',f"Actor file and parameter hashes match before/after; mismatches: action {reg['action_mismatch']}, state {reg['state_mismatch']}, mask {reg['mask_mismatch']}, probability {reg['probability_mismatch']}, reliability {reg['reliability_mismatch']}. Maximum reward and task-delay differences are {reg['reward_max_abs_diff']:.3g} and {reg['delay_max_abs_diff']:.3g}. All production/historical protected hashes match.",
        'The upload observer wrapped only the existing input-delay calculation and queue-registration callback. It forwarded the same return values and events. Every episode still resolves 200 tasks and records 400 upload-to-CPU-queue transitions.',
        '## Value prediction on held-out test episodes','',_markdown_table(pred[['state_variant','model','mae','rmse','pearson','spearman','r2','explained_variance']]),
        '',_markdown_table(rtable),'',
        'S0 exactly reproduces the previous V3 prediction baseline within the tolerance stored in the manifest. MLPs share a [64,32] tanh architecture, target scale, initialization seed and 2,400-update budget; only input dimensions and TRAIN-only scaling of appended features differ. Ridge uses alpha=1 and train-only feature scaling.',
        '## Conditional return variance (same k=20 estimator)','',_markdown_table(cond_table),
        '## State aliasing and hidden workload','',
        '### In-flight workload at decision time','',_markdown_table(pd.read_csv(OUT/'hidden_workload_summary.csv')),'',
        f"The S0 nearest-neighbor 10th-percentile distance is {mask_alias.mean_s0_distance:.4g}; ties at distance zero are all retained, yielding {int(mask_alias.n_pairs)} pairs ({mask_alias.n_pairs/12000:.1%} of 12,000 test states). Safe/effective masks differ in {mask_alias.safe_mask_alias_rate:.1%}/{mask_alias.effective_mask_alias_rate:.1%}; mean Hamming distances are {mask_alias.mean_safe_mask_hamming:.3f}/{mask_alias.mean_effective_mask_hamming:.3f}. Effective-mask aliases have mean return gap {mask_alias.effective_mask_alias_return_gap:.3f}; all near-neighbor pairs have mean return gap {mask_alias.nearest_neighbor_return_gap:.3f}, and mean policy-distribution TV {mask_alias.mean_policy_probability_tv:.3f}.",
        _markdown_table(alias),'',_markdown_table(pd.read_csv(OUT/'inflight_aliasing_analysis.csv')),
        f"High-load MLP S0/S2 R²: {high.loc['S0','r2']:.4f}/{high.loc['S2','r2']:.4f}. Highest-requirement S0/S1 R²: {highest.loc['S0','r2']:.4f}/{highest.loc['S1','r2']:.4f}. Safe-set 6–15 S0/S3 R²: {smallsafe.loc['S0','r2']:.4f}/{smallsafe.loc['S3','r2']:.4f}.",
        'Ten nominal-state aliasing examples are saved with both states’ mask, reliability vector, backlog, in-flight count/workload, upload-time summary and realized delay. Pairwise follow-on queue and latency differences are diagnostic outcomes only and never training features.',
        '## H20 action-credit alignment','',_markdown_table(alignsummary[['state_variant','spearman_overall','sign_agreement_overall','spearman_coupled','sign_agreement_coupled']]),
        '',_markdown_table(pd.read_csv(OUT/'augmented_state_gae_by_load.csv').query("horizon == 20 and load == 'high' and advantage_type == 'normalized'")[['state_variant','spearman','sign_agreement']]),
        '',_markdown_table(pd.read_csv(OUT/'augmented_state_gae_by_requirement.csv').query("horizon == 20 and requirement == 0.9999 and advantage_type == 'normalized'")[['state_variant','spearman','sign_agreement']]),
        '',_markdown_table(pd.read_csv(OUT/'augmented_state_gae_by_safe_set.csv').query("horizon == 20 and safe_set_bin == '6-15' and advantage_type == 'normalized'")[['state_variant','spearman','sign_agreement']]),
        '## Interpretation limits','',
        'Reliability masks alter the stochastic policy distribution even though they are not Actor MLP inputs. In-flight work is physically committed to server queues but is absent from the S0 CPU-backlog vector until upload completes. The audit tests whether adding these decision-time features predicts one realized frozen-policy return and whether the corresponding fitted-value GAE better aligns with the already-created matched counterfactual reference. It does not prove a POMDP or a causal explanation for the benchmark gap.',
        'Full feature, regression, stratified prediction, aliasing, case-study and model outputs are in this directory. Future queue changes and latency are labels for diagnosis only; no future information enters S0–S4.',
        '']
    (OUT/'STATE_SUFFICIENCY_AUDIT.md').write_text('\n'.join(report))
    (OUT/'state_sufficiency_decision.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
