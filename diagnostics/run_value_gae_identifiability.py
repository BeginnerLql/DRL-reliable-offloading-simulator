"""Offline value/GAE audit CLI. No PPO training; every stage is restartable."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from diagnostics.value_gae_identifiability import *
from diagnostics.run_long_horizon_coupling import _interval_returns
from agents.ppo_agent import PPOValueNetwork


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'frozen_policy_dataset_manifest.json'
    if path.exists():return json.loads(path.read_text())
    split=split_episodes();seed_plan=episode_seed_plan();seed_plan.to_csv(OUT/'episode_seed_plan.csv',index=False)
    matched_indices(split).to_csv(OUT/'matched_state_manifest.csv',index=False)
    tracked=subprocess.check_output(['git','ls-files','core','agents','config','Project_main.py',
        'diagnostics/results/external_baselines','diagnostics/results/external_baselines_rerun',
        'diagnostics/results/advantage_side_learner','diagnostics/results/long_horizon_coupling',
        'diagnostics/results/q_credit_audit','diagnostics/results/action_value_critic'],cwd=ROOT,text=True).splitlines()
    protected={f:sha256_file(ROOT/f) for f in tracked if (ROOT/f).is_file()}
    for p in [CHECKPOINT/'actor.pt',CHECKPOINT/'critic.pt',ROOT/'data/server_info.xlsx',ROOT/'data/task_parameters.xlsx']:
        protected[str(p.relative_to(ROOT))]=sha256_file(p)
    (OUT/'protected_hashes.json').write_text(json.dumps(protected,indent=2))
    manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        formal_trial=0,episodes=300,tasks_per_episode=200,transitions=60000,split=split,
        split_seed=20260927,matched_count=1000,matched_selection='Even positions across 60 TEST episodes; index list frozen before collection/fit/Q_H',
        actor_sha256=sha256_file(CHECKPOINT/'actor.pt'),critic_sha256=sha256_file(CHECKPOINT/'critic.pt'),
        checkpoint=str(CHECKPOINT.relative_to(ROOT)),gamma=params.gamma_ppo,gae_lambda=params.gae_lambda_ppo,
        actual_production_value_loss='value_loss_coef * MSE; NOT Huber',
        raw_observation_order='base hazards[8], processing frequencies[8], backlog seconds[8], uplink Mbps[8], input MB, demand MI, R_req',
        normalized_observation_key='states (production EnvironmentState.get_state)',
        episode_seeds='Independent substreams derived from formal trial-0 evaluation seeds; listed in episode_seed_plan.csv',
        value_budget=dict(episode_updates=300,epochs_per_update=params.k_epochs_ppo,batch_size=params.batch_size_ppo,
                          optimizer='Adam',lr=params.critic_lr_ppo,initialization_seed=2026092701,
                          train_episode_cycle='fixed permutation of 180 train episodes, repeated to 300; 2400 gradient updates',
                          gradient_clipping='same 0.5 threshold on side critic only; production joint Actor+V clipping cannot be replicated without Actor gradients'),
        v1_target='Production bootstrapped GAE-return recomputed once per episode update with fresh V; NOT TD(0)',
        v4='Not added: production loss and V2 already use MSE, so V4 would duplicate V2',
        huber_audit='Hypothetical delta=1 saturation only; not an active production loss mechanism',
        go_preregistered=dict(value_mae_relative_improvement=.10,value_r2_improvement=.05,value_ev_improvement=.05,
                             value_pearson_improvement=.10,overall_gae_spearman_or_sign_improvement=.05,
                             coupled_gae_spearman_or_sign_improvement=.02,
                             requires_validation_and_test_value_improvement=True,
                             primary_alignment='normalized GAE vs production-consistent SMDP Q_H, H20',
                             arm_selection='highest validation R2 among V1/V2/V3, before matched Q_H',
                             identifiability_low_r2_threshold=.10),
        reward_lens_note='Primary Q_H uses task-origin rewards and gamma**delta_t; legacy interval/gamma**step Q_H also saved as a sensitivity lens; no Q_H trains any model')
    path.write_text(json.dumps(manifest,indent=2));return manifest


def collect():
    manifest=prepare();dest=OUT/'dataset';dest.mkdir(exist_ok=True)
    agent,rho=make_agent();before=model_hash(agent.policy_old);vbefore=model_hash(agent.value_net)
    matched=pd.read_csv(OUT/'matched_state_manifest.csv');rows=[]
    for seeds in pd.read_csv(OUT/'episode_seed_plan.csv').to_dict('records'):
        ep=seeds['episode'];path=dest/f'episode_{ep:03d}.npz'
        if not path.exists():
            ids=matched.loc[matched.episode==ep,'task_id'].tolist()
            data,loop=run_episode(agent,seeds,capture_ids=ids)
            data.update(episode=np.full(200,ep),decision_index=np.arange(1,201),pair_correlation=rho,
                        selected_pair=np.asarray(agent.pairs)[data['actions']],
                        selected_probability=data['probabilities'][np.arange(200),data['actions']],
                        selected_reliability=data['reliabilities'][np.arange(200),data['actions']],
                        safe_set_empty=~data['safe_mask'].any(1),
                        terminal_type=np.where(data['done'],'natural_episode_task_resolution','next_arrival'),
                        policy_checkpoint_id=np.asarray(manifest['actor_sha256']))
            assignment=pd.DataFrame(loop.task_Assignments_info,columns=TASK_ASSIGNMENT_COLUMNS).sort_values('task_id')
            if not np.array_equal(data['rewards'],assignment.Task_Reward.to_numpy()):raise RuntimeError('Task reward attribution mismatch')
            data['task_delay']=assignment.Task_Delay.to_numpy()
            np.savez_compressed(path,**data)
        with np.load(path) as z:d={k:z[k] for k in z.files}
        g=d['mc_return'];r=d['rewards'];dt=d['delta_t'];done=d['done']
        residual=np.max(np.abs(g-r-params.gamma_ppo**dt*np.r_[g[1:],0.]*(~done)))
        rows.append(dict(episode=ep,number_of_transitions=len(r),terminal_count=int(done.sum()),
            max_recursion_residual=residual,missing_reward_count=int((~np.isfinite(r)).sum()),
            invalid_delta_t_count=int(((dt<0)|~np.isfinite(dt)).sum()),cross_episode_leakage_flag=False,
            terminal_time=float(d['terminal_time']),replica_drain_time=float(d['replica_drain_time']),
            pending_tasks=0,completed_replicas=400))
        if len(r)!=200 or done.sum()!=1 or not done[-1] or residual>1e-8:raise RuntimeError('MC integrity failed')
        if ep%20==0:print(f'collected {ep}/300 frozen episodes',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'mc_return_integrity_checks.csv',index=False)
    data=load_dataset();state=torch.tensor(data['states'][:64],dtype=torch.float32);mask=torch.tensor(data['effective_mask'][:64])
    with torch.no_grad():prob=torch.distributions.Categorical(logits=masked_logits(agent.policy_old(state),mask)).probs.numpy()
    integrity=dict(actor_file_hash_before=manifest['actor_sha256'],actor_file_hash_after_collection=sha256_file(CHECKPOINT/'actor.pt'),
                   actor_parameter_hash_before=before,actor_parameter_hash_after_collection=model_hash(agent.policy_old),
                   critic_parameter_hash_before=vbefore,critic_parameter_hash_after_collection=model_hash(agent.value_net),
                   policy_parameter_mismatch_count=int(before!=model_hash(agent.policy_old)),
                   action_distribution_probe_hash_before_side_training=hashlib.sha256(prob.tobytes()).hexdigest())
    (OUT/'frozen_policy_integrity.json').write_text(json.dumps(integrity,indent=2))
    # Labels/strata fixed before all counterfactual outcomes.
    maxima=data['backlog'].max(1);cuts=np.quantile(maxima,[1/3,2/3])
    matched['dataset_index']=(matched.episode-1)*200+matched.task_id-1
    idx=matched.dataset_index.to_numpy();matched['max_backlog']=maxima[idx]
    matched['load']=np.where(maxima[idx]<=cuts[0],'low',np.where(maxima[idx]<=cuts[1],'medium','high'))
    matched['requirement']=data['requirement'][idx];sizes=data['safe_mask'][idx].sum(1)
    matched['safe_set_bin']=np.where(sizes==0,'empty',np.where(sizes<=5,'1-5',np.where(sizes<=15,'6-15','16-28')))
    matched['safe_set_size']=sizes;matched['effective_set_size']=data['effective_mask'][idx].sum(1)
    matched.to_csv(OUT/'matched_state_manifest.csv',index=False)
    manifest['load_tertile_cuts']=cuts.tolist();manifest['load_definition']='Maximum backlog seconds, empirical dataset tertiles; same statistic/rule as historical diagnostic'
    manifest['dataset_complete']=True;manifest['dataset_file_hashes']={p.name:sha256_file(p) for p in sorted(dest.glob('*.npz'))}
    (OUT/'frozen_policy_dataset_manifest.json').write_text(json.dumps(manifest,indent=2))


def load_dataset():
    paths=sorted((OUT/'dataset').glob('episode_*.npz'))
    if len(paths)!=300:raise RuntimeError('Need exactly 300 frozen episodes')
    chunks=[]
    for path in paths:
        with np.load(path) as z:chunks.append({k:z[k] for k in z.files if z[k].ndim>0 and z[k].shape[0]==200})
    return {k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]}


def _predict(net,x):
    with torch.no_grad():return np.concatenate([net(x[i:i+2048]).numpy() for i in range(0,len(x),2048)])


def fit():
    meta=prepare();data=load_dataset();ep=data['episode'];y=data['mc_return'];x=torch.tensor(data['states'],dtype=torch.float32)
    masks={s:np.isin(ep,ids) for s,ids in meta['split'].items()};mu,sd=train_normalization(y,ep,meta['split']['train'])
    results=[];curves=[];gradients=[];saturations=[];predictions={'V0':data['original_value']}
    order=np.random.default_rng(2026092702).permutation(meta['split']['train']);units=np.resize(order,300)
    out=OUT/'side_models';out.mkdir(exist_ok=True)
    for arm in ['V1','V2','V3','SmallMLP']:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026092701)
            net=PPOValueNetwork(params.num_states,[128,64] if arm=='SmallMLP' else params.hidden_layers_ppo,
                                activation='relu' if arm=='SmallMLP' else params.af_ppo)
        optimizer=torch.optim.Adam(net.parameters(),lr=params.critic_lr_ppo)
        generator=np.random.default_rng(2026092703);step=0;standard=arm in ('V3','SmallMLP')
        def evaluate(unit):
            native=_predict(net,x);pred=inverse_target(native,mu,sd) if standard else native
            for split,mask in masks.items():
                curves.append(dict(arm=arm,unit=unit,gradient_updates=step,split=split,**value_metrics(pred[mask],y[mask])))
            train=masks['train'];scale_target=normalize_target(y[train],mu,sd) if standard else y[train]
            saturations.append(dict(arm=arm,unit=unit,active_loss='MSE',huber_is_hypothetical=True,
                **saturation(native[train],scale_target),raw_unit_saturation=float((np.abs(pred[train]-y[train])>1).mean())))
        evaluate(0)
        for unit,episode in enumerate(units,1):
            ids=np.flatnonzero(ep==episode);states=x[ids]
            if arm=='V1':
                v=_predict(net,states);nv=_predict(net,torch.tensor(data['next_states'][ids],dtype=torch.float32))
                _,_,_,target=production_gae(data['rewards'][ids]*params.reward_scale_ppo,data['delta_t'][ids],data['done'][ids],v,nv,params.gamma_ppo,params.gae_lambda_ppo)
            else:target=normalize_target(y[ids],mu,sd) if standard else y[ids]
            target=torch.tensor(target,dtype=torch.float32)
            for _ in range(params.k_epochs_ppo):
                perm=generator.permutation(len(ids))
                for start in range(0,len(ids),params.batch_size_ppo):
                    b=perm[start:start+params.batch_size_ppo];bx=states[b]
                    before=torch.cat([p.detach().flatten() for p in net.parameters()]).clone()
                    pred=net(bx);old_pred=pred.detach().clone();loss=params.value_loss_coef_ppo*nn.functional.mse_loss(pred,target[b])
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    gn=nn.utils.clip_grad_norm_(net.parameters(),params.max_grad_norm_ppo)
                    optimizer.step();step+=1
                    with torch.no_grad():
                        update=torch.cat([p.detach().flatten() for p in net.parameters()])-before
                        shift=(net(bx)-old_pred).abs().mean()
                    gradients.append(dict(arm=arm,unit=unit,step=step,loss=float(loss.detach()),
                        gradient_norm_before_clip=float(gn),parameter_update_norm=float(update.norm()),prediction_shift=float(shift)))
            if unit%10==0:evaluate(unit)
        native=_predict(net,x);predictions[arm]=inverse_target(native,mu,sd) if standard else native
        torch.save(net.state_dict(),out/f'{arm}.pt')
        print(f'{arm} completed {step} equal-budget updates',flush=True)
    tx=data['states'][masks['train']].astype(float);feature_mean=tx.mean(0);feature_scale=tx.std(0)
    feature_scale=np.where(feature_scale<1e-8,1.,feature_scale)
    design=(data['states'].astype(float)-feature_mean)/feature_scale;train_design=design[masks['train']]
    coef=np.linalg.solve(train_design.T@train_design+np.eye(design.shape[1]),train_design.T@normalize_target(y[masks['train']],mu,sd))
    predictions['Ridge']=inverse_target(design@coef,mu,sd)
    np.savez_compressed(out/'ridge.npz',feature_mean=feature_mean,feature_scale=feature_scale,coef=coef,intercept=0.,target_mean=mu,target_std=sd)
    for arm,pred in predictions.items():
        for split,mask in masks.items():results.append(dict(arm=arm,split=split,n=int(mask.sum()),**value_metrics(pred[mask],y[mask])))
    table=pd.DataFrame(results);table.to_csv(OUT/'value_fit_main_results.csv',index=False)
    table[table.arm.isin(['Ridge','SmallMLP'])].to_csv(OUT/'state_identifiability_baselines.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'value_fit_training_curves.csv',index=False)
    pd.DataFrame(gradients).to_csv(OUT/'gradient_update_diagnostics.csv',index=False)
    pd.DataFrame(saturations).to_csv(OUT/'huber_saturation_diagnostics.csv',index=False)
    np.savez_compressed(OUT/'value_predictions.npz',**predictions)
    normalization=dict(train_mean=mu,train_std=sd,epsilon=1e-8,statistics_episode_ids=meta['split']['train'],
                       gradient_updates_per_arm=2400,v4='omitted: identical to V2 because current loss is MSE')
    eligible=table[(table.split=='validation')&table.arm.isin(['V1','V2','V3'])]
    normalization['best_validation_arm']=str(eligible.sort_values('r2',ascending=False).iloc[0].arm)
    (OUT/'value_fit_metadata.json').write_text(json.dumps(normalization,indent=2))
    # No labels from branches have been read by this function.
    gaes={}
    for arm in ['V0','V1','V2','V3']:
        v=predictions[arm];nv=np.zeros_like(v)
        for episode in range(1,301):
            ids=np.flatnonzero(ep==episode);nv[ids[:-1]]=v[ids[1:]]
            td,a,n,ret=production_gae(data['rewards'][ids],data['delta_t'][ids],data['done'][ids],v[ids],nv[ids],params.gamma_ppo,params.gae_lambda_ppo)
            if arm not in gaes:gaes[arm]=np.empty(len(ep));gaes[arm+'_normalized']=np.empty(len(ep))
            gaes[arm][ids]=a;gaes[arm+'_normalized'][ids]=n
    if not np.array_equal(gaes['V0'],data['original_gae']):raise RuntimeError('Original GAE recompute mismatch')
    np.savez_compressed(OUT/'offline_gae.npz',**gaes)


def counterfactual_episode(episode):
    torch.set_num_threads(1)
    dest=OUT/'counterfactual';dest.mkdir(exist_ok=True);path=dest/f'episode_{episode:03d}.csv'
    if path.exists():return episode,'cached'
    seeds=pd.read_csv(OUT/'episode_seed_plan.csv').set_index('episode').loc[episode].to_dict()
    selected=pd.read_csv(OUT/'matched_state_manifest.csv');selected=selected[selected.episode==episode]
    with np.load(OUT/'dataset'/f'episode_{episode:03d}.npz') as z:ref={k:z[k] for k in z.files}
    agent,rho=make_agent();before=model_hash(agent.policy_old)
    # Replay the unforced entire episode once using the unmodified feasibility path.
    replay,loop=run_episode(agent,seeds,capture_ids=selected.task_id.tolist())
    for key in ['states','actions','probabilities','effective_mask','backlog','task_parameters','rewards','delta_t','rng_trace']:
        if not np.array_equal(replay[key],ref[key]):raise RuntimeError(f'Unforced replay mismatch {episode} {key}')
    rows=[];checks=[]
    for record in selected.to_dict('records'):
        tid=record['task_id'];i=tid-1;actions=np.flatnonzero(ref['effective_mask'][i]);snapshot=agent.snapshots.get(tid)
        lat=ref['latencies'][i];myopic=int(actions[np.argmin(lat[actions])])
        for action in actions:
            branch,loop=run_episode(agent,seeds,details=False,capture_ids=[tid],forced=(tid,int(action)),cache=ref)
            check=replay_checks(agent.snapshots[tid],ref,tid)
            check['action_prefix_mismatch']=int(not np.array_equal(branch['actions'][:i],ref['actions'][:i]))
            check['future_arrival_mismatch']=int(not np.array_equal(branch['arrival_trace'],ref['arrival_trace']))
            check['future_torch_rng_mismatch']=int(not np.array_equal(branch['rng_trace'],ref['rng_trace']))
            if any(check.values()):raise RuntimeError(f'Matched branch mismatch {episode} {tid}: {check}')
            check.update(episode=episode,task_id=tid,action_index=int(action),state_id=record['state_id']);checks.append(check)
            q=truncated_smdp_returns(branch,tid)
            assignments=pd.DataFrame(loop.task_Assignments_info,columns=TASK_ASSIGNMENT_COLUMNS)
            decisions=[dict(task_id=j+1,decision_time=float(t)) for j,t in enumerate(ref['time'])]
            legacy=_interval_returns(assignments,decisions,tid,float(loop.env.now),params.gamma_ppo)
            rows.append({**record,'action_index':int(action),'selected_action':int(ref['actions'][i]),
                         'myopic_action':myopic,'probability':float(ref['probabilities'][i,action]),
                         **{f'q_h{h}':q[h] for h in HORIZONS},
                         **{f'legacy_q_h{h}':legacy[h]['q_return'] for h in HORIZONS}})
    if before!=model_hash(agent.policy_old):raise RuntimeError('Frozen Actor changed')
    pd.DataFrame(checks).to_csv(dest/f'episode_{episode:03d}_checks.csv',index=False)
    pd.DataFrame(rows).to_csv(path,index=False)
    return episode,len(rows)


def branch(workers):
    manifest=prepare()
    if not manifest.get('dataset_complete'):raise RuntimeError('Collect dataset first')
    # Indices have already been frozen; fit labels and counterfactuals stay disjoint.
    episodes=pd.read_csv(OUT/'matched_state_manifest.csv').episode.unique().tolist()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks=[pool.submit(counterfactual_episode,int(ep)) for ep in episodes]
        for i,task in enumerate(as_completed(tasks),1):print('counterfactual episode',task.result(),f'{i}/{len(tasks)}',flush=True)
    frames=[pd.read_csv(OUT/'counterfactual'/f'episode_{ep:03d}.csv') for ep in episodes]
    checks=[pd.read_csv(OUT/'counterfactual'/f'episode_{ep:03d}_checks.csv') for ep in episodes]
    pd.concat(frames,ignore_index=True).to_csv(OUT/'matched_counterfactual_qh.csv',index=False)
    pd.concat(checks,ignore_index=True).to_csv(OUT/'matched_state_replay_checks.csv',index=False)


def verify_protected():
    protected=json.loads((OUT/'protected_hashes.json').read_text())
    bad=[path for path,digest in protected.items() if sha256_file(ROOT/path)!=digest]
    if bad:raise RuntimeError(f'Protected files changed: {bad}')
    agent,_=make_agent();integrity=json.loads((OUT/'frozen_policy_integrity.json').read_text());data=load_dataset()
    with torch.no_grad():p=torch.distributions.Categorical(logits=masked_logits(agent.policy_old(torch.tensor(data['states'][:64],dtype=torch.float32)),torch.tensor(data['effective_mask'][:64]))).probs.numpy()
    integrity.update(actor_file_hash_after_side_training=sha256_file(CHECKPOINT/'actor.pt'),
        critic_file_hash_after_side_training=sha256_file(CHECKPOINT/'critic.pt'),
        actor_parameter_hash_after_side_training=model_hash(agent.policy_old),
        critic_parameter_hash_after_side_training=model_hash(agent.value_net),
        action_distribution_probe_hash_after_side_training=hashlib.sha256(p.tobytes()).hexdigest(),protected_file_mismatch_count=len(bad))
    if integrity['actor_parameter_hash_before']!=integrity['actor_parameter_hash_after_side_training'] or integrity['critic_parameter_hash_before']!=integrity['critic_parameter_hash_after_side_training']:raise RuntimeError('Frozen model parameters changed')
    if integrity['action_distribution_probe_hash_before_side_training']!=integrity['action_distribution_probe_hash_after_side_training']:raise RuntimeError('Probe distribution changed')
    (OUT/'frozen_policy_integrity.json').write_text(json.dumps(integrity,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','collect','fit','branch','verify'])
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();torch.set_num_threads(1)
    if args.stage=='prepare':prepare()
    elif args.stage=='collect':collect()
    elif args.stage=='fit':fit()
    elif args.stage=='branch':branch(args.workers)
    else:verify_protected()
