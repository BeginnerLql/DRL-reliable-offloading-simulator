"""Diagnostic-only frozen-policy data, natural-episode MC and prefix replay.

Production modules are imported, never changed. Historical interval Q_H is
reported separately from task-credit, elapsed-time-discounted Q_H.
"""
from __future__ import annotations
import copy
import hashlib
import io
import json
import random
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import agents.masked_pair_ppo_agent as masked_module
from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent, masked_logits
from config.params import params
from diagnostics.evaluate_reliability_masked_policy import ArrivalTraceLoop
from diagnostics.run_long_horizon_coupling import queue_hazard_snapshot, stable_digest, _interval_returns
from diagnostics.external_reliability_baselines import estimate_pair_completion_latencies
from diagnostics.run_masked_pair_ppo_10seed import OUT as FORMAL, formal_spec
from tools.paired_ppo_experiment import _agent_kwargs, scoped_environment_seeds, sha256_file
from tools.pair_policy_diagnostics import TASK_ASSIGNMENT_COLUMNS
from Project_main import build_pair_correlations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostics/results/value_gae_identifiability'
HORIZONS = (5, 10, 20, 50)  # evaluation only
CHECKPOINT = FORMAL / 'runs/trial_000/masked'
EXCEL = {}


def model_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def smdp_mc(rewards, delta_t, done, gamma):
    r, dt, terminal = np.asarray(rewards, float), np.asarray(delta_t, float), np.asarray(done, bool)
    if r.ndim != 1 or r.shape != dt.shape or r.shape != terminal.shape or not len(r):
        raise ValueError('MC inputs must be matching nonempty vectors')
    if not np.isfinite(r).all() or not np.isfinite(dt).all() or (dt < 0).any() or not terminal[-1]:
        raise ValueError('Invalid reward, delta_t or missing final terminal')
    result = np.empty_like(r); following = 0.0
    for k in reversed(range(len(r))):
        following = r[k] + gamma ** dt[k] * (0.0 if terminal[k] else following)
        result[k] = following
    return result


def production_gae(rewards, delta_t, done, values, next_values, gamma, lam):
    """Exact float32 recurrence/normalization used by masked PPO; one episode."""
    r, dt, d, v, nv = [torch.as_tensor(np.array(x, copy=True), dtype=torch.float32)
                        for x in (rewards, delta_t, done, values, next_values)]
    discount = torch.pow(torch.full_like(dt, float(gamma)), dt)
    residual = r + discount * nv * (1-d) - v
    a = torch.zeros_like(r); running = torch.tensor(0., dtype=torch.float32)
    for k in reversed(range(len(r))):
        running = residual[k] + discount[k] * lam * (1-d[k]) * running
        a[k] = running
    std = a.std(unbiased=False)
    normalized = a-a.mean() if std.item() < 1e-8 else (a-a.mean())/(std+1e-8)
    return residual.numpy(), a.numpy(), normalized.numpy(), (a+v).numpy()


def split_episodes(seed=20260927):
    order = np.random.default_rng(seed).permutation(np.arange(1,301))
    return dict(train=sorted(order[:180].tolist()), validation=sorted(order[180:240].tolist()),
                test=sorted(order[240:].tolist()))


def train_normalization(target, episode, train_ids):
    x = np.asarray(target)[np.isin(episode, train_ids)]
    return float(x.mean()), float(x.std())


def normalize_target(target, mean, std):
    return (np.asarray(target)-mean)/(std+1e-8)


def inverse_target(values, mean, std):
    return np.asarray(values)*(std+1e-8)+mean


def correlation(a,b, rank=False):
    a,b=np.asarray(a,float),np.asarray(b,float)
    ok=np.isfinite(a)&np.isfinite(b);a,b=a[ok],b[ok]
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return float('nan')
    return float(spearmanr(a,b).statistic if rank else np.corrcoef(a,b)[0,1])


def value_metrics(pred,target):
    p,y=np.asarray(pred,float),np.asarray(target,float); err=p-y; var=np.var(y)
    slope=float(np.cov(p,y,ddof=0)[0,1]/np.var(p)) if np.var(p)>1e-15 else float('nan')
    return dict(mae=float(np.abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))),
                pearson=correlation(p,y),spearman=correlation(p,y,True),
                r2=float(1-np.mean(err**2)/var),explained_variance=float(1-np.var(err)/var),
                calibration_slope=slope,calibration_intercept=float(y.mean()-slope*p.mean()),
                prediction_mean=float(p.mean()),prediction_std=float(p.std()),
                target_mean=float(y.mean()),target_std=float(y.std()))


def saturation(pred,target,delta=1.):
    error=np.asarray(pred,float)-np.asarray(target,float); absolute=np.abs(error)
    return dict(hypothetical_huber_saturation_fraction=float(np.mean(absolute>delta)),
                residual_mean=float(error.mean()),residual_std=float(error.std()),
                **{f'absolute_residual_p{q}':float(np.percentile(absolute,q)) for q in (50,90,95,99)})


def episode_seed_plan():
    _,plan=formal_spec(); seed=plan.iloc[0]
    rows=[]
    for episode in range(1,301):
        seeds=np.random.SeedSequence([int(seed.Eval_Arrival_Seed),int(seed.Eval_Spatial_Seed),
                                     20260927,episode]).generate_state(3)
        rows.append(dict(episode=episode,arrival_seed=int(seeds[0]),spatial_seed=int(seeds[1]),action_seed=int(seeds[2])))
    return pd.DataFrame(rows)


def matched_indices(split):
    """Select only held-out test episodes before seeing any Q_H or fitted model."""
    rows=[]
    for i,episode in enumerate(split['test']):
        count=17 if i<40 else 16
        for task in np.linspace(1,200,count).round().astype(int):
            rows.append(dict(episode=episode,task_id=int(task),state_id=f'ep{episode:03d}_task{task:03d}',split='test'))
    return pd.DataFrame(rows)


class FrozenAuditAgent(ReliabilityMaskedPairPPOAgent):
    """Local observer/intervention wrapper; no optimizer is called."""
    def reset_audit(self, details=True, capture_ids=(), forced=None):
        self.clear_rollout();self.selection_archive=[];self.records=[];self.capture_ids=set(capture_ids)
        self.details=details;self.forced=forced;self.snapshots={};self.rng_trace=[];self.data=None

    def select_action(self,state,*args,**kwargs):
        c=self.current_decision;task,env=self.audit_context;tid=int(task.id)
        record=None
        self.rng_trace.append(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
        if self.details or tid in self.capture_ids:
            with torch.no_grad():
                logits=self.policy_old(self._to_tensor(state).unsqueeze(0)).squeeze(0)
                ml=masked_logits(logits,c['effective_mask']); probs=torch.distributions.Categorical(logits=ml).probs.numpy()
            servers=[env.get_server_by_id(sid) for sid in sorted(env.servers)]
            backlog=np.asarray([env.get_server_backlog_time(s.server_id,task.env.now) for s in servers])
            frequencies=np.asarray([s.processing_frequency for s in servers]);uplink=np.asarray([s.uplink_rate_mbps for s in servers])
            base=np.asarray([s.base_failure_rate for s in servers])
            raw=np.concatenate([base,frequencies,backlog,uplink,[task.input_data_size_mb,task.computation_demand,task.reliability_requirement]])
            latencies,_=estimate_pair_completion_latencies(task,env,self.pairs)
            record=dict(state=np.asarray(state).copy(),raw_observation=raw,time=float(task.env.now),
                        safe_mask=c['safe_mask'].copy(),effective_mask=c['effective_mask'].copy(),
                        probabilities=probs.copy(),masked_logits=ml.numpy().copy(),
                        reliabilities=c['reliabilities'].copy(),requirement=task.reliability_requirement,
                        backlog=backlog,uplink=uplink,frequencies=frequencies,
                        effective_rates=np.asarray([env.get_active_failure_rate(s.server_id) for s in servers]),
                        latencies=latencies.copy(),task_parameters=np.asarray([task.input_data_size_mb,task.computation_demand,task.reliability_requirement]))
        if tid in self.capture_ids:
            record['snapshot_digest']=stable_digest(queue_hazard_snapshot(self.loop,task,env,state))
            self.snapshots[tid]=record
        action=super().select_action(state,*args,**kwargs)  # consumes the same draw even when intervening
        if self.forced is not None and tid==self.forced[0]:
            action=int(self.forced[1]);mask,_,_,s=self.pending_decisions[tid]
            if not mask[action]:raise RuntimeError('Forced action outside effective support')
            probability=float(record['probabilities'][action])
            self.pending_decisions[tid]=(mask,float(np.log(probability)),action,s)
            self.selection_archive[-1]['action_index']=action
        if self.details:
            record['snapshot_digest']=record.get('snapshot_digest','')
            record['action']=action;self.records.append(record)
        return action

    def prepare_action(self,task,env,episode,state):
        self.audit_context=(task,env)
        super().prepare_action(task,env,episode,state)

    def train_step(self):
        if any(r is None for r in self.rewards) or self.pending_decisions or self.pending_task_rewards:
            raise RuntimeError('Unresolved frozen rollout reward')
        self.data={k:np.asarray(v).copy() for k,v in dict(states=self.states,next_states=self.next_states,
            actions=self.actions,rewards=self.rewards,delta_t=self.delta_times,done=self.dones,task_ids=self.task_ids).items()}
        if self.details:
            for key in self.records[0]:
                self.data[key]=np.asarray([r[key] for r in self.records])
        super().train_step()  # frozen=True: validation and clear only


def make_agent():
    _,plan=formal_spec();pairs,rho=build_pair_correlations()
    agent=FrozenAuditAgent(**_agent_kwargs('pair_scoring',np.asarray(rho),int(plan.iloc[0].PPO_Minibatch_Seed)),
                           frozen=True,deployment_mode='stochastic')
    actor=torch.load(CHECKPOINT/'actor.pt',weights_only=True,map_location='cpu')
    value=torch.load(CHECKPOINT/'critic.pt',weights_only=True,map_location='cpu')
    agent.policy_net.load_state_dict(actor);agent.policy_old.load_state_dict(actor);agent.value_net.load_state_dict(value)
    agent.policy_old.eval();agent.value_net.eval()
    for model in (agent.policy_net,agent.policy_old,agent.value_net):
        for p in model.parameters():p.requires_grad_(False)
    return agent,np.asarray(rho)


def run_episode(agent,seeds,details=True,capture_ids=(),forced=None,cache=None,max_tasks=200):
    if not EXCEL:
        for p in (ROOT/'data/task_parameters.xlsx',ROOT/'data/server_info.xlsx'):EXCEL[p.resolve()]=pd.read_excel(p)
    original_read=pd.read_excel
    def read(path,*args,**kwargs):
        return EXCEL[Path(path).resolve()] if Path(path).resolve() in EXCEL else original_read(path,*args,**kwargs)
    original_reliability=masked_module.production_reliability_vector
    def reliability(task,env,pairs):
        if cache is None:return original_reliability(task,env,pairs)
        i=int(task.id)-1
        physical=np.asarray([env.get_active_failure_rate(sid) for sid in sorted(env.servers)])
        if i==0 and not np.array_equal(physical,cache['effective_rates'][0]):raise RuntimeError('Cached hazards changed')
        if not np.array_equal(np.asarray([task.input_data_size_mb,task.computation_demand,task.reliability_requirement]),cache['task_parameters'][i]):
            raise RuntimeError('Cached task changed')
        return cache['reliabilities'][i].copy()
    agent.reset_audit(details,capture_ids,forced)
    # Every episode has explicit independent substreams derived from one formal seed.
    random.seed(int(seeds['action_seed']));np.random.seed(int(seeds['action_seed']))
    torch.manual_seed(int(seeds['action_seed']))
    with patch.object(pd,'read_excel',read),patch.object(masked_module,'production_reliability_vector',reliability):
        with scoped_environment_seeds(int(seeds['arrival_seed']),int(seeds['spatial_seed'])):
            loop=ArrivalTraceLoop(agent,1,max_tasks,params.num_states,params.num_actions);agent.loop=loop
            with redirect_stdout(io.StringIO()):loop.EP()
    if loop.pendingList or loop.env_state.num_resolved_tasks!=max_tasks or loop.env_state.num_completed_replicas!=2*max_tasks:
        raise RuntimeError('Episode failed to drain tasks/replicas')
    data=agent.data
    data['arrival_trace']=np.asarray(loop.interarrival_trace);data['rng_trace']=np.asarray(agent.rng_trace)
    data['terminal_time']=np.asarray(loop._get_ppo_terminal_time());data['replica_drain_time']=np.asarray(loop.env.now)
    if details:
        with torch.no_grad():
            v=agent.value_net(torch.tensor(data['states'],dtype=torch.float32)).numpy()
            nv=agent.value_net(torch.tensor(data['next_states'],dtype=torch.float32)).numpy()
        td,gae,norm,ret=production_gae(data['rewards']*params.reward_scale_ppo,data['delta_t'],data['done'],v,nv,params.gamma_ppo,params.gae_lambda_ppo)
        data.update(original_value=v,original_next_value=nv,original_td=td,original_gae=gae,
                    original_normalized_gae=norm,original_gae_return=ret,
                    mc_return=smdp_mc(data['rewards'],data['delta_t'],data['done'],params.gamma_ppo))
    return data,loop


def replay_checks(observed,reference,tid):
    i=tid-1
    result={}
    for name,key in [('state','states'),('mask','effective_mask'),('probability','probabilities'),
                     ('backlog','backlog'),('task','task_parameters')]:
        x=observed[key] if key in observed else observed[{'states':'state'}[key]]
        result[name+'_mismatch']=int(not np.array_equal(x,reference[key][i]))
    result['snapshot_mismatch']=int(observed['snapshot_digest']!=reference['snapshot_digest'][i])
    return result


def truncated_smdp_returns(data,tid):
    r=np.asarray(data['rewards']);dt=np.asarray(data['delta_t']);i=tid-1;result={}
    for h in HORIZONS:
        end=min(len(r),i+h);weights=np.r_[1.,np.cumprod(params.gamma_ppo**dt[i:end-1])]
        result[h]=float(weights@r[i:end])
    return result
