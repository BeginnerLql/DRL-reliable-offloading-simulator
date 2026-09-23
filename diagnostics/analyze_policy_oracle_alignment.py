"""Analyze a saved instrumented Pair PPO run against the production reward oracle.

All counterfactual rewards reuse the previously validated EpisodeTrace,
episode_rates, Task.initialize_reliability_evaluation, and MainLoop.calcReward.
No training/update function is called here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr

from agents.ppo_agent import PPOPairScoringPolicyNetwork, PPOValueNetwork
from config.params import params
from diagnostics.requirement_conditioning_diagnostic import (
    FORMAL, REQS, EpisodeTrace, episode_rates, simulator_reward,
)
from Project_main import build_pair_correlations
from tools.paired_ppo_experiment import sha256_file

DEFAULT_RUN=ROOT/"diagnostics/results/policy_oracle_alignment"
EXACT_TOL=1e-10


def vector(text):
    value=np.asarray(json.loads(text),dtype=float)
    if not np.isfinite(value).all(): raise ValueError("Non-finite vector")
    return value


def corr(x,y,kind="spearman"):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)<3 or np.ptp(x)<1e-12 or np.ptp(y)<1e-12: return np.nan
    if kind=="pearson": return float(pearsonr(x,y).statistic)
    if kind=="kendall": return float(kendalltau(x,y).statistic)
    return float(spearmanr(x,y).statistic)


def boot_ci(values,seed=2043,n=2000):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    if not len(values): return np.nan,np.nan
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(n,len(values)))].mean(axis=1)
    return tuple(map(float,np.quantile(means,[.025,.975])))


def summary_group(frame,columns,group="R_req"):
    rows=[]
    for req,g in frame.groupby(group):
        row={group:req,"N":len(g)}
        for column in columns:
            data=g[column].dropna().to_numpy(dtype=float)
            ci=boot_ci(data)
            row.update({f"{column}_mean":float(np.mean(data)) if len(data) else np.nan,
                        f"{column}_median":float(np.median(data)) if len(data) else np.nan,
                        f"{column}_p25":float(np.quantile(data,.25)) if len(data) else np.nan,
                        f"{column}_p75":float(np.quantile(data,.75)) if len(data) else np.nan,
                        f"{column}_p90":float(np.quantile(data,.90)) if len(data) else np.nan,
                        f"{column}_ci_low":ci[0],f"{column}_ci_high":ci[1]})
        rows.append(row)
    return pd.DataFrame(rows)


def load_actor(run:Path,rho):
    actor=PPOPairScoringPolicyNetwork(params.num_states,params.num_actions,params.hidden_layers_ppo,
        params.serverNo,rho,activation=params.af_ppo)
    critic=PPOValueNetwork(params.num_states,params.hidden_layers_ppo,activation=params.af_ppo)
    actor.load_state_dict(torch.load(run/"actor_final.pt",map_location="cpu",weights_only=True))
    critic.load_state_dict(torch.load(run/"critic_final.pt",map_location="cpu",weights_only=True))
    actor.eval();critic.eval()
    return actor,critic


def task_ids_by_tier(tasks,n_per_tier):
    result=[]
    for i,req in enumerate(REQS):
        group=tasks.index[np.isclose(tasks.Reliability_Requirement,req)].to_numpy(dtype=int)
        n=n_per_tier[i] if isinstance(n_per_tier,tuple) else n_per_tier
        result.extend(group[np.linspace(0,len(group)-1,n).round().astype(int)].tolist())
    assert len(result)==len(set(result))
    return sorted(result)


def oracle_for_traces(assignments:pd.DataFrame,states:pd.DataFrame,tasks:pd.DataFrame,
                      servers:pd.DataFrame,pairs:list,spatial_seed:int,selected_ids:list[int],source:str):
    rates_by_ep=episode_rates(spatial_seed,int(assignments.Episode.max()),servers)
    rows=[];surface_context=[]
    for ep,frame in assignments.groupby("Episode",sort=True):
        trace=EpisodeTrace(frame.rename(columns={"Episode":"episode","Task_ID":"task_id"}),tasks,servers)
        rates=rates_by_ep[int(ep)]
        for tid in selected_ids:
            selected=states[(states.Episode==ep)&(states.Task_ID==tid)]
            assert len(selected)==1
            observed=frame[frame.Task_ID==tid]
            assert len(observed)==1
            observed=observed.iloc[0];selected=selected.iloc[0]
            req=float(observed.Reliability_Requirement)
            assert np.isclose(req,float(selected.R_req),rtol=0,atol=1e-12)
            rewards=[]
            for action,pair in enumerate(pairs):
                delay=trace.candidate_delay(tid,pair)
                actual=simulator_reward(tid,float(tasks.loc[tid,"Computation_Demand"]),req,
                                        pair,delay,rates,servers)
                rewards.append(float(actual["reward_total"]))
                if action==int(observed.action_index):
                    assert pair==(int(observed.Primary),int(observed.Backup))
                    assert abs(delay-float(observed.Task_Delay))<=1e-8
                    assert abs(rewards[-1]-float(observed.Task_Reward))<=1e-8
            best=max(rewards)
            logits=vector(selected.raw_logits);probs=vector(selected.probabilities)
            assert len(logits)==len(probs)==len(rewards)==28
            assert abs(probs.sum()-1)<1e-6
            optimal=np.flatnonzero(best-np.asarray(rewards)<=EXACT_TOL)
            top=np.argsort(-logits,kind="stable")
            item={"source":source,"Episode":int(ep),"Task_ID":int(tid),"R_req":req,
                  "sampled_action":int(observed.action_index),"greedy_action":int(top[0]),
                  "sampled_regret":best-rewards[int(observed.action_index)],
                  "greedy_regret":best-rewards[int(top[0])],
                  "optimal_set_size":len(optimal),"top1_hit":int(top[0] in optimal),
                  "top3_hit":int(bool(set(top[:3])&set(optimal))),
                  "top5_hit":int(bool(set(top[:5])&set(optimal))),
                  "optimal_set_probability_mass":float(probs[optimal].sum()),
                  "uniform_optimal_set_mass":float(len(optimal)/28),
                  "optimal_mass_lift_over_uniform":float(probs[optimal].sum()-len(optimal)/28),
                  "spearman_logit_reward":corr(logits,rewards),
                  "kendall_logit_reward":corr(logits,rewards,"kendall"),
                  "spearman_probability_negative_regret":corr(probs,-(best-np.asarray(rewards)))}
            rows.append(item)
            if source=="training":
                continue
            if tid in selected_ids[:20]:
                surface_context.append({"Episode":int(ep),"Task_ID":int(tid),"delay_78":trace.candidate_delay(tid,(7,8)),
                                        "demand":float(tasks.loc[tid,"Computation_Demand"]),"rates":rates.copy()})
    return pd.DataFrame(rows),surface_context


def untrained_bias(states:pd.DataFrame,rho,out:Path):
    batch=torch.as_tensor(np.stack(states.state.map(vector)),dtype=torch.float32)
    counts=np.zeros(28,dtype=int);per_seed=[]
    seeds=np.random.default_rng(2026).integers(0,2**31,size=100)
    for index,seed in enumerate(seeds):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            actor=PPOPairScoringPolicyNetwork(params.num_states,params.num_actions,
                params.hidden_layers_ppo,params.serverNo,rho,activation=params.af_ppo)
        with torch.no_grad():
            top=actor(batch).argmax(dim=1).numpy()
        counts+=np.bincount(top,minlength=28)
        per_seed.append({"Seed_Index":index,"Top1_Pair_78_Fraction":float(np.mean(top==27))})
    table=pd.DataFrame({"action_index":range(28),"pair":[str(pair) for pair in build_pair_correlations()[0]],
                        "greedy_top1_count":counts,"greedy_top1_fraction":counts/counts.sum()})
    table.to_csv(out/"untrained_actor_top1_frequency.csv",index=False)
    pd.DataFrame(per_seed).to_csv(out/"untrained_actor_seed_summary.csv",index=False)
    return table


def rho_surface(actor,contexts:list,servers,rho,out:Path):
    rho_grid=np.linspace(float(np.min(rho)),float(np.max(rho)),31)
    rows=[]
    for context in contexts[:20]:
        for req in REQS:
            state=context["state"].copy()
            state[-1]=float((-np.log10(1-req)-1)/3)
            with torch.no_grad():
                features=actor.build_pair_features(torch.as_tensor(state,dtype=torch.float32))
            oracle_reward=simulator_reward(context["Task_ID"],context["demand"],req,(7,8),
                                          context["delay_78"],context["rates"],servers)["reward_total"]
            for value in rho_grid:
                with torch.no_grad():
                    changed=features.clone();changed[27,8]=float(value)
                    scores=actor.scorer(changed).flatten()
                    probability=torch.softmax(scores,dim=0)[27]
                rows.append({"Episode":context["Episode"],"Task_ID":context["Task_ID"],
                             "R_req":req,"rho":float(value),"actor_score":float(scores[27]),
                             "actor_probability":float(probability),"oracle_reward":float(oracle_reward)})
    frame=pd.DataFrame(rows);frame.to_csv(out/"rreq_rho_surface.csv",index=False)
    sensitivity=[]
    for req,g in frame.groupby("R_req"):
        slopes=[];oracle_slopes=[]
        for _,h in g.groupby(["Episode","Task_ID"]):
            h=h.sort_values("rho")
            slopes.extend(np.abs(np.diff(h.actor_score)/np.diff(h.rho)))
            oracle_slopes.extend(np.abs(np.diff(h.oracle_reward)/np.diff(h.rho)))
        sensitivity.append({"R_req":req,"mean_abs_actor_score_slope":float(np.mean(slopes)),
                            "mean_abs_oracle_reward_slope":float(np.mean(oracle_slopes))})
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(out/"rho_sensitivity_by_requirement.csv",index=False)
    grid=frame.groupby(["R_req","rho"]).actor_score.mean().unstack("rho")
    fig,ax=plt.subplots(figsize=(8,4))
    im=ax.imshow(grid.to_numpy(),aspect="auto",origin="lower",cmap="viridis")
    ax.set_yticks(range(4),[str(x) for x in REQS]);ax.set_xticks([0,10,20,30],[f"{rho_grid[x]:.3f}" for x in (0,10,20,30)])
    ax.set(xlabel="Counterfactual pair rho",ylabel="R_req",title="Actor (7,8) score; fixed node hazards")
    fig.colorbar(im,ax=ax);fig.savefig(out/"rreq_rho_score_heatmap.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([str(x) for x in sensitivity.R_req],sensitivity.mean_abs_actor_score_slope,marker="o",label="Actor score")
    ax.plot([str(x) for x in sensitivity.R_req],sensitivity.mean_abs_oracle_reward_slope,marker="o",label="Conditional oracle reward")
    ax.set(xlabel="R_req",ylabel="Mean absolute rho slope");ax.legend()
    fig.savefig(out/"rho_sensitivity_vs_requirement.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    oracle_grid=frame.groupby(["R_req","rho"]).oracle_reward.mean().unstack("rho")
    im=ax.imshow(oracle_grid.to_numpy(),aspect="auto",origin="lower",cmap="magma")
    ax.set_yticks(range(4),[str(x) for x in REQS]);ax.set_xticks([0,10,20,30],[f"{rho_grid[x]:.3f}" for x in (0,10,20,30)])
    ax.set(xlabel="Counterfactual pair rho",ylabel="R_req",title="Conditional oracle reward; fixed node hazards")
    fig.colorbar(im,ax=ax);fig.savefig(out/"oracle_rreq_rho_reward_heatmap.png");plt.close(fig)
    return sensitivity


def analyze(run:Path,output:Path,smoke:bool=False):
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((run/"run_metadata.json").read_text())
    assert (meta["train_episodes"],meta["eval_episodes"]) == ((2,1) if smoke else (300,20))
    formal=json.loads((FORMAL/"metadata.json").read_text())
    for name,key in (("server_info.xlsx","server_info_sha256"),("task_parameters.xlsx","task_parameters_sha256")):
        assert sha256_file(ROOT/"data"/name)==formal[key]
    pairs,rho=build_pair_correlations();pairs=list(pairs);rho=np.asarray(rho,dtype=float)
    actor,critic=load_actor(run,rho)
    servers=pd.read_excel(ROOT/"data/server_info.xlsx").set_index("Server_ID").sort_index()
    tasks=pd.read_excel(ROOT/"data/task_parameters.xlsx").set_index("Task_ID").sort_index()
    evaluation=pd.read_csv(run/"evaluation_task_assignments.csv").rename(columns={"episode":"Episode","task_id":"Task_ID"})
    eval_states=pd.read_csv(run/"evaluation_actor_outputs.csv")
    assert len(eval_states)==len(evaluation)==meta["eval_episodes"]*meta["tasks_per_episode"]
    with torch.no_grad():
        sample=torch.as_tensor(np.stack(eval_states.state.iloc[:20].map(vector)),dtype=torch.float32)
        saved=np.stack(eval_states.raw_logits.iloc[:20].map(vector))
        assert np.allclose(actor(sample).numpy(),saved,rtol=0,atol=1e-6)
    tier_ids=task_ids_by_tier(tasks,(13,13,12,12))
    selected=eval_states[eval_states.Task_ID.isin(tier_ids)].copy()
    assert len(selected)==meta["eval_episodes"]*len(tier_ids)
    trial=pd.read_csv(FORMAL/"seed_plan.csv").set_index("Trial_ID").loc[int(meta["trial_id"])]
    eval_alignment,contexts=oracle_for_traces(evaluation,selected,tasks,servers,pairs,
        int(trial.Eval_Spatial_Seed),tier_ids,"evaluation")
    assert len(eval_alignment)==len(selected)
    # Attach the unchanged state to the representative surface contexts.
    for c in contexts:
        row=selected[(selected.Episode==c["Episode"])&(selected.Task_ID==c["Task_ID"])].iloc[0]
        c["state"]=vector(row.state)
    eval_alignment.to_csv(output/"actor_oracle_rank_alignment.csv",index=False)
    eval_summary=summary_group(eval_alignment,["spearman_logit_reward","kendall_logit_reward",
        "spearman_probability_negative_regret","top1_hit","top3_hit","top5_hit",
        "optimal_set_probability_mass","uniform_optimal_set_mass",
        "optimal_mass_lift_over_uniform","sampled_regret","greedy_regret"])
    eval_summary.to_csv(output/"actor_oracle_alignment_by_requirement.csv",index=False)
    eval_alignment[["Episode","Task_ID","R_req","optimal_set_size","optimal_set_probability_mass"]].to_csv(
        output/"optimal_set_probability_mass.csv",index=False)
    # Full per-pair probabilities/regrets are retained for the matched evaluation subset.
    action_rows=[]
    rates_by_ep=episode_rates(int(trial.Eval_Spatial_Seed),meta["eval_episodes"],servers)
    for ep,frame in evaluation.groupby("Episode"):
        trace=EpisodeTrace(frame.rename(columns={"Episode":"episode","Task_ID":"task_id"}),tasks,servers)
        for tid in tier_ids:
            row=selected[(selected.Episode==ep)&(selected.Task_ID==tid)].iloc[0]
            req=float(row.R_req);rates=rates_by_ep[int(ep)]
            rewards=[simulator_reward(tid,float(tasks.loc[tid,"Computation_Demand"]),req,pair,
                trace.candidate_delay(tid,pair),rates,servers)["reward_total"] for pair in pairs]
            logits=vector(row.raw_logits);probs=vector(row.probabilities);best=max(rewards)
            for index,pair in enumerate(pairs):
                action_rows.append({"Episode":ep,"Task_ID":tid,"R_req":req,"action_index":index,
                    "pair":str(pair),"oracle_reward":rewards[index],"oracle_regret":best-rewards[index],
                    "actor_logit":logits[index],"actor_probability":probs[index],
                    "oracle_optimal":best-rewards[index]<=EXACT_TOL})
    actions=pd.DataFrame(action_rows);assert len(actions)==len(selected)*28
    actions.to_csv(output/"evaluation_action_vectors.csv",index=False)
    # Training credit/alignment: four fixed task IDs per tier in all 300 episodes.
    rollout_path=run/"training_rollout.csv.gz"
    if smoke and not rollout_path.exists(): rollout_path=run/"training_rollout.csv"
    train_states=pd.read_csv(rollout_path)
    training=pd.read_csv(run/"training_task_assignments.csv").rename(columns={"episode":"Episode","task_id":"Task_ID"})
    assert len(train_states)==len(training)==meta["train_episodes"]*meta["tasks_per_episode"]
    train_ids=task_ids_by_tier(tasks,4)
    chosen=train_states[train_states.Task_ID.isin(train_ids)].copy()
    train_alignment,_=oracle_for_traces(training,chosen,tasks,servers,pairs,
        int(trial.Train_Spatial_Seed),train_ids,"training")
    train_alignment=train_alignment.merge(chosen[["Episode","Task_ID","raw_advantage","normalized_advantage",
        "return","value","next_value","old_log_probability","entropy","reward","delta_t"]],
        on=["Episode","Task_ID"],validate="one_to_one")
    assert len(train_alignment)==meta["train_episodes"]*len(train_ids)
    train_alignment.to_csv(output/"training_advantage_oracle_regret.csv",index=False)
    advantage_rows=[];sign_rows=[]
    for req,g in train_alignment.groupby("R_req"):
        for kind in ("raw_advantage","normalized_advantage"):
            advantage_rows.append({"R_req":req,"advantage_type":kind,"N":len(g),
                "advantage_mean":float(g[kind].mean()),"advantage_std":float(g[kind].std()),
                "positive_fraction":float((g[kind]>0).mean()),
                "oracle_regret_mean":float(g.sampled_regret.mean()),
                "optimal_set_hit_rate":float((g.sampled_regret<=EXACT_TOL).mean()),
                "pearson_with_negative_regret":corr(g[kind],-g.sampled_regret,"pearson"),
                "spearman_with_negative_regret":corr(g[kind],-g.sampled_regret)})
            for label,part in (("positive",g[g[kind]>0]),("negative",g[g[kind]<0])):
                sign_rows.append({"R_req":req,"advantage_type":kind,"sign":label,"N":len(part),
                    "mean_oracle_regret":float(part.sampled_regret.mean()) if len(part) else np.nan,
                    "median_oracle_regret":float(part.sampled_regret.median()) if len(part) else np.nan,
                    "optimal_set_hit_rate":float((part.sampled_regret<=EXACT_TOL).mean()) if len(part) else np.nan})
    advantage=pd.DataFrame(advantage_rows);advantage.to_csv(output/"advantage_alignment_by_requirement.csv",index=False)
    signs=pd.DataFrame(sign_rows);signs.to_csv(output/"advantage_sign_action_quality.csv",index=False)
    # Full 60,000-decision pair visitation and exploration trajectory.
    probs=np.stack(train_states.probabilities.map(vector))
    sampled=train_states.sampled_action.to_numpy(dtype=int)
    greedy=train_states.greedy_action.to_numpy(dtype=int)
    visitation=[]
    regret_lookup=train_alignment.groupby("sampled_action").sampled_regret.mean()
    for action,pair in enumerate(pairs):
        selected_rows=train_states[sampled==action]
        visitation.append({"action_index":action,"pair":str(pair),"sampled_count":len(selected_rows),
            "greedy_count":int(np.count_nonzero(greedy==action)),
            "mean_probability":float(probs[:,action].mean()),
            "mean_probability_when_selected":float(probs[sampled==action,action].mean()) if len(selected_rows) else np.nan,
            "mean_normalized_advantage_when_selected":float(selected_rows.normalized_advantage.mean()),
            "positive_advantage_fraction_when_selected":float((selected_rows.normalized_advantage>0).mean()),
            "mean_oracle_regret_replayed_subset":float(regret_lookup.get(action,np.nan))})
    visitation=pd.DataFrame(visitation);visitation.to_csv(output/"pair_visitation_exploration.csv",index=False)
    trajectory=[]
    for ep,g in train_states.groupby("Episode"):
        indices=g.index.to_numpy();chosen78=g[g.sampled_action==27]
        trajectory.append({"Episode":int(ep),"sampled_78_frequency":float((g.sampled_action==27).mean()),
            "greedy_78_frequency":float((g.greedy_action==27).mean()),
            "mean_78_probability":float(probs[indices,27].mean()),
            "mean_78_advantage_when_selected":float(chosen78.normalized_advantage.mean()) if len(chosen78) else np.nan,
            "positive_78_advantage_fraction":float((chosen78.normalized_advantage>0).mean()) if len(chosen78) else np.nan})
    trajectory=pd.DataFrame(trajectory);trajectory.to_csv(output/"pair_78_training_trajectory.csv",index=False)
    untrained=untrained_bias(selected,rho,output)
    surface_contexts=[]
    for req in REQS:
        tier=[c for c in contexts if np.isclose(tasks.loc[c["Task_ID"],"Reliability_Requirement"],req)]
        if not tier:
            continue
        surface_contexts.extend(tier[i] for i in np.linspace(0,len(tier)-1,min(5,len(tier))).round().astype(int))
    sensitivity=rho_surface(actor,surface_contexts,servers,rho,output)
    # Compact figures.
    fig,ax=plt.subplots(figsize=(6,4))
    ax.errorbar([str(x) for x in eval_summary.R_req],eval_summary.spearman_logit_reward_mean,
        yerr=[eval_summary.spearman_logit_reward_mean-eval_summary.spearman_logit_reward_ci_low,
              eval_summary.spearman_logit_reward_ci_high-eval_summary.spearman_logit_reward_mean],marker="o")
    ax.axhline(0,color="gray");ax.set(xlabel="R_req",ylabel="Spearman(actor logit, oracle reward)")
    fig.savefig(output/"actor_oracle_rank_alignment.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([str(x) for x in eval_summary.R_req],eval_summary.optimal_set_probability_mass_mean,marker="o")
    ax.set_ylim(0,1);ax.set(xlabel="R_req",ylabel="Mean actor probability mass on Oracle optimal set")
    fig.savefig(output/"optimal_set_probability_mass_by_requirement.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    edges=[-1e-9,1e-10,.01,1,5,10,20,40,80,np.inf]
    labels=["exact", "(0,.01]", "(.01,1]", "(1,5]", "(5,10]",
            "(10,20]", "(20,40]", "(40,80]", ">80"]
    for req,g in actions.groupby("R_req"):
        bins=pd.cut(g.oracle_regret,bins=edges,labels=labels)
        means=g.groupby(bins,observed=False).actor_probability.mean().reindex(labels)
        ax.plot(range(len(labels)),means.to_numpy(),marker="o",label=str(req))
    ax.set_xticks(range(len(labels)),labels,rotation=35,ha="right")
    ax.set(xlabel="Oracle regret bin",ylabel="Mean actor probability")
    ax.legend();fig.tight_layout()
    fig.savefig(output/"probability_vs_regret.png");plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True)
    for ax,column in zip(axes,("sampled_78_frequency","mean_78_probability","mean_78_advantage_when_selected")):
        ax.plot(trajectory.Episode,trajectory[column]);ax.set_ylabel(column)
    axes[-1].set_xlabel("Training episode")
    fig.savefig(output/"pair_78_training_trajectory.png");plt.close(fig)
    result={"trial_id":int(meta["trial_id"]),"evaluation_aligned_states":len(eval_alignment),
        "training_replayed_states":len(train_alignment),"training_decisions":len(train_states),
        "untrained_actor_count":100,"checkpoint_reloaded_and_logits_verified":True,
        "conditional_oracle_rho_slope_zero":bool(np.allclose(sensitivity.mean_abs_oracle_reward_slope,0)),
        "evaluation_probabilities_finite":bool(np.isfinite(probs).all())}
    (output/"alignment_status.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run-dir",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();print(json.dumps(analyze(args.run_dir,args.output_dir,args.smoke),indent=2))
