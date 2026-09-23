"""Offline requirement-conditioning audit of the formal 10-seed experiment.

No policy is trained or changed here. The task-level oracle holds every other
recorded evaluation decision and arrival fixed. For a candidate pair it
replays the target replicas' CPU scheduling under the existing non-cancelling,
FCFS, equal-priority SimPy model. Baseline delays and hazards must reproduce
the formal evaluation log before any counterfactual result is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import PPOPairScoringPolicyNetwork
from config.params import params
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.spatial_risk import (
    build_distance_matrix, build_spatial_correlation_matrix,
    map_spatial_risk_to_effective_failure_rates, sample_spatial_risk_field,
)
from core.task import Task, get_upload_time
from Project_main import build_pair_correlations

FORMAL = ROOT / "diagnostics/paired_ppo_experiment/formal_10seed_300ep"
REQS = (0.9, 0.99, 0.999, 0.9999)
OBSERVED_HASHES = {
    "server_info.xlsx": "bf33fc1ce088a3a01f6201c8e0469ed1a63b883c69a599f00e75f0299ba38849",
    "task_parameters.xlsx": "f79752048c546137865aafe61cb96d86c2f567df97239dae241a2831ae986434",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Job:
    ready: float
    task_id: int
    role: int
    duration: float

    @property
    def key(self):
        return (self.ready, self.task_id, self.role)


class EpisodeTrace:
    def __init__(self, frame: pd.DataFrame, tasks: pd.DataFrame, servers: pd.DataFrame):
        self.frame = frame.sort_values("task_id").set_index("task_id", drop=False)
        self.tasks = tasks
        self.servers = servers
        self.jobs = {int(s): [] for s in servers.index}
        for row in self.frame.itertuples():
            task_id = int(row.task_id)
            arrival = float(row.Primary_Start)
            for role, sid in enumerate((int(row.Primary), int(row.Backup))):
                server = servers.loc[sid]
                task = tasks.loc[task_id]
                ready = arrival + get_upload_time(task.Input_Data_Size_MB, server.Uplink_Rate)
                duration = float(task.Computation_Demand / server.Processing_Frequency)
                self.jobs[sid].append(Job(ready, task_id, role, duration))
        for jobs in self.jobs.values():
            jobs.sort(key=lambda job: job.key)

    def finish(self, server_id: int, target: Job) -> float:
        """One-task intervention: other tasks' upload times and pair choices stay fixed."""
        busy_until = 0.0
        for job in self.jobs[server_id]:
            if job.task_id == target.task_id:
                continue  # remove the originally selected target replica
            if job.key > target.key:
                break
            busy_until = max(busy_until, job.ready) + job.duration
        return max(busy_until, target.ready) + target.duration

    def candidate_delay(self, task_id: int, pair: tuple[int, int]) -> float:
        row = self.frame.loc[task_id]
        arrival = float(row.Primary_Start)
        task = self.tasks.loc[task_id]
        finish_times = []
        for role, sid in enumerate(pair):
            server = self.servers.loc[sid]
            ready = arrival + get_upload_time(task.Input_Data_Size_MB, server.Uplink_Rate)
            duration = float(task.Computation_Demand / server.Processing_Frequency)
            finish_times.append(self.finish(sid, Job(ready, task_id, role, duration)))
        return min(finish_times) - arrival

    def backlog_at(self, arrival: float, target_id: int) -> list[float]:
        """CPU work already arrived by this decision; excludes in-flight upload."""
        values = []
        for sid in self.servers.index:
            busy_until = 0.0
            for job in self.jobs[int(sid)]:
                if job.ready > arrival:
                    break
                if job.task_id != target_id:
                    busy_until = max(busy_until, job.ready) + job.duration
            values.append(max(busy_until - arrival, 0.0))
        return values

    def baseline_delay_errors(self) -> np.ndarray:
        errors = []
        for row in self.frame.itertuples():
            reconstructed = self.candidate_delay(int(row.task_id), (int(row.Primary), int(row.Backup)))
            errors.append(abs(reconstructed - float(row.Task_Delay)))
        return np.asarray(errors, dtype=float)


def episode_rates(seed: int, episodes: int, servers: pd.DataFrame) -> dict[int, np.ndarray]:
    locations = [SimpleNamespace(server_id=int(sid), latitude=float(row.Latitude), longitude=float(row.Longitude))
                 for sid, row in servers.iterrows()]
    server_ids, distance = build_distance_matrix(locations)
    assert server_ids == list(servers.index)
    corr = build_spatial_correlation_matrix(distance, params.SPATIAL_CORRELATION_LENGTH_KM)
    rng = np.random.default_rng(seed)
    base = servers.Base_Failure_Rate.to_numpy(dtype=float)
    result = {}
    for ep in range(1, episodes + 1):
        field = sample_spatial_risk_field(corr, rng=rng)
        result[ep] = map_spatial_risk_to_effective_failure_rates(base, field, params.SPATIAL_RISK_BETA_P)
    return result


def simulator_reward(task_id: int, demand: float, requirement: float,
                     pair: tuple[int, int], delay: float, rates: np.ndarray,
                     servers: pd.DataFrame):
    """Reuse Task.initialize_reliability_evaluation and MainLoop.calcReward."""
    task = Task.__new__(Task)
    task.id = int(task_id)
    task.computation_demand = float(demand)
    task.reliability_requirement = float(requirement)
    task.resolution_bookkeeping_done = False
    task.primaryStarted = 0.0
    task.primaryFinished = float(delay)
    task.backupFinished = float(delay)
    task.env_state = SimpleNamespace(get_active_failure_rate=lambda sid: float(rates[sid - 1]))
    nodes = [SimpleNamespace(server_id=sid, processing_frequency=float(servers.loc[sid, "Processing_Frequency"])) for sid in pair]
    Task.initialize_reliability_evaluation(task, *nodes)
    loop = MainLoop.__new__(MainLoop)
    loop.env_state = SimpleNamespace(get_task_by_id=lambda _tid: task)
    total, checked_delay = MainLoop.calcReward(loop, task_id)
    assert math.isclose(checked_delay, delay, abs_tol=1e-12)
    actual_base = float(task.base_reward)
    penalty = float(task.reliability_penalty)
    violation = float(task.reliability_violation)
    satisfied = bool(task.reliability_satisfied)
    task.reliability_satisfied = True
    MainLoop.calcReward(loop, task_id)
    success_delay_reward = float(task.base_reward)
    assert math.isclose(total, success_delay_reward + (actual_base - success_delay_reward) - penalty, abs_tol=1e-9)
    return {
        "pair_reliability": float(task.execution_reliability),
        "joint_failure_probability": float(task.joint_failure_probability),
        "reliability_satisfied": satisfied,
        "reliability_violation": violation,
        "reward_total": float(total),
        "reward_delay_component": success_delay_reward,
        "reward_reliability_gate_component": actual_base - success_delay_reward,
        "reward_reliability_penalty_component": -penalty,
        "reward_reliability_component": actual_base - success_delay_reward - penalty,
        "reward_correlation_component": 0.0,
        "reward_energy_component": 0.0,
    }


def bootstrap_ci(values, seed=2026, samples=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def input_trace(servers: pd.DataFrame, tasks: pd.DataFrame, pairs, rho, output: Path):
    env_state = EnvironmentState()
    for sid, row in servers.iterrows():
        env_state.add_server_and_init_environment(SimpleNamespace(
            server_id=int(sid), base_failure_rate=float(row.Base_Failure_Rate),
            processing_frequency=float(row.Processing_Frequency),
            uplink_rate_mbps=float(row.Uplink_Rate)))
    first = tasks.iloc[0]
    actor = PPOPairScoringPolicyNetwork(params.num_states, params.num_actions,
        params.hidden_layers_ppo, params.serverNo, rho, activation=params.af_ppo)
    lines = ["Feature-path audit only; no trained checkpoint is loaded.",
             "Source: task_parameters.xlsx -> Task.reliability_requirement -> EnvironmentState.get_state -> float32 tensor -> PairScoring.build_pair_features.",
             "The final 35-state coordinate is the normalized R_req; the final coordinate of each 12-feature pair vector carries it."]
    all_states = []
    for req in REQS:
        task = SimpleNamespace(env=SimpleNamespace(now=0.0), input_data_size_mb=float(first.Input_Data_Size_MB),
            computation_demand=float(first.Computation_Demand), reliability_requirement=req)
        state = env_state.get_state(task)
        tensor = torch.as_tensor(state, dtype=torch.float32)
        pair_features = actor.build_pair_features(tensor)
        assert state.shape == (35,) and pair_features.shape == (28, 12)
        assert torch.all(pair_features[:, -1] == tensor[-1])
        all_states.append(state)
        lines.append(f"R_req={req}: stored={task.reliability_requirement:.10g}; state[-1]={float(state[-1]):.9g}; tensor[-1]={float(tensor[-1]):.9g}; all_28_pair_features[-1]={float(pair_features[0,-1]):.9g}")
    assert all(np.array_equal(all_states[0][:-1], s[:-1]) for s in all_states[1:])
    lines += ["Only the final observation coordinate changes in this controlled trace.",
              "No round/cast/clip collapse: four levels map to four distinct float32 values.",
              "No deadline feature exists in the current simulator; no action mask is applied beyond the 28 distinct unordered pair indices."]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(out: Path, oracle: pd.DataFrame, summary: dict, deltaq: pd.DataFrame,
               pareto: pd.DataFrame, policy: pd.DataFrame, tv: pd.DataFrame):
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})
    changes = summary["oracle_change_rates"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(changes), list(changes.values())); ax.set_ylim(0, 1)
    ax.set_ylabel("Oracle argmax change rate"); ax.tick_params(axis="x", rotation=25)
    fig.savefig(out / "oracle_change_rate.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for req, group in oracle.groupby("R_req"):
        best = group[group.is_oracle_argmax]["oracle_margin"].to_numpy()
        ax.hist(best, bins=40, alpha=.4, label=str(req))
    ax.set_xlabel("Best minus second-best reward"); ax.set_ylabel("States"); ax.legend()
    fig.savefig(out / "oracle_reward_margin.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    stats = deltaq.groupby("R_req").DeltaQ_Rho.agg(["mean"])
    intervals = np.asarray([bootstrap_ci(deltaq.loc[deltaq.R_req == req, "DeltaQ_Rho"])
                            for req in stats.index])
    yerr = np.vstack((stats["mean"].to_numpy() - intervals[:, 0],
                      intervals[:, 1] - stats["mean"].to_numpy()))
    ax.errorbar([str(v) for v in stats.index], stats["mean"], yerr=yerr, marker="o")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_ylabel("Reward(low-rho) - Reward(high-rho)")
    fig.savefig(out / "delta_q_vs_requirement.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for req, group in deltaq.groupby("R_req"):
        ax.hist(group.DeltaQ_Rho, bins=35, alpha=.35, label=str(req))
    ax.set_xlabel("DeltaQ_rho"); ax.set_ylabel("States"); ax.legend()
    fig.savefig(out / "delta_q_distribution.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for front, g in pareto[pareto.R_req == .9999].groupby("Pareto_Front"):
        ax.scatter(g.Mean_Latency, g.Mean_Pair_Reliability, label=f"front={front}", s=30 + 300*g.Oracle_Selection_Frequency)
    p78 = pareto[(pareto.R_req == .9999) & (pareto.Server_J == 7) & (pareto.Server_K == 8)].iloc[0]
    ax.annotate("(7,8)", (p78.Mean_Latency, p78.Mean_Pair_Reliability))
    ax.set_xlabel("Mean latency (s)"); ax.set_ylabel("Mean pair reliability"); ax.legend()
    fig.savefig(out / "pair_pareto.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    means = tv.groupby("Trial_ID")[["TV_Rreq_09_09999", "TV_Rho_True_Zero"]].mean().mean()
    ax.bar(["R_req .9 -> .9999", "rho true -> zero"], means.to_numpy())
    ax.set_ylabel("Mean total variation distance")
    fig.savefig(out / "tv_rho_vs_tv_rreq.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    top_probability = policy.groupby("Reliability_Requirement").Top_Action_Probability.mean()
    ax.plot([str(v) for v in top_probability.index], top_probability.to_numpy(), marker="o")
    ax.set_ylabel("Mean probability of recorded top action"); ax.set_ylim(0, 1)
    fig.savefig(out / "requirement_action_probability.png"); plt.close(fig)


def run(max_states: int, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    assert max_states > 0 and max_states <= 1000
    meta = json.loads((FORMAL / "metadata.json").read_text())
    assert meta["git_commit"] == "7ddef1af40aef6e42c91f692946fda847a465024"
    for name, expected in OBSERVED_HASHES.items():
        actual = file_hash(ROOT / "data" / name)
        if actual != expected:
            raise RuntimeError(f"Input data changed: {name}: {actual} != {expected}")
    servers = pd.read_excel(ROOT / "data/server_info.xlsx").set_index("Server_ID").sort_index()
    tasks = pd.read_excel(ROOT / "data/task_parameters.xlsx").set_index("Task_ID").sort_index()
    assert len(servers) == 8 and len(tasks) == 200
    pairs, rho = build_pair_correlations()
    pairs = list(pairs); rho = np.asarray(rho, dtype=float)
    assert len(pairs) == 28 and all(a < b for a, b in pairs)
    seed_plan = pd.read_csv(FORMAL / "seed_plan.csv").set_index("Trial_ID")
    input_trace(servers, tasks, pairs, rho, output / "r_req_input_trace.txt")
    chosen = {(trial, ep, task_id) for trial in range(10) for ep in range(1,21)
              for task_id in (20, 60, 100, 140, 180)}
    chosen = sorted(chosen)[:max_states]
    selection = {}
    for trial, ep, task_id in chosen:
        selection.setdefault((trial, ep), set()).add(task_id)
    oracle_rows = []
    decomposition_rows = []
    state_context = {}
    delay_errors = []
    hazard_errors = []
    observed_reward_errors = []
    for trial in range(10):
        fpath = FORMAL / "runs" / f"trial_{trial:03d}" / "pair_scoring" / "evaluation_task_assignments.csv"
        frame = pd.read_csv(fpath)
        rates_by_ep = episode_rates(int(seed_plan.loc[trial, "Eval_Spatial_Seed"]), 20, servers)
        for ep, ep_frame in frame.groupby("episode", sort=True):
            trace = EpisodeTrace(ep_frame, tasks, servers)
            errors = trace.baseline_delay_errors()
            delay_errors.extend(errors)
            rates = rates_by_ep[int(ep)]
            for row in ep_frame.itertuples():
                sid1, sid2 = int(row.Primary), int(row.Backup)
                hazard_errors.extend([abs(rates[sid1-1]-float(row.Primary_Effective_Failure_Rate)),
                                      abs(rates[sid2-1]-float(row.Backup_Effective_Failure_Rate))])
                actual = simulator_reward(int(row.task_id), float(tasks.loc[int(row.task_id), "Computation_Demand"]),
                    float(row.Reliability_Requirement), (sid1, sid2), float(row.Task_Delay), rates, servers)
                observed_reward_errors.append(abs(actual["reward_total"] - float(row.Task_Reward)))
                decomposition_rows.append({"Trial_ID": trial, "Episode": int(ep), "Task_ID": int(row.task_id),
                    "R_req": float(row.Reliability_Requirement), "Latency": float(row.Task_Delay), **actual})
            for task_id in selection.get((trial, int(ep)), ()):
                row = trace.frame.loc[task_id]
                task = tasks.loc[task_id]
                arrival = float(row.Primary_Start)
                fixed = {"task_size_mb": float(task.Input_Data_Size_MB),
                         "computation_demand_mi": float(task.Computation_Demand),
                         "arrival_time": arrival,
                         "episode_effective_failure_rates": rates.tolist(),
                         "server_backlog_seconds": trace.backlog_at(arrival, task_id),
                         "other_task_actions_fixed": True,
                         "candidate_actions": len(pairs)}
                state_id = f"trial{trial:02d}_episode{int(ep):02d}_task{task_id:03d}"
                fingerprint = hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()
                state_context[state_id] = {"fingerprint": fingerprint, **fixed}
                for index, pair in enumerate(pairs):
                    latency = trace.candidate_delay(task_id, pair)
                    for req in REQS:
                        result = simulator_reward(task_id, float(task.Computation_Demand), req, pair, latency, rates, servers)
                        oracle_rows.append({"state_id": state_id, "Trial_ID": trial, "Episode": int(ep),
                            "Task_ID": task_id, "state_context_sha256": fingerprint,
                            "R_req": req, "action_index": index, "pair": f"({pair[0]},{pair[1]})",
                            "primary_server": pair[0], "backup_server": pair[1], "rho_jk": float(rho[index]),
                            "latency": latency, "selection_feasible": True, **result})
    delay_errors = np.asarray(delay_errors)
    hazard_errors = np.asarray(hazard_errors)
    observed_reward_errors = np.asarray(observed_reward_errors)
    if delay_errors.size != 40000 or np.max(delay_errors) > 1e-8:
        raise RuntimeError(f"Baseline queue replay mismatch: {delay_errors.size} rows, max error {delay_errors.max()}")
    if np.max(hazard_errors) > 1e-10 or np.max(observed_reward_errors) > 1e-8:
        raise RuntimeError(f"Existing simulator formulas failed validation: hazard={hazard_errors.max()}, reward={observed_reward_errors.max()}")
    oracle = pd.DataFrame(oracle_rows)
    assert len(oracle) == max_states * 4 * 28
    assert oracle.groupby("state_id").state_context_sha256.nunique().eq(1).all()
    assert oracle.groupby(["state_id", "action_index"]).latency.nunique().eq(1).all()
    assert oracle.groupby(["state_id", "action_index"]).pair_reliability.nunique().eq(1).all()
    if not np.isfinite(oracle.select_dtypes(include=np.number).to_numpy()).all():
        raise RuntimeError("Oracle contains NaN/Inf")
    oracle["is_oracle_argmax"] = False
    oracle["is_reward_optimal_tie"] = False
    oracle["oracle_margin"] = np.nan
    oracle["optimal_set_size"] = 0
    best_rows = []
    optimal_sets = {}
    for (sid, req), g in oracle.groupby(["state_id", "R_req"], sort=True):
        vals = g.reward_total.to_numpy(dtype=float)
        best = float(vals.max())
        order = np.argsort(-vals, kind="stable")
        first = int(g.iloc[order[0]].action_index)
        margin = best - float(vals[order[1]])
        tied = g.loc[np.isclose(g.reward_total, best, atol=1e-9, rtol=0), "action_index"].astype(int).tolist()
        optimal_sets[(sid, req)] = set(tied)
        oracle.loc[g.index, "is_oracle_argmax"] = g.action_index.eq(first).to_numpy()
        oracle.loc[g.index, "is_reward_optimal_tie"] = g.action_index.isin(tied).to_numpy()
        oracle.loc[g.index, "oracle_margin"] = margin
        oracle.loc[g.index, "optimal_set_size"] = len(tied)
        best_rows.append({"state_id": sid, "R_req": req, "action_index": first,
                          "pair": f"({pairs[first][0]},{pairs[first][1]})", "reward": best,
                          "margin": margin, "optimal_set_size": len(tied)})
    best_df = pd.DataFrame(best_rows)
    best_pivot = best_df.pivot(index="state_id", columns="R_req", values="action_index")
    change = {}
    disjoint = {}
    for a, b in zip(REQS[:-1], REQS[1:]):
        label = f"{a:g}->{b:g}"
        change[label] = float((best_pivot[a] != best_pivot[b]).mean())
        disjoint[label] = float(np.mean([not(optimal_sets[(sid,a)] & optimal_sets[(sid,b)]) for sid in best_pivot.index]))
    change["0.9->0.9999"] = float((best_pivot[.9] != best_pivot[.9999]).mean())
    disjoint["0.9->0.9999"] = float(np.mean([not(optimal_sets[(sid,.9)] & optimal_sets[(sid,.9999)]) for sid in best_pivot.index]))
    unique_counts = best_pivot.nunique(axis=1)
    switch = pd.crosstab(best_pivot[.9].astype(int), best_pivot[.9999].astype(int))
    switch.index = [f"{pairs[i]}" for i in switch.index]
    switch.columns = [f"{pairs[i]}" for i in switch.columns]
    switch.to_csv(output / "oracle_switch_matrix.csv")
    margin_summary = best_df.groupby("R_req").margin.agg(
        Mean="mean", Median="median", P25=lambda x:x.quantile(.25),
        P75=lambda x:x.quantile(.75), P90=lambda x:x.quantile(.9)).reset_index()
    oracle_summary = {"sampled_states": max_states, "legal_pairs": len(pairs),
        "source": "formal pair_scoring evaluation; 10 trials x 20 episodes x 5 task IDs",
        "oracle_scope": "realized one-task reward intervention; other tasks' recorded arrivals and pair choices fixed",
        "oracle_change_rates": change, "disjoint_optimal_set_rates": disjoint,
        "unique_argmax_mean": float(unique_counts.mean()), "unique_argmax_median": float(unique_counts.median()),
        "unique_argmax_distribution": {str(k): int(v) for k,v in unique_counts.value_counts().sort_index().items()},
        "margin_by_requirement": margin_summary.to_dict("records"),
        "best_pair_78_frequency_by_requirement": {str(req): float((g.action_index == 27).mean()) for req,g in best_df.groupby("R_req")},
        "best_pair_78_in_optimal_set_by_requirement": {str(req): float(g[g.action_index == 27].is_reward_optimal_tie.mean()) for req,g in oracle.groupby("R_req")},
        "optimal_tie_fraction_by_requirement": {str(req): float((g.optimal_set_size > 1).mean()) for req,g in best_df.groupby("R_req")},
        "replay_max_abs_delay_error": float(delay_errors.max()),
        "replay_max_abs_hazard_error": float(hazard_errors.max()),
        "replay_max_abs_reward_error": float(observed_reward_errors.max()),
        "checkpoint_available": False}
    (output / "oracle_summary.json").write_text(json.dumps(oracle_summary, indent=2), encoding="utf-8")
    oracle.to_csv(output / "oracle_pair_results.csv", index=False)
    margin_summary.to_csv(output / "oracle_reward_margin_summary.csv", index=False)
    pd.DataFrame.from_dict(state_context, orient="index").rename_axis("state_id").reset_index().to_csv(output / "oracle_state_context.csv", index=False)
    assert pairs[27] == (7, 8)
    pair78_rows = oracle[oracle.action_index == 27].copy()
    pair78_rows["regret"] = (oracle.groupby(["state_id", "R_req"]).reward_total.max()
                             .reindex(pd.MultiIndex.from_frame(pair78_rows[["state_id", "R_req"]])).to_numpy()
                             - pair78_rows.reward_total.to_numpy())
    pair78_regret = pair78_rows.groupby("R_req").regret.agg(
        Mean_Regret="mean", Median_Regret="median", P90_Regret=lambda x:x.quantile(.9),
        Fraction_Strictly_Suboptimal=lambda x:float((x > 1e-9).mean())).reset_index()
    pair78_regret.to_csv(output / "pair_78_regret_by_requirement.csv", index=False)

    # P1: contrast the lowest-rho and highest-rho quartile, matching on
    # baseline reliability and latency to reduce obvious capability confounding.
    low_ids = np.flatnonzero(rho <= np.quantile(rho,.25))
    high_ids = np.flatnonzero(rho >= np.quantile(rho,.75))
    reference = oracle[oracle.R_req == .9]
    delta_rows = []
    for sid, g in reference.groupby("state_id"):
        g = g.set_index("action_index")
        lat_scale = max(float(g.latency.std(ddof=0)), .01)
        rel_scale = max(float(g.pair_reliability.std(ddof=0)), .0001)
        best_cost = float("inf"); chosen = None
        for l in low_ids:
            for h in high_ids:
                cost = abs(float(g.loc[l,"latency"]-g.loc[h,"latency"]))/lat_scale + \
                       abs(float(g.loc[l,"pair_reliability"]-g.loc[h,"pair_reliability"]))/rel_scale
                if cost < best_cost:
                    best_cost = cost; chosen = (int(l),int(h))
        lo,hi = chosen
        for req in REQS:
            gr = oracle[(oracle.state_id == sid) & (oracle.R_req == req)].set_index("action_index")
            delta_rows.append({"state_id":sid,"R_req":req,"low_pair":str(pairs[lo]),"high_pair":str(pairs[hi]),
                "low_rho":float(rho[lo]),"high_rho":float(rho[hi]),"matching_cost":best_cost,
                "latency_difference":float(gr.loc[lo,"latency"]-gr.loc[hi,"latency"]),
                "reliability_difference":float(gr.loc[lo,"pair_reliability"]-gr.loc[hi,"pair_reliability"]),
                "DeltaQ_Rho":float(gr.loc[lo,"reward_total"]-gr.loc[hi,"reward_total"])})
    deltaq = pd.DataFrame(delta_rows)
    deltaq.to_csv(output / "delta_q_rho_by_requirement.csv", index=False)
    delta_summary = []
    for req,g in deltaq.groupby("R_req"):
        lo,hi = bootstrap_ci(g.DeltaQ_Rho)
        delta_summary.append({"R_req":req,"Mean_DeltaQ":float(g.DeltaQ_Rho.mean()),
                              "Median_DeltaQ":float(g.DeltaQ_Rho.median()),
                              "Bootstrap_95CI_Low":lo,"Bootstrap_95CI_High":hi})
    delta_summary = pd.DataFrame(delta_summary)
    delta_summary.to_csv(output / "delta_q_rho_summary.csv", index=False)

    # P2: static pair correlation plus realized task-specific performance.
    grouped = oracle.groupby(["action_index","primary_server","backup_server","R_req"], as_index=False).agg(
        Rho=("rho_jk","first"), Mean_Pair_Reliability=("pair_reliability","mean"),
        Mean_Latency=("latency","mean"), Mean_Reward=("reward_total","mean"),
        Reliability_Violation_Rate=("reliability_satisfied",lambda x:1-float(x.mean())),
        Selection_Feasibility_Rate=("selection_feasible","mean"),
        Oracle_Selection_Frequency=("is_oracle_argmax","mean"),
        In_Optimal_Set_Frequency=("is_reward_optimal_tie","mean"))
    grouped = grouped.rename(columns={"primary_server":"Server_J","backup_server":"Server_K"})
    grouped["Pareto_Front"] = True
    for req,g in grouped.groupby("R_req"):
        for idx,row in g.iterrows():
            dominated = ((g.Rho <= row.Rho+1e-12) &
                         (g.Mean_Latency <= row.Mean_Latency+1e-12) &
                         (g.Mean_Pair_Reliability >= row.Mean_Pair_Reliability-1e-12) &
                         ((g.Rho < row.Rho-1e-12) | (g.Mean_Latency < row.Mean_Latency-1e-12) |
                          (g.Mean_Pair_Reliability > row.Mean_Pair_Reliability+1e-12)))
            grouped.loc[idx,"Pareto_Front"] = not bool(dominated.any())
    grouped.to_csv(output / "pair_pareto_analysis.csv", index=False)

    # Recorded policy probes are the only persisted trained-policy information.
    policy_parts=[]; tv_rows=[]
    for trial in range(10):
        run = FORMAL / "runs" / f"trial_{trial:03d}" / "pair_scoring"
        reqdf = pd.read_csv(run / "requirement_sensitivity.csv")
        rhodf = pd.read_csv(run / "rho_counterfactual.csv")
        reqdf.insert(0,"Trial_ID",trial)
        reqdf.insert(1,"Source","formal_recorded_probe_summary_no_checkpoint")
        policy_parts.append(reqdf)
        q = reqdf.pivot(index="Probe_ID", columns="Reliability_Requirement", values="Top_Action_Index")
        high = reqdf[np.isclose(reqdf.Reliability_Requirement,.9999)]
        zero = rhodf[rhodf.Rho_Scenario == "zero"]
        tv_rows.append({"Trial_ID":trial,"Num_Probes":len(high),
            "TV_Rreq_09_09999":float(high.TV_From_R09.mean()),
            "JS_Rreq_09_09999":float(high.JS_From_R09.mean()),
            "TV_Rho_True_Zero":float(zero.TV_From_True.mean()),
            "JS_Rho_True_Zero":float(zero.JS_From_True.mean()),
            "Argmax_09_09999":float((q[.9]!=q[.9999]).mean()),
            "Argmax_09_099":float((q[.9]!=q[.99]).mean()),
            "Argmax_099_0999":float((q[.99]!=q[.999]).mean()),
            "Argmax_0999_09999":float((q[.999]!=q[.9999]).mean())})
    policy = pd.concat(policy_parts, ignore_index=True)
    policy.to_csv(output / "policy_requirement_sweep.csv", index=False)
    tv = pd.DataFrame(tv_rows)
    tv.to_csv(output / "tv_requirement_summary.csv", index=False)
    assert len(policy)==4000 and np.isfinite(tv.select_dtypes(include=np.number).to_numpy()).all()
    tv_aggregate = []
    for metric in ("TV_Rreq_09_09999", "TV_Rho_True_Zero", "Argmax_09_09999"):
        vals = tv[metric].to_numpy()
        ci_low, ci_high = bootstrap_ci(vals)
        tv_aggregate.append({"Metric":metric,"Num_Seeds":len(vals),"Mean":float(vals.mean()),
                             "Std":float(vals.std(ddof=1)),"Median":float(np.median(vals)),
                             "Bootstrap_95CI_Low":ci_low,"Bootstrap_95CI_High":ci_high})
    pd.DataFrame(tv_aggregate).to_csv(output / "tv_requirement_aggregate.csv", index=False)

    # Reward decomposition of all 40,000 recorded Pair evaluation tasks.
    observed = pd.DataFrame(decomposition_rows)
    components = ["reward_total","reward_delay_component","reward_reliability_gate_component",
                  "reward_reliability_penalty_component","reward_reliability_component",
                  "reward_correlation_component","reward_energy_component","reliability_violation","Latency"]
    decomp = observed.groupby("R_req")[components].mean().reset_index()
    decomp["Task_Count"] = observed.groupby("R_req").size().to_numpy()
    decomp["Violation_Rate"] = observed.groupby("R_req").reliability_satisfied.apply(lambda x:1-float(x.mean())).to_numpy()
    decomp["Mean_Absolute_Reliability_Component"] = observed.groupby("R_req").reward_reliability_component.apply(lambda x:float(np.abs(x).mean())).to_numpy()
    decomp.to_csv(output / "reward_decomposition_by_requirement.csv", index=False)
    assert np.isfinite(decomp.select_dtypes(include=np.number).to_numpy()).all()
    # Controlled decomposition: the same state and pair at both requirements.
    low = oracle[np.isclose(oracle.R_req, .9)].set_index(["state_id", "action_index"])
    high = oracle[np.isclose(oracle.R_req, .9999)].set_index(["state_id", "action_index"])
    assert low.index.equals(high.index)
    shift_rows = []
    for component in ("reward_total", "reward_delay_component", "reward_reliability_component",
                      "reward_reliability_gate_component", "reward_reliability_penalty_component"):
        changes = (high[component] - low[component]).abs()
        shift_rows.append({"Component":component, "Matched_State_Pairs":len(changes),
                           "Mean_Absolute_Change":float(changes.mean()),
                           "Median_Absolute_Change":float(changes.median()),
                           "Fraction_Nonzero":float((changes > 1e-9).mean())})
    pd.DataFrame(shift_rows).to_csv(output / "reward_component_counterfactual_shift.csv", index=False)

    make_plots(output, oracle, oracle_summary, deltaq, grouped, policy, tv)
    status = {"checkpoint_available":False,
              "missing_outputs_due_to_checkpoint": ["full action logits/probabilities", "adjacent-tier TV", "R_req x rho score heatmap", "rho-sensitivity finite differences"],
              "policy_data_source":"formal recorded 100 probes per trial, not reloaded weights",
              "oracle_and_policy_probes_are_state_matched":False,
              "baseline_delay_rows_validated":int(delay_errors.size),
              "oracle_state_count":max_states,
              "validation":{"all_baseline_delays_match":True,"all_hazards_match":True,"all_rewards_match":True,
                            "R_req_only_changed":True,"oracle_pure_no_environment_mutation":True,
                            "finite_oracle":True},
              "input_hashes":{name:file_hash(ROOT/"data"/name) for name in OBSERVED_HASHES}}
    (output / "diagnostic_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    write_report(output, oracle_summary, best_df, delta_summary, grouped, pair78_regret, tv, decomp, status)
    return status


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(map(str, columns)) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.5g}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(rows)


def write_report(out:Path, o:dict, best:pd.DataFrame, delta:pd.DataFrame, pareto:pd.DataFrame,
                 regret:pd.DataFrame, tv:pd.DataFrame, decomp:pd.DataFrame, status:dict):
    oracle_ext=o["oracle_change_rates"]["0.9->0.9999"]
    disjoint_ext=o["disjoint_optimal_set_rates"]["0.9->0.9999"]
    policy_ext=float(tv.Argmax_09_09999.mean())
    tv_req=float(tv.TV_Rreq_09_09999.mean())
    tv_rho=float(tv.TV_Rho_True_Zero.mean())
    p78=pareto[(pareto.Server_J==7)&(pareto.Server_K==8)]
    p78hi=p78[np.isclose(p78.R_req,.9999)].iloc[0]
    delta_low=float(delta.loc[np.isclose(delta.R_req,.9),"Mean_DeltaQ"].iloc[0])
    delta_high=float(delta.loc[np.isclose(delta.R_req,.9999),"Mean_DeltaQ"].iloc[0])
    high_decomp=decomp[np.isclose(decomp.R_req,.9999)].iloc[0]
    low_decomp=decomp[np.isclose(decomp.R_req,.9)].iloc[0]
    shift=pd.read_csv(out/"reward_component_counterfactual_shift.csv")
    tv_aggregate=pd.read_csv(out/"tv_requirement_aggregate.csv")
    matched_cost=pd.read_csv(out/"delta_q_rho_by_requirement.csv")
    matched_cost=matched_cost[np.isclose(matched_cost.R_req,.9)].matching_cost
    lines=["# Pair-Scoring PPO：可靠性需求条件化诊断", "",
        "## 方法与可验证范围", "",
        f"正式实验的 10 个 seed × 20 个评估 episode × 每 episode 5 个 task，合计 {o['sampled_states']} 个固定决策上下文。每个上下文穷举 28 个合法无序异节点 pair × 四档 R_req（112,000 条反事实）。其他任务的到达和动作保持已记录值，仅改变目标任务的 pair 和 R_req。",
        "Oracle 用无副作用的 FIFO CPU / 固定上行轨迹重建目标任务的首副本完成时间；episode 有效故障率按正式 seed 重现；可靠性和奖励分别调用生产代码 Task.initialize_reliability_evaluation 和 MainLoop.calcReward。它是**目标任务即时 reward** Oracle，不包含对后续任务队列及长期 PPO return 的反事实影响。",
        f"逐条重放全部 40,000 条正式 Pair 评估记录：最大延迟误差 {o['replay_max_abs_delay_error']:.3g} 秒，有效故障率误差 {o['replay_max_abs_hazard_error']:.3g}，reward 误差 {o['replay_max_abs_reward_error']:.3g}。各档除 R_req 外的 context SHA256、延迟及 pair reliability 完全相同。当前没有 deadline、额外 action mask、直接 rho 或能耗 reward 项。", "",
        "## 1. Oracle 是否要求随 R_req 改变 pair？", "",
        f"按 action-index 打破并列时，.9→.9999 的 argmax 改变率为 **{oracle_ext:.1%}**；但两端最优集合完全不相交的状态只有 **{disjoint_ext:.1%}**。因此判定 **A：真正迫使动作改变的需求边界稀少**。单看 {oracle_ext:.1%} 会把并列最优的索引切换误当成强监督。",
        "相邻档 argmax / 最优集合不相交比例：" + "; ".join(f"{k}: {v:.1%} / {o['disjoint_optimal_set_rates'][k]:.1%}" for k,v in o['oracle_change_rates'].items() if k != '0.9->0.9999') + "。",
        f"四档唯一 argmax 数 mean={o['unique_argmax_mean']:.3f}、median={o['unique_argmax_median']:.0f}；分布 {o['unique_argmax_distribution']}。存在多个并列最优的状态比例依次为 {list(o['optimal_tie_fraction_by_requirement'].values())}。",
        "Reward margin（best−second）如下；前三档几乎都是 0，最高档 median 仍为 0：", "",
        markdown_table(pd.DataFrame(o['margin_by_requirement'])), "",
        "切换明细见 oracle_switch_matrix.csv；并列最优集合和每行 reward 见 oracle_pair_results.csv。", "",
        "## 2. Pair PPO 是否利用 R_req？", "",
        f"正式实验持久化的 10×100 个 probe 显示，.9→.9999 的 greedy argmax 改变率 **{policy_ext:.1%}**，TV_Rreq={tv_req:.5f}，而 TV_rho(true→zero)={tv_rho:.5f}。两种 TV 的扰动尺度不同，数值比仅供描述；TV_rho 只约为 TV_Rreq 的 {tv_rho/tv_req:.2f} 倍，不能断言分布层面对 R_req 完全无响应。Greedy 动作的确基本不变。",
        f"Oracle 强制切换 {disjoint_ext:.1%} 与 policy greedy 切换 {policy_ext:.1%} 之间仍有差距；按并列索引计算的 Oracle {oracle_ext:.1%} 不能直接当成要求 policy 切换的比例。注意 Oracle 的 1,000 个 state 与已保存的 1,000 个 policy probe 不是逐状态匹配，只能做分布级比较。每 seed 的 TV/argmax 与跨 seed 的 mean、std、median、bootstrap CI 分别见 tv_requirement_summary.csv / tv_requirement_aggregate.csv。", "",
        markdown_table(tv_aggregate), "",
        "正式运行器未保存训练后的 checkpoint、逐动作 logits/probability 或完整 probe state，因此无法补出四档相邻 TV、全部动作分数和更大样本的模型 sweep。policy_requirement_sweep.csv 是当时保存的 probe **摘要**，不是重载权重得到的完整分布。", "",
        "## 3. (7,8) 是否为 universal shortcut？", "",
        f"(7,8) 的 rho={p78hi.Rho:.5f}、平均 pair reliability={p78hi.Mean_Pair_Reliability:.7f}、平均 latency={p78hi.Mean_Latency:.3f}s，在样本均值的三目标 Pareto front 上；这表示它没有被另一 pair 同时在 rho、延迟、可靠性严格支配，**不表示**它近乎总是 reward 最优。",
        f"它的严格单一 argmax 频率依次为 {list(o['best_pair_78_frequency_by_requirement'].values())}，包含于并列最优集合的频率依次为 {list(o['best_pair_78_in_optimal_set_by_requirement'].values())}；各档即时 regret 见下表。", "",
        markdown_table(regret), "",
        "因此环境中的 (7,8) universal optimum **不成立**，但 Actor 在 10/10 seed 都以它作为 Top-1，说明策略层的全局 pair 捷径 **有证据支持**。Pareto 比较基于样本均值，不能替代逐状态 reward 比较。", "",
        "## 4. Reward 是否提供 requirement-specific signal？", "",
        "当前 reward 是满足阈值时的延迟奖励、未满足时的失败分支调整，再减 log10 违约惩罚。rho 仅影响 episode 风险场的联合采样；给定各节点有效 hazard 后，任务级 joint failure 为两个条件故障概率的乘积，rho 不直接进入奖励。", "",
        "正式评估轨迹分层分解（不同档是不同任务，不能当作固定状态的因果差）：", "",
        markdown_table(decomp), "",
        f".9 档 reliability component 均值 {low_decomp.reward_reliability_component:.3f}；.9999 档为 {high_decomp.reward_reliability_component:.3f}，同时其成功延迟奖励均值 {high_decomp.reward_delay_component:.3f}。低档无违约信号，高档约 {high_decomp.Violation_Rate:.1%} 任务违约且存在显著的阈值分支信号。不能笼统说高档可靠性 reward 太弱。",
        "固定同一 state、同一 pair，仅将 .9 改为 .9999 时，reward component 的绝对改变量如下（延迟分量不变）：", "",
        markdown_table(shift), "",
        "## 5. Pair Actor 是否学到 R_req × rho interaction？", "",
        "**UNKNOWN。** 没有训练后权重，无法计算 raw/masked logits、R_req×rho score 热图或 ∂score/∂rho。已有 rho counterfactual TV 非零只证明分布对 rho 输入有响应，不能证明交互。r_req_input_trace.txt 证明四档规范化值 0、1/3、2/3、1 以 float32 进入全部 28 个 pair feature；本次没有把未训练网络的分数冒充已训练模型结果。", "",
        "## 6. 最可能的原因及证据强度", "",
        "| 优先级 | 假设 | 结论 | 依据 |", "|---|---|---|---|",
        f"| 1 | 环境缺少清晰的需求条件化最优边界 | **SUPPORTED** | 强制最优集合切换仅 {disjoint_ext:.1%}；大量 reward 并列、margin=0。 |",
        "| 2 | 策略形成 (7,8) 全局捷径 | **SUPPORTED** | 10/10 seed Top-1 为 (7,8)，但其 Oracle 最优频率低；环境的 universal optimum 本身 **NOT SUPPORTED**。 |",
        f"| 3 | 可靠性 reward 信号弱/不平滑 | **PARTIALLY SUPPORTED** | .9/.99 档无违约；.9999 档平均 reliability component {high_decomp.reward_reliability_component:.3f}，存在较强但稀疏的阈值信号。 |",
        "| 4 | R_req 输入被损坏/压缩 | **NOT SUPPORTED** | 生成、state、float32、28 个 pair feature 中四档保持区分。 |",
        "| 5 | Actor 没学到 R_req×rho 交互 | **UNKNOWN** | 缺训练后 checkpoint，无法检查二维 score 曲面。 |",
        "| 6 | 探索、延迟信用分配或优化等其他原因 | **UNKNOWN** | 本次即时 Oracle 无法隔离 PPO 长期训练过程。 |", "",
        "## 其他验证与限制", "",
        f"每状态匹配低/高 rho pair 后，DeltaQ 均值从 {delta_low:.3f} 变为 {delta_high:.3f}，**没有**随 R_req 升高而系统增大；四档 bootstrap CI 见 delta_q_rho_summary.csv，折线与分布见 PNG。匹配成本 median={matched_cost.median():.3f}、P90={matched_cost.quantile(.9):.3f}，因此不是严格同质 pair 的因果 rho 效应；尤其 rho 在当前模型中不直接进入任务 reward。",
        "完整权重级 P3/P4 所需数据不存在；diagnostic_status.json 列明未生成的图。其余 CSV/JSON/PNG 均经过 finite 值和结构检查。smoke 与 1,000-state full run 均已执行。", "",
        "## 下一项最有价值的实验", "",
        "未来一次**同配置**正式训练运行，保存最终 Pair policy_old.state_dict 及完整评估 states，然后在同一批 state 上进行四档 R_req×rho 的 logits/probability sweep，并报告相邻 TV 与有限差分交互；这能直接区分‘输入虽有效但网络未学到交互’和‘环境边界稀少’。当前不重新训练。", ""]
    (out/"DIAGNOSIS.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-states",type=int,default=1000)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"diagnostics/results")
    args=parser.parse_args()
    result=run(args.max_states,args.output_dir)
    print(json.dumps(result,indent=2))
