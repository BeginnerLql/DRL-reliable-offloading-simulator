"""Compare four frozen Pair PPO selectors under matched exogenous simulation seeds.

The feasibility calculation calls Task.initialize_reliability_evaluation on an
unregistered probe, exactly reusing the production episode-effective hazard and
pair-reliability logic. Queue statistics instrument EnvironmentState events only.
No Actor, PPO update, reward, or simulator production code is changed.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

import core.main_loop as main_loop_module
from config.params import params
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.task import Task
from Project_main import build_pair_correlations
from tools.pair_policy_diagnostics import DiagnosticPPOAgent, TASK_ASSIGNMENT_COLUMNS
from tools.paired_ppo_experiment import (
    FrozenEvaluationPPOAgent, _agent_kwargs, _state_dict_snapshot,
    assert_state_dict_unchanged, scoped_environment_seeds, sha256_file,
)

DEFAULT_CHECKPOINT = ROOT / "diagnostics/results/policy_oracle_alignment"
DEFAULT_OUTPUT = ROOT / "diagnostics/results/reliability_masked_policy"
POLICIES = ("greedy", "native_sample", "masked_sample", "masked_greedy")


def production_pair_reliability(task: Task, env_state: EnvironmentState, pairs):
    """Run production reliability initialization for every legal pair."""
    probe = object.__new__(Task)
    probe.env_state = env_state
    probe.computation_demand = task.computation_demand
    probe.reliability_requirement = task.reliability_requirement
    reliability = np.empty(len(pairs), dtype=float)
    failure = np.empty(len(pairs), dtype=float)
    for index, (j, k) in enumerate(pairs):
        probe.initialize_reliability_evaluation(
            env_state.get_server_by_id(j), env_state.get_server_by_id(k)
        )
        reliability[index] = probe.execution_reliability
        failure[index] = probe.joint_failure_probability
    if not np.isfinite(reliability).all() or not np.isfinite(failure).all():
        raise RuntimeError("Production pair reliability returned non-finite values")
    return reliability, failure


def select_from_probabilities(mode, probabilities, reliability, requirement):
    """Apply a production-reliability feasibility mask with explicit fallback."""
    probabilities = np.asarray(probabilities, dtype=float)
    reliability = np.asarray(reliability, dtype=float)
    if probabilities.shape != reliability.shape or probabilities.ndim != 1:
        raise ValueError("Probability and reliability vectors must have the same 1-D shape")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-6):
        raise ValueError("Actor probabilities must be finite and sum to one")
    if not np.isfinite(reliability).all():
        raise ValueError("Pair reliability must be finite")
    safe = reliability >= float(requirement)  # identical to Task.reliability_satisfied
    empty = not bool(np.any(safe))
    if mode in ("masked_sample", "masked_greedy"):
        if empty:
            # numpy.argmax has deterministic lowest-index tie breaking.
            return int(np.argmax(reliability)), np.zeros_like(probabilities), safe, True
        masked = np.where(safe, probabilities, 0.0)
        masked /= masked.sum()
        if mode == "masked_greedy":
            action = int(np.argmax(masked))
        else:
            action = int(Categorical(probs=torch.as_tensor(masked, dtype=torch.float32)).sample().item())
        return action, masked, safe, False
    if mode == "greedy":
        action = int(np.argmax(probabilities))
    elif mode == "native_sample":
        action = int(Categorical(probs=torch.as_tensor(probabilities, dtype=torch.float32)).sample().item())
    else:
        raise ValueError(f"Unknown policy mode: {mode}")
    return action, probabilities.copy(), safe, empty


class TrackingEnvironmentState(EnvironmentState):
    """Read actual waiting/running replica transitions to integrate queue length."""

    instances = []
    active_agent = None

    def __init__(self):
        super().__init__()
        self.episode = len(type(self).instances) + 1
        type(self).instances.append(self)
        self._last = {}
        self._queue_area = {}
        self._busy_time = {}
        self._segments = {}
        self._enqueued = {}
        self._waits = {}

    def add_server_and_init_environment(self, server_object):
        super().add_server_and_init_environment(server_object)
        sid = int(server_object.server_id)
        self._last[sid] = float(server_object.env.now)
        self._queue_area[sid] = self._busy_time[sid] = 0.0
        self._segments[sid] = []
        self._waits[sid] = []

    def _advance(self, server_id, now):
        sid = int(server_id)
        now = float(now)
        duration = now - self._last[sid]
        if duration < -1e-9:
            raise RuntimeError("SimPy time moved backwards")
        if duration > 0:
            info = self.servers[sid]
            queue_length = len(info["waiting_replicas"])
            self._queue_area[sid] += queue_length * duration
            self._busy_time[sid] += int(info["running_replica"] is not None) * duration
            self._segments[sid].append((queue_length, duration))
        self._last[sid] = now

    def get_state(self, task):
        state = super().get_state(task)
        agent = type(self).active_agent
        if agent is None:
            raise RuntimeError("No active frozen diagnostic agent")
        agent.current_context = (task, self, self.episode)
        return state

    def register_waiting_replica(self, server_id, task, selection, service_time):
        self._advance(server_id, task.env.now)
        super().register_waiting_replica(server_id, task, selection, service_time)
        key = (int(server_id), int(task.id), selection)
        if key in self._enqueued:
            raise RuntimeError("Replica was enqueued twice")
        self._enqueued[key] = float(task.env.now)

    def start_replica_execution(self, server_id, task, selection, service_time, service_start_time):
        self._advance(server_id, service_start_time)
        super().start_replica_execution(server_id, task, selection, service_time, service_start_time)
        key = (int(server_id), int(task.id), selection)
        self._waits[int(server_id)].append(float(service_start_time) - self._enqueued.pop(key))

    def complete_replica_execution(self, server_id, task, selection):
        self._advance(server_id, task.env.now)
        super().complete_replica_execution(server_id, task, selection)

    def finalize(self):
        if self._enqueued:
            raise RuntimeError("Episode ended with queued replicas")
        for sid in self.servers:
            self._advance(sid, self.servers[sid]["server_object"].env.now)
            if self.servers[sid]["waiting_replicas"] or self.servers[sid]["running_replica"] is not None:
                raise RuntimeError("Episode ended with unfinished CPU work")


class ArrivalTraceLoop(MainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interarrival_trace = []

    def _sample_interarrival_time(self):
        value = super()._sample_interarrival_time()
        self.interarrival_trace.append(value)
        return value


class FrozenSelectionAgent(FrozenEvaluationPPOAgent):
    def select_action(self, state, epsilon=0.0, use_softmax=False, temperature=1.5):
        task, env_state, episode = self.current_context
        with torch.no_grad():
            logits = self.policy_old(self._to_tensor(state).unsqueeze(0)).squeeze(0)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Frozen Actor produced non-finite logits")
            original = torch.softmax(logits, dim=-1).cpu().numpy().astype(float)
        reliability, failure = production_pair_reliability(task, env_state, self.pairs)
        action, masked, safe, empty = select_from_probabilities(
            "greedy" if self.mode == "native_sample" else self.mode,
            original, reliability, task.reliability_requirement
        )
        if self.mode == "native_sample":
            # Match PPOAgent.select_action's Categorical(logits=...) sampler.
            action = int(Categorical(logits=logits).sample().item())
        if self.mode.startswith("masked") and not empty and not safe[action]:
            raise RuntimeError("Masked selector chose an unsafe pair")
        self.selection_log.append({
            "policy": self.mode, "episode": episode, "task_id": task.id,
            "Reliability_Requirement": float(task.reliability_requirement),
            "safe_set_size": int(safe.sum()), "safe_set_empty": bool(empty),
            "selected_action_safe": bool(safe[action]), "selected_action": action,
            "selected_pair": str(self.pairs[action]), "selected_rho": float(self.rho[action]),
            "selected_pair_reliability": float(reliability[action]),
            "selected_joint_failure_probability": float(failure[action]),
            "probability_before": json.dumps(original.tolist()),
            "probability_after": json.dumps(masked.tolist()),
            "safe_mask": json.dumps(safe.astype(int).tolist()),
        })
        return action


def weighted_quantile(segments, quantile):
    lengths = np.asarray([x[0] for x in segments], dtype=float)
    durations = np.asarray([x[1] for x in segments], dtype=float)
    if not len(lengths) or durations.sum() <= 0:
        return 0.0
    order = np.argsort(lengths)
    return float(lengths[order][np.searchsorted(np.cumsum(durations[order]), quantile * durations.sum(), side="left")])


def summarize_server_load(states, assignments, policy):
    total_horizon = sum(max(state._last.values()) for state in states)
    if total_horizon <= 0:
        raise RuntimeError("Evaluation horizon must be positive")
    rows = []
    for sid in range(1, params.serverNo + 1):
        segments = [segment for state in states for segment in state._segments[sid]]
        waits = [value for state in states for value in state._waits[sid]]
        queue_area = sum(state._queue_area[sid] for state in states)
        busy_time = sum(state._busy_time[sid] for state in states)
        primary = int((assignments.Primary == sid).sum())
        backup = int((assignments.Backup == sid).sum())
        rows.append({"policy":policy,"server_id":sid,"selection_count":primary+backup,
            "primary_count":primary,"backup_count":backup,
            "utilization":busy_time/total_horizon,"mean_queue_length":queue_area/total_horizon,
            "p95_queue_length":weighted_quantile(segments,.95),
            "mean_waiting_time":float(np.mean(waits)) if waits else 0.0,
            "busy_time":busy_time,"observation_time":total_horizon,"replica_wait_count":len(waits)})
    return pd.DataFrame(rows)


def summarize_policy(assignments, selection, pairs, mode):
    counts = assignments.action_index.value_counts().reindex(range(len(pairs)), fill_value=0).to_numpy(dtype=float)
    frequencies = counts / counts.sum()
    positive = frequencies[frequencies > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    def metrics(frame, log):
        return {"policy":mode,"task_count":len(frame),
            "mean_reward":float(frame.Task_Reward.mean()),
            "total_reward":float(frame.Task_Reward.sum()),
            "mean_latency":float(frame.Task_Delay.mean()),
            "p50_latency":float(frame.Task_Delay.quantile(.5)),
            "p90_latency":float(frame.Task_Delay.quantile(.9)),
            "p95_latency":float(frame.Task_Delay.quantile(.95)),
            "rsr":float(frame.Reliability_Satisfied.mean()),
            "reliability_violation_rate":float((~frame.Reliability_Satisfied.astype(bool)).mean()),
            "selected_rho_mean":float(log.selected_rho.mean()),
            "selected_effective_failure_probability_mean":float(log.selected_joint_failure_probability.mean()),
            "selected_pair_reliability_mean":float(log.selected_pair_reliability.mean()),
            "safe_set_empty_rate":float(log.safe_set_empty.mean()),
            "selected_unsafe_rate":float((~log.selected_action_safe).mean()),
            "pair_selection_entropy":entropy if len(frame)==len(assignments) else np.nan,
            "pair_selection_hhi":float(np.sum(frequencies**2)) if len(frame)==len(assignments) else np.nan,
            "unique_pair_count":int(np.count_nonzero(counts)) if len(frame)==len(assignments) else np.nan,
            "top1_pair_frequency":float(frequencies.max()) if len(frame)==len(assignments) else np.nan,
            "pair_78_frequency":float(frequencies[-1]) if len(frame)==len(assignments) else float((frame.action_index==len(pairs)-1).mean())}
    overall = pd.DataFrame([metrics(assignments, selection)])
    tiers = []
    for req, frame in assignments.groupby("Reliability_Requirement"):
        log = selection[np.isclose(selection.Reliability_Requirement, req)]
        item = metrics(frame, log)
        item["R_req"] = req
        tiers.append(item)
    return overall, pd.DataFrame(tiers)


def run(checkpoint_dir=DEFAULT_CHECKPOINT, output_dir=DEFAULT_OUTPUT, eval_episodes=20, tasks_per_episode=200, action_seed=2026):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "run_metadata.json").read_text())
    trial = pd.Series(metadata["formal_seed_row"])
    _, rho = build_pair_correlations()
    rho = np.asarray(rho, dtype=float)
    pairs = list(MainLoop.generate_combinations())
    if len(pairs) != params.num_actions or len(rho) != len(pairs):
        raise RuntimeError("Pair mapping and Actor context disagree")
    trained = DiagnosticPPOAgent(**_agent_kwargs("pair_scoring", rho, int(trial.PPO_Minibatch_Seed)))
    trained.policy_old.load_state_dict(torch.load(checkpoint_dir / "actor_final.pt", map_location="cpu", weights_only=True))
    trained.policy_net.load_state_dict(trained.policy_old.state_dict())
    trained.value_net.load_state_dict(torch.load(checkpoint_dir / "critic_final.pt", map_location="cpu", weights_only=True))
    before = _state_dict_snapshot(trained)
    all_assignments, all_selections, all_server_load = [], [], []
    arrival_reference = risk_reference = None
    for mode in POLICIES:
        frozen = FrozenSelectionAgent.from_trained(trained)
        frozen.mode, frozen.pairs, frozen.rho = mode, pairs, rho
        frozen.selection_log = []
        TrackingEnvironmentState.instances = []
        TrackingEnvironmentState.active_agent = frozen
        torch.manual_seed(int(action_seed))
        with patch.object(main_loop_module, "EnvironmentState", TrackingEnvironmentState):
            with scoped_environment_seeds(int(trial.Eval_Arrival_Seed), int(trial.Eval_Spatial_Seed)):
                with (output_dir / f"simulator_{mode}.log").open("w") as log_file, redirect_stdout(log_file):
                    loop = ArrivalTraceLoop(frozen, eval_episodes, tasks_per_episode,
                                            params.num_states, params.num_actions)
                    loop.EP()
        assert_state_dict_unchanged(before, frozen)
        assert_state_dict_unchanged(before, trained)
        assert len(TrackingEnvironmentState.instances) == eval_episodes
        for state in TrackingEnvironmentState.instances:
            state.finalize()
        assignments = pd.DataFrame(loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS).sort_values(
            ["episode", "task_id"]).reset_index(drop=True)
        selections = pd.DataFrame(frozen.selection_log).sort_values(["episode", "task_id"]).reset_index(drop=True)
        if len(assignments) != eval_episodes * tasks_per_episode or len(assignments) != len(selections):
            raise RuntimeError("Incomplete task-level evaluation")
        assert np.array_equal(assignments.action_index, selections.selected_action)
        assert np.allclose(assignments.Execution_Reliability, selections.selected_pair_reliability, atol=1e-12, rtol=0)
        assert np.array_equal(assignments.Reliability_Satisfied.astype(bool), selections.selected_action_safe)
        assert np.allclose(assignments.Joint_Failure_Probability, selections.selected_joint_failure_probability, atol=1e-12, rtol=0)
        arrivals = np.asarray(loop.interarrival_trace)
        risk = pd.DataFrame(loop.episode_spatial_risk_log).sort_values(["episode", "server_id"]).reset_index(drop=True)
        if arrival_reference is None:
            arrival_reference, risk_reference = arrivals.copy(), risk.copy()
        elif not np.array_equal(arrivals, arrival_reference) or not risk.equals(risk_reference):
            raise RuntimeError("External arrival or spatial-risk streams differ between policies")
        assignments.to_csv(output_dir / f"task_assignments_{mode}.csv", index=False)
        selections.to_csv(output_dir / f"action_diagnostics_{mode}.csv", index=False)
        all_assignments.append(assignments.assign(policy=mode))
        all_selections.append(selections)
        all_server_load.append(summarize_server_load(TrackingEnvironmentState.instances, assignments, mode))
        print(f"{mode}: {len(assignments)} tasks, RSR={assignments.Reliability_Satisfied.mean():.4f}")
    assignments_all = pd.concat(all_assignments, ignore_index=True)
    selections_all = pd.concat(all_selections, ignore_index=True)
    server_load = pd.concat(all_server_load, ignore_index=True)
    summaries, tiers = [], []
    pair_rows = []
    for mode in POLICIES:
        assignments = assignments_all[assignments_all.policy == mode]
        selection = selections_all[selections_all.policy == mode]
        summary, tier = summarize_policy(assignments, selection, pairs, mode)
        summaries.append(summary)
        tiers.append(tier)
        frequency = assignments.action_index.value_counts().reindex(range(len(pairs)), fill_value=0)
        for index, count in frequency.items():
            pair_rows.append({"policy":mode,"action_index":index,"pair":str(pairs[index]),
                              "selection_count":int(count),"selection_frequency":float(count/len(assignments))})
    policy_summary = pd.concat(summaries, ignore_index=True)
    tier_summary = pd.concat(tiers, ignore_index=True)
    pair_frequency = pd.DataFrame(pair_rows)
    safe_stats = selections_all.groupby(["policy", "Reliability_Requirement"], as_index=False).agg(
        tasks=("task_id", "size"), safe_set_empty_count=("safe_set_empty", "sum"),
        safe_set_mean_size=("safe_set_size", "mean"),
        selected_unsafe_count=("selected_action_safe", lambda x: int((~x).sum())),
    )
    safe_stats["safe_set_empty_rate"] = safe_stats.safe_set_empty_count / safe_stats.tasks
    safe_stats["selected_unsafe_rate"] = safe_stats.selected_unsafe_count / safe_stats.tasks
    paired = None
    for mode, assignments, selection in zip(POLICIES, all_assignments, all_selections):
        frame = assignments[["episode","task_id","Reliability_Requirement","action_index","Primary","Backup",
                             "Task_Delay","Execution_Reliability","Joint_Failure_Probability","Task_Reward",
                             "Reliability_Satisfied"]].merge(
            selection[["episode","task_id","selected_rho","safe_set_size","safe_set_empty","selected_action_safe"]],
            on=["episode","task_id"], validate="one_to_one"
        ).rename(columns={"Reliability_Requirement":f"{mode}_R_req","action_index":f"{mode}_action_index",
                           "Primary":f"{mode}_server_j","Backup":f"{mode}_server_k",
                           "Task_Delay":f"{mode}_latency","Execution_Reliability":f"{mode}_pair_reliability",
                           "Joint_Failure_Probability":f"{mode}_joint_failure_probability",
                           "Task_Reward":f"{mode}_reward","Reliability_Satisfied":f"{mode}_reliability_satisfied",
                           "selected_rho":f"{mode}_rho","safe_set_size":f"{mode}_safe_set_size",
                           "safe_set_empty":f"{mode}_safe_set_empty",
                           "selected_action_safe":f"{mode}_selected_action_safe"})
        frame[f"{mode}_pair"] = list(zip(frame[f"{mode}_server_j"], frame[f"{mode}_server_k"]))
        paired = frame if paired is None else paired.merge(frame,on=["episode","task_id"],validate="one_to_one")
    for mode in POLICIES[1:]:
        if not np.allclose(paired.greedy_R_req, paired[f"{mode}_R_req"], atol=1e-12, rtol=0):
            raise RuntimeError("Task reliability requirements differ across policies")
        if not np.array_equal(paired.greedy_safe_set_size, paired[f"{mode}_safe_set_size"]):
            raise RuntimeError("Safe sets differ across matched external conditions")
        paired.drop(columns=f"{mode}_R_req", inplace=True)
    paired.rename(columns={"greedy_R_req":"R_req"}, inplace=True)
    for name, frame in (("policy_summary",policy_summary),("requirement_level_summary",tier_summary),
                        ("server_load_by_policy",server_load),("pair_selection_frequency",pair_frequency),
                        ("paired_action_selection_comparison",paired),("safe_set_statistics",safe_stats)):
        frame.to_csv(output_dir/f"{name}.csv",index=False)
    plot_results(output_dir, policy_summary, tier_summary, server_load, pair_frequency)
    manifest = {"checkpoint_sha256":{name:sha256_file(checkpoint_dir/name) for name in ("actor_final.pt","critic_final.pt")},
                "eval_episodes":eval_episodes,"tasks_per_episode":tasks_per_episode,
                "external_arrival_spatial_risk_streams_identical":True,"frozen_weights_unchanged":True,
                "action_seed":action_seed,"policies":list(POLICIES)}
    (output_dir/"run_metadata.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return policy_summary


def plot_results(output, overall, tiers, load, frequency):
    modes=list(POLICIES)
    def bars(column, filename, ylabel):
        fig,ax=plt.subplots(figsize=(8,4))
        data=overall.set_index("policy").loc[modes,column]
        ax.bar(modes,data)
        ax.set_ylabel(ylabel);ax.tick_params(axis="x",rotation=15)
        fig.tight_layout();fig.savefig(output/filename);plt.close(fig)
    bars("mean_reward","reward_by_policy.png","Mean task reward")
    bars("mean_latency","latency_by_policy.png","Mean task latency (s)")
    bars("rsr","rsr_by_policy.png","Overall RSR")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,column,title in zip(axes,("pair_selection_hhi","pair_selection_entropy"),("HHI","Selection entropy")):
        ax.bar(modes,overall.set_index("policy").loc[modes,column]);ax.set_title(title);ax.tick_params(axis="x",rotation=15)
    fig.tight_layout();fig.savefig(output/"pair_concentration.png");plt.close(fig)
    for column,name,ylabel in (("selection_count","server_selection_distribution.png","Replica selections"),
                              ("mean_queue_length","server_queue_distribution.png","Time-weighted mean queue length")):
        fig,ax=plt.subplots(figsize=(9,4))
        for mode in modes:
            data=load[load.policy==mode].sort_values("server_id")
            ax.plot(data.server_id,data[column],marker="o",label=mode)
        ax.set(xlabel="Server ID",ylabel=ylabel);ax.legend();fig.tight_layout();fig.savefig(output/name);plt.close(fig)
    high=tiers[np.isclose(tiers.R_req, .9999)].set_index("policy").loc[modes]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,column,title in zip(axes,("rsr","mean_latency","mean_reward"),("Highest-tier RSR","Highest-tier latency","Highest-tier reward")):
        ax.bar(modes,high[column]);ax.set_title(title);ax.tick_params(axis="x",rotation=15)
    fig.tight_layout();fig.savefig(output/"highest_requirement_comparison.png");plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir",type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-episodes",type=int,default=20)
    parser.add_argument("--tasks-per-episode",type=int,default=200)
    parser.add_argument("--action-seed",type=int,default=2026)
    args=parser.parse_args()
    print(run(args.checkpoint_dir,args.output_dir,args.eval_episodes,args.tasks_per_episode,args.action_seed).to_string(index=False))
