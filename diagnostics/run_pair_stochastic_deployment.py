"""Evaluate frozen Pair PPO checkpoints with the native stochastic policy.

This is deployment-only: it loads the existing formal 10-seed Pair checkpoints,
performs no optimizer steps, and uses a private Torch generator for policy draws
so task-arrival and spatial-risk streams remain independent.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
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
from config.params import params
from diagnostics.evaluate_reliability_masked_policy import (
    ArrivalTraceLoop,
    TrackingEnvironmentState,
    summarize_server_load,
)
from diagnostics.run_masked_pair_ppo import trace
from diagnostics.run_masked_pair_ppo_10seed import OUT, action_seed, formal_spec
from agents.masked_pair_ppo_agent import production_reliability_vector
from Project_main import build_pair_correlations
from tools.pair_policy_diagnostics import DiagnosticPPOAgent, TASK_ASSIGNMENT_COLUMNS
from tools.paired_ppo_experiment import (
    FrozenEvaluationPPOAgent,
    _agent_kwargs,
    _state_dict_snapshot,
    assert_state_dict_unchanged,
    scoped_environment_seeds,
    sha256_file,
)

DEPLOYMENT_OUT = OUT / "pair_stochastic"
POLICY_RNG_TAG = 0x50414952  # "PAIR", distinct from environment seed streams.


def native_policy_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return the unmasked, untempered Pair Actor distribution."""
    if logits.ndim != 1 or logits.numel() != params.num_actions:
        raise ValueError(f"Expected {params.num_actions} Pair logits, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Pair Actor produced non-finite logits")
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all() or not torch.isclose(
        probabilities.sum(), torch.ones((), dtype=probabilities.dtype, device=probabilities.device),
        rtol=0.0, atol=1e-6,
    ):
        raise RuntimeError("Native Pair policy probabilities are invalid")
    return probabilities


def sample_native_action(logits: torch.Tensor, generator: torch.Generator) -> tuple[int, torch.Tensor]:
    """Sample from the complete legal action distribution with a private RNG."""
    probabilities = native_policy_probabilities(logits)
    if probabilities.device.type != "cpu" or generator.device.type != "cpu":
        raise ValueError("The formal deployment runner uses a CPU policy and CPU action generator")
    action = int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())
    return action, probabilities


def policy_action_seed(trial: dict) -> int:
    """Derive a repeatable policy-only seed, independent of environment seeds."""
    return int(np.random.SeedSequence([
        int(trial["Trial_ID"]), int(action_seed(trial)), POLICY_RNG_TAG,
    ]).generate_state(1, dtype=np.uint32)[0])


class PairStochasticEvaluationAgent(FrozenEvaluationPPOAgent):
    """Frozen baseline Pair Actor sampled from its native distribution."""

    def __init__(self, trained_agent, pairs, action_rng_seed):
        super().__init__(trained_agent)
        self.pairs = list(pairs)
        self.policy_action_generator = torch.Generator(device="cpu")
        self.policy_action_generator.manual_seed(int(action_rng_seed))
        self.selection_archive = []
        self.policy_probability_rows = []

    def select_action(self, state, epsilon=0.0, use_softmax=False, temperature=1.5):
        task, env_state, episode = self.current_context
        with torch.no_grad():
            logits = self.policy_old(self._to_tensor(state).unsqueeze(0)).squeeze(0)
            action, probabilities = sample_native_action(logits, self.policy_action_generator)
            reliability = production_reliability_vector(task, env_state, self.pairs)

        safe_mask = reliability >= float(task.reliability_requirement)
        safe_count = int(safe_mask.sum())
        best_reliability = float(reliability.max())
        probability_np = probabilities.detach().cpu().numpy().astype(np.float64)
        entropy = float(-(probability_np[probability_np > 0] * np.log(
            probability_np[probability_np > 0]
        )).sum())
        selected_pair = self.pairs[action]
        selected_safe = bool(safe_mask[action])
        self.selection_archive.append({
            "episode": int(episode),
            "task_id": int(task.id),
            "action_index": action,
            "selected_pair": str(selected_pair),
            "server_j": int(selected_pair[0]),
            "server_k": int(selected_pair[1]),
            "R_req": float(task.reliability_requirement),
            "safe_set_size": safe_count,
            "safe_set_empty": safe_count == 0,
            "safe_mask": safe_mask.astype(int).tolist(),
            "best_achievable_reliability": best_reliability,
            "reliability_deficit": max(float(task.reliability_requirement) - best_reliability, 0.0),
            "selected_action_safe": selected_safe,
            "selected_pair_reliability": float(reliability[action]),
            "selected_rho": float(self.pair_correlations[action]),
            "selected_action_probability": float(probability_np[action]),
            "native_policy_entropy": entropy,
        })
        self.policy_probability_rows.append(probability_np)
        return action


def make_pair_agent(trial, pairs, rho, trial_dir):
    checkpoint_dir = trial_dir / "pair"
    manifest = json.loads((trial_dir / "completed.json").read_text(encoding="utf-8"))
    manifest_row = next(row for row in manifest["runs"] if row["policy"] == "pair")
    actor_path = checkpoint_dir / "actor.pt"
    critic_path = checkpoint_dir / "critic.pt"
    if sha256_file(actor_path) != manifest_row["actor_sha256"]:
        raise RuntimeError(f"Trial {trial['Trial_ID']}: Pair Actor checkpoint hash mismatch")
    if sha256_file(critic_path) != manifest_row["critic_sha256"]:
        raise RuntimeError(f"Trial {trial['Trial_ID']}: Pair Critic checkpoint hash mismatch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(trial["Torch_Init_Seed"]))
        agent = DiagnosticPPOAgent(**_agent_kwargs(
            "pair_scoring", rho, int(trial["PPO_Minibatch_Seed"])
        ))
    actor_state = torch.load(actor_path, map_location="cpu", weights_only=True)
    critic_state = torch.load(critic_path, map_location="cpu", weights_only=True)
    agent.policy_net.load_state_dict(actor_state)
    agent.policy_old.load_state_dict(actor_state)
    agent.value_net.load_state_dict(critic_state)
    agent.pair_correlations = np.asarray(rho, dtype=float)
    return agent, manifest_row


def assignments_frame(loop):
    result = pd.DataFrame(loop.task_Assignments_info, columns=TASK_ASSIGNMENT_COLUMNS)
    return result.sort_values(["episode", "task_id"]).reset_index(drop=True)


def evaluate(agent, trial, run_dir, pairs, mode):
    TrackingEnvironmentState.instances = []
    TrackingEnvironmentState.active_agent = agent
    before = _state_dict_snapshot(agent)
    with patch.object(main_loop_module, "EnvironmentState", TrackingEnvironmentState):
        with scoped_environment_seeds(
            int(trial["Eval_Arrival_Seed"]), int(trial["Eval_Spatial_Seed"])
        ):
            with (run_dir / f"{mode}_evaluation.log").open("w", encoding="utf-8") as log:
                with redirect_stdout(log):
                    loop = ArrivalTraceLoop(agent, 20, 200, params.num_states, params.num_actions)
                    loop.EP()
    assert_state_dict_unchanged(before, agent)
    for state in TrackingEnvironmentState.instances:
        state.finalize()
    assignments = assignments_frame(loop)
    if len(assignments) != 4000:
        raise RuntimeError(f"Trial {trial['Trial_ID']} {mode}: expected 4000 task assignments")
    if not hasattr(agent, "selection_archive"):
        return loop, assignments, None, summarize_server_load(
            TrackingEnvironmentState.instances, assignments, mode
        )
    if len(agent.selection_archive) != 4000:
        raise RuntimeError(f"Trial {trial['Trial_ID']} {mode}: expected 4000 decisions")
    decisions = pd.DataFrame(agent.selection_archive).sort_values(
        ["episode", "task_id"]
    ).reset_index(drop=True)
    if not np.array_equal(assignments.action_index.to_numpy(), decisions.action_index.to_numpy()):
        raise RuntimeError(f"Trial {trial['Trial_ID']} {mode}: action log differs from simulator")
    actions = decisions.action_index.to_numpy(dtype=int)
    if ((actions < 0) | (actions >= len(pairs))).any():
        raise RuntimeError(f"Trial {trial['Trial_ID']} {mode}: illegal pair action")
    if not np.allclose(
        decisions.selected_pair_reliability.to_numpy(dtype=float),
        assignments.Execution_Reliability.to_numpy(dtype=float), rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError(f"Trial {trial['Trial_ID']} {mode}: post-hoc reliability mismatch")
    if not np.array_equal(
        decisions.selected_action_safe.astype(bool).to_numpy(),
        assignments.Reliability_Satisfied.astype(bool).to_numpy(),
    ):
        raise RuntimeError(f"Trial {trial['Trial_ID']} {mode}: feasibility rule mismatch")
    load = summarize_server_load(TrackingEnvironmentState.instances, assignments, mode)
    return loop, assignments, decisions, load


def compare_pair_greedy_replay(trial_id, replay):
    old_path = OUT / "runs" / f"trial_{trial_id:03d}" / "pair" / "evaluation_task_assignments.csv"
    old = pd.read_csv(old_path).sort_values(["episode", "task_id"]).reset_index(drop=True)
    if len(old) != 4000 or len(replay) != 4000:
        raise RuntimeError(f"Trial {trial_id}: Pair greedy reference does not have 4000 tasks")
    exact_columns = ["episode", "task_id", "action_index", "Primary", "Backup"]
    for column in exact_columns:
        if not np.array_equal(old[column].to_numpy(), replay[column].to_numpy()):
            raise RuntimeError(f"Trial {trial_id}: greedy replay mismatch in {column}")
    for column in ("Task_Reward", "Task_Delay", "Execution_Reliability"):
        if not np.allclose(
            old[column].to_numpy(dtype=float), replay[column].to_numpy(dtype=float),
            rtol=0.0, atol=1e-10,
        ):
            raise RuntimeError(f"Trial {trial_id}: greedy replay mismatch in {column}")
    return {
        "trial_id": int(trial_id),
        "tasks_compared": int(len(old)),
        "action_mismatch_count": 0,
        "reward_max_abs_error": float(np.max(np.abs(old.Task_Reward - replay.Task_Reward))),
        "latency_max_abs_error": float(np.max(np.abs(old.Task_Delay - replay.Task_Delay))),
        "reliability_max_abs_error": float(np.max(np.abs(
            old.Execution_Reliability - replay.Execution_Reliability
        ))),
    }


def run_trial(trial_id):
    meta, plan = formal_spec()
    trial = plan.iloc[int(trial_id)].to_dict()
    trial_dir = OUT / "runs" / f"trial_{trial_id:03d}"
    target = DEPLOYMENT_OUT / f"trial_{trial_id:03d}"
    marker = target / "completed.json"
    if marker.exists():
        print(f"trial {trial_id} already evaluated; skipping", flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    pairs, rho = build_pair_correlations()
    pairs = list(pairs)
    rho = np.asarray(rho, dtype=float)
    if len(pairs) != params.num_actions:
        raise RuntimeError("Formal Pair action set does not match num_actions")
    agent, manifest = make_pair_agent(trial, pairs, rho, trial_dir)
    original_hash = manifest["actor_sha256"]

    # Re-run deterministic Pair greedy once per seed to anchor both the saved
    # result and its external traces before evaluating stochastic deployment.
    greedy = FrozenEvaluationPPOAgent.from_trained(agent)
    greedy.pairs = pairs
    greedy_loop, greedy_assignments, _, _ = evaluate(greedy, trial, target, pairs, "greedy_replay")
    greedy_check = compare_pair_greedy_replay(trial_id, greedy_assignments)

    action_seed_value = policy_action_seed(trial)
    stochastic = PairStochasticEvaluationAgent(agent, pairs, action_seed_value)
    stochastic_loop, assignments, decisions, load = evaluate(
        stochastic, trial, target, pairs, "stochastic"
    )
    # Arrival draws and the full sampled spatial-risk field are exogenous and
    # must be identical although the queue states and actions diverge.
    greedy_trace, stochastic_trace = trace(greedy_loop), trace(stochastic_loop)
    if not np.array_equal(greedy_trace[0], stochastic_trace[0]):
        raise RuntimeError(f"Trial {trial_id}: stochastic deployment changed task-arrival draws")
    if not greedy_trace[1].equals(stochastic_trace[1]):
        raise RuntimeError(f"Trial {trial_id}: stochastic deployment changed spatial-risk draws")

    assignments.to_csv(target / "evaluation_task_assignments.csv", index=False)
    decisions = decisions.copy()
    decisions["safe_mask"] = decisions.safe_mask.map(json.dumps)
    decisions.to_csv(target / "evaluation_decisions.csv", index=False)
    load.insert(0, "trial_id", int(trial_id))
    load.to_csv(target / "server_load.csv", index=False)
    probabilities = np.stack(stochastic.policy_probability_rows)
    if probabilities.shape != (4000, params.num_actions):
        raise RuntimeError(f"Trial {trial_id}: action probability archive has wrong shape")
    np.savez_compressed(target / "evaluation_policy_probabilities.npz", probabilities=probabilities)
    metadata = {
        "trial_id": int(trial_id),
        "policy": "Pair PPO native stochastic, no reliability mask",
        "checkpoint_actor_sha256": original_hash,
        "checkpoint_critic_sha256": manifest["critic_sha256"],
        "action_rng_seed": action_seed_value,
        "action_rng_tag": POLICY_RNG_TAG,
        "distribution": "Categorical(logits=policy_old(state)); implemented by multinomial(softmax(logits))",
        "temperature": 1.0,
        "reliability_mask_applied": False,
        "posthoc_feasibility_rule": "Task.initialize_reliability_evaluation; execution_reliability >= reliability_requirement",
        "evaluation_episodes": 20,
        "tasks_per_episode": 200,
        "evaluated_tasks": 4000,
        "greedy_replay_matches_formal_pair_results": True,
        "arrival_stream_matches_greedy_replay": True,
        "spatial_risk_stream_matches_greedy_replay": True,
        "training_performed": False,
        "formal_reference_commit": meta["git_commit"],
    }
    (target / "greedy_replay_validation.json").write_text(
        json.dumps(greedy_check, indent=2), encoding="utf-8"
    )
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    marker.write_text(json.dumps({"trial_id": int(trial_id), "complete": True}, indent=2), encoding="utf-8")
    print(json.dumps({"trial_id": int(trial_id), "tasks": len(assignments),
                      "actor_sha256": original_hash, "action_rng_seed": action_seed_value,
                      "greedy_replay": greedy_check,
                      "native_sample_entropy_mean": float(decisions.native_policy_entropy.mean()),
                      "selected_safe_rate_posthoc": float(decisions.selected_action_safe.mean())}, indent=2),
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", type=int, choices=range(10))
    args = parser.parse_args()
    _, plan = formal_spec()
    trials = [args.trial_id] if args.trial_id is not None else list(range(len(plan)))
    for trial_id in trials:
        run_trial(trial_id)


if __name__ == "__main__":
    main()
