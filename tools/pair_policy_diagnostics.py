"""Pure analysis helpers for PPO pair-policy behavior diagnostics."""

from __future__ import annotations

import copy
import math
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from agents.ppo_agent import PPOAgent
from core.env_state import EnvironmentState


RELIABILITY_REQUIREMENT_TIERS = (0.9, 0.99, 0.999, 0.9999)
PAIR_DIAGNOSTIC_SEED = 2041
PAIR_DIAGNOSTIC_TORCH_SEED = 2042

# This mirrors the named order used by save_params_and_logs when it converts
# MainLoop.task_Assignments_info. The saver currently defines this schema
# locally rather than exporting a reusable canonical-column constant.
TASK_ASSIGNMENT_COLUMNS = [
    "episode",
    "task_id",
    "Primary",
    "Primary_Start",
    "Primary_End",
    "Primary_Status",
    "Backup",
    "Backup_Start",
    "Backup_End",
    "Backup_Status",
    "Z",
    "Reliability_Requirement",
    "Primary_Effective_Failure_Rate",
    "Backup_Effective_Failure_Rate",
    "Primary_Reliability_Service_Time",
    "Backup_Reliability_Service_Time",
    "Primary_Failure_Probability",
    "Backup_Failure_Probability",
    "Joint_Failure_Probability",
    "Execution_Reliability",
    "Reliability_Satisfied",
    "Task_Reward",
    "Task_Delay",
    "Reliability_Margin",
    "Reliability_Shortfall",
    "Reliability_Excess",
    "Base_Reward",
    "Reliability_Violation_Log10",
    "Reliability_Penalty",
    "action_index",
    "server_j",
    "server_k",
]


class DiagnosticPPOAgent(PPOAgent):
    """PPOAgent subclass that only archives copies of observed states."""

    def __init__(self, *args, **kwargs):
        self.observation_archive = []
        super().__init__(*args, **kwargs)

    def store_transition(self, s, a, r, s_next, delta_t, done=False, task_id=None):
        self.observation_archive.append(
            {
                "state": np.array(s, dtype=np.float32, copy=True),
                "task_id": task_id,
                "action": int(a),
            }
        )
        return super().store_transition(
            s,
            a,
            r,
            s_next,
            delta_t,
            done=done,
            task_id=task_id,
        )


def _probability_vector(values, name="probabilities"):
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(vector).all() or (vector < 0.0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.isclose(vector.sum(), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} must sum to one")
    return vector


def get_action_probabilities(policy, state) -> np.ndarray:
    """Return the complete action distribution without sampling or mutation."""
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.ndim != 1 or not np.isfinite(state_array).all():
        raise ValueError("state must be a finite one-dimensional vector")
    try:
        parameter = next(policy.parameters())
        device = parameter.device
    except StopIteration:
        try:
            device = next(policy.buffers()).device
        except StopIteration:
            device = torch.device("cpu")
    except AttributeError as exc:
        raise ValueError("policy must be a callable torch policy") from exc

    state_tensor = torch.as_tensor(state_array, dtype=torch.float32, device=device)
    try:
        with torch.no_grad():
            logits = policy(state_tensor)
    except Exception as exc:
        raise ValueError(f"policy failed to produce action logits: {exc}") from exc
    if not isinstance(logits, torch.Tensor):
        raise ValueError("policy logits must be a torch tensor")
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 1 or logits.shape[0] != 28:
        raise ValueError(f"policy must produce 28 action logits, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("policy logits must be finite")
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("policy probabilities must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("policy probabilities must sum to one")
    return probabilities


def total_variation_distance(probabilities_p, probabilities_q) -> float:
    p = _probability_vector(probabilities_p, "probabilities_p")
    q = _probability_vector(probabilities_q, "probabilities_q")
    if p.shape != q.shape:
        raise ValueError("probability vectors must have the same shape")
    return float(0.5 * np.abs(p - q).sum())


def jensen_shannon_divergence(probabilities_p, probabilities_q) -> float:
    p = _probability_vector(probabilities_p, "probabilities_p")
    q = _probability_vector(probabilities_q, "probabilities_q")
    if p.shape != q.shape:
        raise ValueError("probability vectors must have the same shape")
    middle = 0.5 * (p + q)

    def kl_to_middle(distribution):
        positive = distribution > 0.0
        return float(np.sum(distribution[positive] * np.log(
            distribution[positive] / middle[positive]
        )))

    divergence = 0.5 * kl_to_middle(p) + 0.5 * kl_to_middle(q)
    return float(max(divergence, 0.0))


def expected_pair_correlation(probabilities, true_pair_correlations) -> float:
    probabilities = _probability_vector(probabilities)
    try:
        correlations = np.asarray(true_pair_correlations, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("true_pair_correlations must be numeric") from exc
    if correlations.shape != probabilities.shape:
        raise ValueError("probabilities and pair correlations must have matching shapes")
    if not np.isfinite(correlations).all() or (
        (correlations < -1e-10).any() or (correlations > 1.0 + 1e-10).any()
    ):
        raise ValueError("true pair correlations must be finite and lie in [0, 1]")
    return float(np.dot(probabilities, np.clip(correlations, 0.0, 1.0)))


def low_rho_quartile_mass(probabilities, true_pair_correlations) -> float:
    probabilities = _probability_vector(probabilities)
    try:
        correlations = np.asarray(true_pair_correlations, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("true_pair_correlations must be numeric") from exc
    if correlations.shape != probabilities.shape:
        raise ValueError("probabilities and pair correlations must have matching shapes")
    if not np.isfinite(correlations).all():
        raise ValueError("true pair correlations must be finite")
    quartile_count = max(1, int(math.ceil(0.25 * len(correlations))))
    lowest_indices = np.argsort(correlations, kind="stable")[:quartile_count]
    return float(probabilities[lowest_indices].sum())


def policy_entropy(probabilities) -> float:
    probabilities = _probability_vector(probabilities)
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def sample_probe_indices(state_count: int, num_probes: int) -> np.ndarray:
    if isinstance(state_count, bool) or int(state_count) < 0:
        raise ValueError("state_count must be a non-negative integer")
    if isinstance(num_probes, bool) or int(num_probes) <= 0:
        raise ValueError("num_probes must be a positive integer")
    if state_count == 0:
        return np.asarray([], dtype=int)
    indices = np.linspace(
        0,
        int(state_count) - 1,
        num=min(int(num_probes), int(state_count)),
        dtype=int,
    )
    return np.unique(indices)


def normalized_reliability_feature(requirement: float) -> float:
    """Reuse EnvironmentState's formal reliability-feature normalization."""
    return EnvironmentState.normalize_reliability_requirement(requirement)


def make_requirement_counterfactual_state(state, requirement: float) -> np.ndarray:
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.shape != (35,) or not np.isfinite(state_array).all():
        raise ValueError("requirement counterfactual requires one finite 35-D state")
    result = state_array.copy()
    result[-1] = normalized_reliability_feature(requirement)
    return result


def build_rho_counterfactual_vectors(true_pair_correlations, seed=PAIR_DIAGNOSTIC_SEED):
    """Build deterministic true/zero/shuffled rho vectors without a policy."""
    true_rho = np.asarray(true_pair_correlations, dtype=float)
    if true_rho.shape != (28,) or not np.isfinite(true_rho).all():
        raise ValueError("true_pair_correlations must be a finite vector of shape (28,)")
    if (true_rho < -1e-10).any() or (true_rho > 1.0 + 1e-10).any():
        raise ValueError("true_pair_correlations must lie in [0, 1]")
    scenarios = {"true": true_rho.copy(), "zero": np.zeros(28, dtype=float)}
    permutation = np.random.default_rng(int(seed)).permutation(len(true_rho))
    scenarios["shuffled"] = true_rho[permutation].copy()
    if not np.array_equal(np.sort(scenarios["shuffled"]), np.sort(true_rho)):
        raise RuntimeError("shuffled rho scenario did not preserve the rho multiset")
    return scenarios


def rho_counterfactual_probabilities(
    policy, state, true_pair_correlations, seed=PAIR_DIAGNOSTIC_SEED
):
    """Evaluate true/zero/shuffled rho on clones; never mutate the input policy."""
    if not hasattr(policy, "pair_correlations"):
        raise ValueError("rho counterfactual is not applicable to a flat policy")
    true_rho = np.asarray(true_pair_correlations, dtype=float)
    scenarios = build_rho_counterfactual_vectors(true_rho, seed=seed)
    original_buffer = policy.pair_correlations.detach().cpu().clone()
    if not torch.equal(original_buffer, torch.as_tensor(true_rho, dtype=torch.float32)):
        raise ValueError("policy pair_correlations do not match the true rho vector")

    result = {}
    for name, scenario_rho in scenarios.items():
        policy_clone = copy.deepcopy(policy)
        with torch.no_grad():
            policy_clone.pair_correlations.copy_(
                torch.as_tensor(
                    scenario_rho,
                    dtype=policy_clone.pair_correlations.dtype,
                    device=policy_clone.pair_correlations.device,
                )
            )
        result[name] = get_action_probabilities(policy_clone, state)
    if not torch.equal(policy.pair_correlations.detach().cpu(), original_buffer):
        raise RuntimeError("rho counterfactual unexpectedly changed the original policy")
    return result


def action_pairs_for_server_count(num_servers: int):
    return list(combinations(range(1, int(num_servers) + 1), 2))


def _top_action_fields(probabilities, true_rho, action_pairs):
    action_index = int(np.argmax(probabilities))
    server_j, server_k = action_pairs[action_index]
    return {
        "Top_Action_Index": action_index,
        "Top_Server_J": int(server_j),
        "Top_Server_K": int(server_k),
        "Top_Action_Probability": float(probabilities[action_index]),
        "Top_Action_True_Rho": float(true_rho[action_index]),
        "Expected_True_Rho": expected_pair_correlation(probabilities, true_rho),
        "Low_Rho_Quartile_Mass": low_rho_quartile_mass(probabilities, true_rho),
        "Policy_Entropy": policy_entropy(probabilities),
    }


def requirement_sensitivity_rows(policy, probes, true_rho, action_pairs):
    rows = []
    for probe_id, probe in enumerate(probes):
        state = np.asarray(probe["state"], dtype=np.float32)
        tier_distributions = {}
        tier_metrics = {}
        for requirement in RELIABILITY_REQUIREMENT_TIERS:
            probe_state = make_requirement_counterfactual_state(state, requirement)
            probabilities = get_action_probabilities(policy, probe_state)
            metrics = _top_action_fields(probabilities, true_rho, action_pairs)
            tier_distributions[requirement] = probabilities
            tier_metrics[requirement] = metrics
        baseline = tier_distributions[0.9]
        for requirement in RELIABILITY_REQUIREMENT_TIERS:
            probabilities = tier_distributions[requirement]
            metrics = tier_metrics[requirement]
            rows.append(
                {
                    "Probe_ID": f"probe-{probe_id:04d}",
                    "Task_ID": probe.get("task_id"),
                    "Reliability_Requirement": requirement,
                    "Normalized_Reliability_Feature": normalized_reliability_feature(requirement),
                    **metrics,
                    "TV_From_R09": total_variation_distance(probabilities, baseline),
                    "JS_From_R09": jensen_shannon_divergence(probabilities, baseline),
                    "Argmax_Changed_From_R09": bool(
                        np.argmax(probabilities) != np.argmax(baseline)
                    ),
                }
            )
    return pd.DataFrame(rows)


def rho_counterfactual_rows(policy, probes, true_rho, action_pairs):
    if not hasattr(policy, "pair_correlations"):
        return None
    rows = []
    for probe_id, probe in enumerate(probes):
        state = np.asarray(probe["state"], dtype=np.float32)
        scenario_probabilities = rho_counterfactual_probabilities(
            policy, state, true_rho
        )
        true_probabilities = scenario_probabilities["true"]
        for scenario in ("true", "zero", "shuffled"):
            probabilities = scenario_probabilities[scenario]
            metrics = _top_action_fields(probabilities, true_rho, action_pairs)
            if scenario == "true":
                tv_from_true = 0.0
                js_from_true = 0.0
                argmax_changed = False
            else:
                tv_from_true = total_variation_distance(probabilities, true_probabilities)
                js_from_true = jensen_shannon_divergence(probabilities, true_probabilities)
                argmax_changed = bool(
                    np.argmax(probabilities) != np.argmax(true_probabilities)
                )
            rows.append(
                {
                    "Probe_ID": f"probe-{probe_id:04d}",
                    "Task_ID": probe.get("task_id"),
                    "Rho_Scenario": scenario,
                    **metrics,
                    "TV_From_True": tv_from_true,
                    "JS_From_True": js_from_true,
                    "Argmax_Changed_From_True": argmax_changed,
                }
            )
    return pd.DataFrame(rows)


def summarize_behavior(actor_mode, requirement_df, rho_df=None):
    if requirement_df.empty:
        raise ValueError("requirement sensitivity results must not be empty")
    summary = {"Num_Probe_States": int(requirement_df["Probe_ID"].nunique())}
    for requirement, label in zip(RELIABILITY_REQUIREMENT_TIERS, ("R09", "R099", "R0999", "R09999")):
        tier = requirement_df[
            np.isclose(requirement_df["Reliability_Requirement"], requirement)
        ]
        summary[f"Mean_Expected_Rho_{label}"] = float(tier["Expected_True_Rho"].mean())
        summary[f"Mean_Low_Rho_Mass_{label}"] = float(
            tier["Low_Rho_Quartile_Mass"].mean()
        )
    summary["Mean_Delta_Expected_Rho_HighMinusLow"] = (
        summary["Mean_Expected_Rho_R09999"] - summary["Mean_Expected_Rho_R09"]
    )
    high_rows = requirement_df[
        np.isclose(requirement_df["Reliability_Requirement"], 0.9999)
    ]
    summary["Mean_TV_R09999_vs_R09"] = float(high_rows["TV_From_R09"].mean())
    summary["Mean_JS_R09999_vs_R09"] = float(high_rows["JS_From_R09"].mean())
    summary["Argmax_Change_Rate_R09999_vs_R09"] = float(
        high_rows["Argmax_Changed_From_R09"].astype(bool).mean()
    )

    if actor_mode == "pair_scoring" and rho_df is not None:
        zero_rows = rho_df[rho_df["Rho_Scenario"].eq("zero")]
        shuffled_rows = rho_df[rho_df["Rho_Scenario"].eq("shuffled")]
        summary["Mean_TV_True_vs_ZeroRho"] = float(zero_rows["TV_From_True"].mean())
        summary["Mean_JS_True_vs_ZeroRho"] = float(zero_rows["JS_From_True"].mean())
        summary["Argmax_Change_Rate_True_vs_ZeroRho"] = float(
            zero_rows["Argmax_Changed_From_True"].astype(bool).mean()
        )
        summary["Mean_TV_True_vs_ShuffledRho"] = float(
            shuffled_rows["TV_From_True"].mean()
        )
        summary["Mean_JS_True_vs_ShuffledRho"] = float(
            shuffled_rows["JS_From_True"].mean()
        )
        summary["Argmax_Change_Rate_True_vs_ShuffledRho"] = float(
            shuffled_rows["Argmax_Changed_From_True"].astype(bool).mean()
        )
    else:
        for column in (
            "Mean_TV_True_vs_ZeroRho",
            "Mean_JS_True_vs_ZeroRho",
            "Argmax_Change_Rate_True_vs_ZeroRho",
            "Mean_TV_True_vs_ShuffledRho",
            "Mean_JS_True_vs_ShuffledRho",
            "Argmax_Change_Rate_True_vs_ShuffledRho",
        ):
            summary[column] = "Not Applicable for flat actor"
    return summary


def performance_summary(task_assignment_rows, true_rho):
    frame = pd.DataFrame(task_assignment_rows, columns=TASK_ASSIGNMENT_COLUMNS)
    if frame.empty:
        raise ValueError("MainLoop produced no task assignment rows")
    for column in (
        "Reliability_Requirement",
        "Task_Reward",
        "Task_Delay",
        "action_index",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Reliability_Satisfied"] = frame["Reliability_Satisfied"].map(
        lambda value: value if isinstance(value, (bool, np.bool_)) else (
            str(value).strip().lower() == "true" if pd.notna(value) else np.nan
        )
    )
    frame["Selected_Rho"] = frame["action_index"].map(
        lambda action: float(true_rho[int(action)])
        if pd.notna(action) and 0 <= int(action) < len(true_rho)
        else np.nan
    )

    def metrics(group):
        return {
            "Count": int(len(group)),
            "Reliability_Satisfaction_Rate": float(
                pd.to_numeric(group["Reliability_Satisfied"], errors="coerce").mean()
            ),
            "Mean_Task_Delay": float(group["Task_Delay"].mean()),
            "Mean_Reward": float(group["Task_Reward"].mean()),
            "Mean_Selected_Rho": float(group["Selected_Rho"].mean()),
        }

    overall = metrics(frame)
    by_requirement = []
    for requirement in RELIABILITY_REQUIREMENT_TIERS:
        group = frame[np.isclose(frame["Reliability_Requirement"], requirement)]
        by_requirement.append(
            {"Reliability_Requirement": requirement, **metrics(group)}
            if not group.empty
            else {
                "Reliability_Requirement": requirement,
                "Count": 0,
                "Reliability_Satisfaction_Rate": np.nan,
                "Mean_Task_Delay": np.nan,
                "Mean_Reward": np.nan,
                "Mean_Selected_Rho": np.nan,
            }
        )
    return overall, pd.DataFrame(by_requirement)
