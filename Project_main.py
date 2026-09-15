# Project_main.py
# ------------------------------------------------------------
# Entry point for the simulator.
# - Original logic preserved.
# - Updated imports/paths to match the new modular structure.
# ------------------------------------------------------------

import os
import sys
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Ensure the project root is on PYTHONPATH when running directly.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from io_utils.save_parameters_and_logs import save_params_and_logs
from config.params import params
from core.main_loop import MainLoop
from core.spatial_risk import (
    build_distance_matrix,
    build_spatial_correlation_matrix,
    extract_pair_correlations,
)


def _build_pair_context():
    """Return action pairs, distances, and static spatial-risk correlations."""
    action_pairs = list(combinations(range(1, params.serverNo + 1), 2))
    expected_actions = params.serverNo * (params.serverNo - 1) // 2
    if len(action_pairs) != expected_actions:
        raise ValueError("action-pair enumeration has an invalid length")
    if not params.SPATIAL_RISK_ENABLED:
        zeros = np.zeros(expected_actions, dtype=float)
        return action_pairs, zeros.copy(), zeros

    server_info_path = os.path.join(PROJECT_ROOT, "data", "server_info.xlsx")
    if not os.path.exists(server_info_path):
        raise FileNotFoundError(f"server topology file not found: {server_info_path}")
    server_df = pd.read_excel(server_info_path)
    required_columns = {"Server_ID", "Latitude", "Longitude"}
    missing = sorted(required_columns.difference(server_df.columns))
    if missing:
        raise ValueError(
            "server_info.xlsx is missing required topology columns: "
            + ", ".join(missing)
        )
    server_df = server_df.copy()
    server_df["Server_ID"] = server_df["Server_ID"].astype(int)
    if server_df["Server_ID"].duplicated().any():
        raise ValueError("server_info.xlsx contains duplicate Server_ID values")
    if len(server_df) != params.serverNo:
        raise ValueError(
            f"server_info.xlsx has {len(server_df)} servers; expected {params.serverNo}"
        )
    servers = [
        SimpleNamespace(
            server_id=int(row["Server_ID"]),
            latitude=float(row["Latitude"]),
            longitude=float(row["Longitude"]),
        )
        for row in server_df.to_dict(orient="records")
    ]
    server_ids, distance_matrix = build_distance_matrix(servers)
    correlation_matrix = build_spatial_correlation_matrix(
        distance_matrix, params.SPATIAL_CORRELATION_LENGTH_KM
    )
    pair_correlations = extract_pair_correlations(
        server_ids, correlation_matrix, action_pairs
    )
    id_to_index = {server_id: index for index, server_id in enumerate(server_ids)}
    pair_distances = np.asarray(
        [
            distance_matrix[id_to_index[server_j], id_to_index[server_k]]
            for server_j, server_k in action_pairs
        ],
        dtype=float,
    )
    return action_pairs, pair_distances, pair_correlations


def build_pair_correlations():
    """Build static pair correlations in the formal action order."""
    action_pairs, _, pair_correlations = _build_pair_context()
    return action_pairs, pair_correlations


def _print_pair_actor_audit(action_pairs, pair_distances, pair_correlations):
    correlations = np.asarray(pair_correlations, dtype=float)
    print(
        "PPO actor audit: mode=pair_scoring, "
        f"state_dim={params.num_states}, num_servers={params.serverNo}, "
        f"num_actions={params.num_actions}, pair_feature_dim=12, "
        "scorer=12->64->32->1"
    )
    print(
        "pair correlation audit: "
        f"length={len(correlations)}, min={correlations.min():.8f}, "
        f"max={correlations.max():.8f}, mean={correlations.mean():.8f}, "
        f"median={np.median(correlations):.8f}"
    )
    print("action_index | server_j | server_k | distance_km | rho_jk")
    for action_index, ((server_j, server_k), distance, rho) in enumerate(
        zip(action_pairs, pair_distances, correlations)
    ):
        print(
            f"{action_index:12d} | {server_j:8d} | {server_k:8d} | "
            f"{distance:11.8f} | {rho:.8f}"
        )


def build_model():
    """Build and return the model/agent object based on params.model_summary.

    NOTE:
      - Replay buffer (if any) is created INSIDE the model (DDPG/DQN).
      - MainLoop only relies on the agent "contract":
          * DQN/PPO: select_action(state, epsilon) -> int
          * DDPG   : policy(state) -> score vector (len=num_actions)
    """
    model_name = str(params.model_summary).strip().lower()

    if model_name == "ddpg":
        from agents.ddpg_agent import ddpgModel

        model = ddpgModel(
            params.num_states,
            params.num_actions,
            params.std_dev_ddpg,
            params.critic_lr_ddpg,
            params.actor_lr_ddpg,
            params.gamma_ddpg,
            params.tau_ddpg,
            params.activation_function_ddpg,
            buffer_capacity=params.buffer_capacity_ddpg,
            batch_size=params.batch_size_ddpg,
        )
        print("DDPGAgent is set.")
        return model

    if model_name == "dqn":
        from agents.dqn_agent import DQNAgent

        model = DQNAgent(
            num_states=params.num_states,
            num_actions=params.num_actions,
            hidden_layers=params.hidden_layers_dqn,
            device="cpu",
            gamma=params.gamma_dqn,
            lr=params.lr_dqn,
            tau=params.tau_dqn,
            buffer_size=params.buffer_capacity_dqn,
            batch_size=params.batch_size_dqn,
            activation=params.af_dqn,
        )
        print("DQNAgent is set.")
        return model

    if model_name == "ppo":
        from agents.ppo_agent import PPOAgent

        actor_mode = str(params.PPO_ACTOR_MODE).strip().lower()
        if actor_mode not in {"flat", "pair_scoring"}:
            raise ValueError(
                "PPO_ACTOR_MODE must be either 'flat' or 'pair_scoring', "
                f"got {params.PPO_ACTOR_MODE!r}"
            )
        actor_kwargs = {"actor_mode": actor_mode}
        if actor_mode == "pair_scoring":
            action_pairs, pair_distances, pair_correlations = _build_pair_context()
            if len(action_pairs) != params.num_actions:
                raise ValueError("pair correlation context does not match num_actions")
            _print_pair_actor_audit(action_pairs, pair_distances, pair_correlations)
            actor_kwargs.update(
                num_servers=params.serverNo,
                pair_correlations=pair_correlations,
            )

        model = PPOAgent(
            num_states=params.num_states,
            num_actions=params.num_actions,
            hidden_layers=params.hidden_layers_ppo,
            device="cpu",
            gamma=params.gamma_ppo,
            actor_lr=params.actor_lr_ppo,
            critic_lr=params.critic_lr_ppo,
            clip_eps=params.clip_eps_ppo,
            k_epochs=params.k_epochs_ppo,
            batch_size=params.batch_size_ppo,
            entropy_coef=params.entropy_coef_ppo,
            reward_scale=params.reward_scale_ppo,
            gae_lambda=params.gae_lambda_ppo,
            value_loss_coef=params.value_loss_coef_ppo,
            max_grad_norm=params.max_grad_norm_ppo,
            activation=params.af_ppo,
            minibatch_seed=params.PPO_MINIBATCH_SEED,
            **actor_kwargs,
        )
        print(f"PPOAgent is set (actor_mode={actor_mode}).")
        return model

    raise ValueError(
        f"Unsupported params.model_summary='{params.model_summary}'. "
        "Use one of: 'ddpg', 'dqn', 'ppo'."
    )


def run_simulation():
    model = build_model()

    ml = MainLoop(model, params.total_episodes, params.taskno, params.num_states, params.num_actions)
    ml.EP()

    save_params_and_logs(
        params,
        ml.log_data,
        ml.task_Assignments_info,
        ml.episode_spatial_risk_log,
        ml.replica_completion_log,
    )


def main():
    run_simulation()


if __name__ == "__main__":
    main()
