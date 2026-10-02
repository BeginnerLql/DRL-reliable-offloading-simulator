from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from io_utils.save_parameters_and_logs import save_params_and_logs
from config.params import params
from config.paths import RESULTS_DIR
from core.main_loop import MainLoop


def build_model():
    """Build the configured DQN, PPO or DDPG agent."""
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

        model = PPOAgent(
            num_states=params.num_states,
            num_actions=params.num_actions,
            hidden_layers=params.hidden_layers_ppo,
            device="cuda",
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
        )
        print(f"PPOAgent is set. Device: {model.device}")
        return model

    raise ValueError(
        f"Unsupported params.model_summary='{params.model_summary}'. "
        "Use one of: 'ddpg', 'dqn', 'ppo'."
    )


def run_simulation():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.random.seed(params.EXPERIMENT_SEED)
    torch.manual_seed(params.EXPERIMENT_SEED)
    model = build_model()

    ml = MainLoop(model, params.total_episodes, params.taskno, params.num_states, params.num_actions)
    ml.EP()

    save_params_and_logs(
        params, ml.log_data, ml.task_results,
        run_id=run_id,
    )
    if str(params.model_summary).strip().lower() == "ppo":
        checkpoint_path = Path(RESULTS_DIR) / f"ppo_{run_id}_{params.EXPERIMENT_TAG}.pt"
        model.save_model(checkpoint_path)
        print(f"Saved PPO checkpoint: {checkpoint_path}")


def main():
    run_simulation()

if __name__ == "__main__":
    main()
