"""Runtime parameters derived from :mod:`config.configuration`."""

from config.configuration import parameters

class params:

    # Experiment
    model_summary = parameters.model_summary  # Options: "dqn", "ppo", "ddpg"
    total_episodes = parameters.total_episodes

    # Infrastructure
    NUM_SERVERS = parameters.NUM_SERVERS

    # Workload
    TASK_SIZE_RANGE = parameters.TASK_SIZE_RANGE
    Low_demand, High_demand = parameters.Low_demand, parameters.High_demand
    taskno = parameters.taskno
    TASK_ARRIVAL_RATE = parameters.TASK_ARRIVAL_RATE

    # Unified server parameters
    SERVER_PROCESSING_FREQ_RANGE = parameters.SERVER_PROCESSING_FREQ_RANGE
    SERVER_FAILURE_RATE_RANGE = parameters.SERVER_FAILURE_RATE_RANGE

    BACKLOG_TIME_SCALE_SEC = parameters.BACKLOG_TIME_SCALE_SEC
    SPATIAL_RISK_ENABLED = parameters.SPATIAL_RISK_ENABLED
    SPATIAL_CORRELATION_LENGTH_KM = parameters.SPATIAL_CORRELATION_LENGTH_KM
    SPATIAL_RISK_BETA_P = parameters.SPATIAL_RISK_BETA_P
    MASTER_SEED = parameters.MASTER_SEED

    # New state and action dimensions; runtime state/action construction follows later.
    num_states = 4 * NUM_SERVERS + 3
    num_actions = NUM_SERVERS * (NUM_SERVERS - 1) // 2
    
    # ----------- DDPG ----------------
    std_dev_ddpg = parameters.std_dev_ddpg
    critic_lr_ddpg = parameters.critic_lr_ddpg
    actor_lr_ddpg = parameters.actor_lr_ddpg
    gamma_ddpg = parameters.gamma_ddpg
    tau_ddpg = parameters.tau_ddpg
    buffer_capacity_ddpg = parameters.buffer_capacity_ddpg
    batch_size_ddpg = parameters.batch_size_ddpg
    activation_function_ddpg =parameters.activation_function_ddpg
    # ------------- DQN --------------
    hidden_layers_dqn = parameters.hidden_layers_dqn
    af_dqn = parameters.af_dqn
    lr_dqn = parameters.lr_dqn
    gamma_dqn = parameters.gamma_dqn
    tau_dqn = parameters.tau_dqn
    buffer_capacity_dqn = parameters.buffer_capacity_dqn
    batch_size_dqn = parameters.batch_size_dqn
    epsilon_start_dqn = parameters.epsilon_start_dqn
    epsilon_end_dqn = parameters.epsilon_end_dqn
    epsilon_decay_dqn = parameters.epsilon_decay_dqn
    # ----------- PPO --------------
    hidden_layers_ppo = parameters.hidden_layers_ppo
    af_ppo = parameters.af_ppo
    actor_lr_ppo = parameters.actor_lr_ppo
    critic_lr_ppo = parameters.critic_lr_ppo
    # Per-second discount base for event-driven PPO/SMDP.
    gamma_ppo = parameters.gamma_ppo
    clip_eps_ppo = parameters.clip_eps_ppo
    k_epochs_ppo = parameters.k_epochs_ppo
    batch_size_ppo = parameters.batch_size_ppo
    entropy_coef_ppo = parameters.entropy_coef_ppo
    reward_scale_ppo = parameters.reward_scale_ppo
    gae_lambda_ppo = parameters.gae_lambda_ppo
    value_loss_coef_ppo = parameters.value_loss_coef_ppo
    max_grad_norm_ppo = parameters.max_grad_norm_ppo
