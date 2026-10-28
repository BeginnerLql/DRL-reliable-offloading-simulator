"""Central configuration for simulator parameters."""


class parameters:
    # Change experiment settings here before running comparisons; do not override
    # them in runner scripts. Regenerate workbooks after changing profile ranges.
    # Experiment
    model_summary = "ppo"  # Options: "dqn", "ppo", "ddpg"
    total_episodes = 100
    EXPERIMENT_TAG = "baseline"

    # Infrastructure
    NUM_SERVERS = 8

    # Workload
    TASK_ARRIVAL_RATE = 0.5  # Poisson task-arrival rate, unit: tasks/s
    TASK_SIZE_RANGE = (5.0, 10.0)  # MB, continuous uniform
    TASK_RELIABILITY_REQUIREMENT_RANGE = (0.99, 0.999)  # Continuous uniform.
    TASK_DEMAND_RANGE = (4.0e9, 8.0e9)  # CPU cycles, continuous uniform; 2x original demand.
    taskno = 200

    # Unified server parameters
    SERVER_PROCESSING_FREQ_RANGE = (1.0e9, 4.0e9)  # cycles/s; equivalent to 1–4 GHz
    SERVER_TRANSMISSION_RATE_RANGE = (5.0, 20.0)  # MB/s, continuous uniform
    # TEMPORARY: base transient failure intensity range, unit: s^-1.
    SERVER_FAILURE_RATE_RANGE = (0.01, 0.05)

    # Integrate unknown Gaussian risk using GH; no episode risk field is sampled.
    # False sets pair correlation to zero, retaining the same marginal risk.
    SPATIAL_RISK_ENABLED = True
    # Baseline correlation length = median server-pair distance / ln(2).
    # TEMPORARY exploratory calibration value.
    # Not yet a literature-calibrated final experiment parameter.
    SPATIAL_RISK_BETA_P = 0.5
    SPATIAL_RISK_GH_ORDER = 32  # Quadrature nodes per dimension.
    # SeedSequence derives independent server/train/evaluation profile streams.
    PROFILE_SEED = 2028
    # Runtime seed for task arrivals and agents; GH reliability is deterministic.
    EXPERIMENT_SEED = 2026

    # RL hyperparameters

    # DDPG
    std_dev_ddpg = 0.25
    critic_lr_ddpg = 0.001
    actor_lr_ddpg = 0.0003
    gamma_ddpg = 0.85
    tau_ddpg = 0.005
    buffer_capacity_ddpg = 100000
    batch_size_ddpg = 256
    activation_function_ddpg = "softmax"
    # DQN
    hidden_layers_dqn = [128, 64]
    af_dqn = "relu"
    lr_dqn = 5e-4
    gamma_dqn = 0.90
    tau_dqn = 0.005
    buffer_capacity_dqn = 200_000
    batch_size_dqn = 256
    epsilon_start_dqn = 1.0
    epsilon_end_dqn = 0.01
    epsilon_decay_dqn = 300
    # PPO
    RELIABILITY_ACTION_MASK_ENABLED = True  # PPO only; follows SPATIAL_RISK_ENABLED.
    hidden_layers_ppo = [64, 32]
    af_ppo = "tanh"
    actor_lr_ppo = 1e-4
    critic_lr_ppo = 5e-4
    # Per-second discount base for event-driven PPO/SMDP.
    # Transition-specific discount: gamma_k = gamma_ppo ** delta_t_seconds.
    gamma_ppo = 0.8
    clip_eps_ppo = 0.2
    k_epochs_ppo = 2
    batch_size_ppo = 64
    entropy_coef_ppo = 0.01
    reward_scale_ppo = 1
    gae_lambda_ppo = 0.95
    value_loss_coef_ppo = 0.5
    max_grad_norm_ppo = 0.5
