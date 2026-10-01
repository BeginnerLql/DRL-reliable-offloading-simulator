"""Central configuration for simulator parameters."""


class parameters:
    # Experiment
    model_summary = "ppo"  # Options: "dqn", "ppo", "ddpg"
    total_episodes = 100
    EXPERIMENT_TAG = "baseline"

    # Infrastructure
    NUM_SERVERS = 8

    # Workload
    TASK_ARRIVAL_RATE = 0.5  # Poisson task-arrival rate, unit: tasks/s
    TASK_SIZE_RANGE = (10, 100)  # MB
    # Main-experiment task reliability requirement tiers.
    TASK_RELIABILITY_REQUIREMENT_LEVELS = (0.9, 0.99, 0.999, 0.9999)
    Low_demand, High_demand = 1, 100  # MI
    taskno = 200

    SERVER_TRANSMISSION_RATES = (
        20.0,
        24.0,
        28.0,
        32.0,
        36.0,
        40.0,
        45.0,
        50.0,
    )  # MB/s, indexed by Server_ID 1..8

    # Unified server parameters
    # TEMPORARY: value to be calibrated later.
    SERVER_PROCESSING_FREQ_RANGE = (10, 15)  # MIPS
    # TEMPORARY: base transient failure intensity range, unit: s^-1.
    SERVER_FAILURE_RATE_RANGE = (0.001, 0.005)

    # Retained for backlog normalization; the value remains provisional.
    BACKLOG_TIME_SCALE_SEC = 4.0

    # One quasi-static spatial-risk field is sampled per episode when enabled.
    SPATIAL_RISK_ENABLED = True
    SPATIAL_CORRELATION_LENGTH_KM = 0.5
    # TEMPORARY exploratory calibration value.
    # Not yet a literature-calibrated final experiment parameter.
    SPATIAL_RISK_BETA_P = 0.5
    MASTER_SEED = 2026

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
    hidden_layers_ppo = [64, 32]
    af_ppo = "tanh"
    actor_lr_ppo = 1e-4
    critic_lr_ppo = 5e-4
    # Per-second discount base for event-driven PPO/SMDP.
    # Transition-specific discount: gamma_k = gamma_ppo ** delta_t_seconds.
    gamma_ppo = 0.90
    clip_eps_ppo = 0.2
    k_epochs_ppo = 2
    batch_size_ppo = 64
    entropy_coef_ppo = 0.01
    reward_scale_ppo = 1
    gae_lambda_ppo = 0.95
    value_loss_coef_ppo = 0.5
    max_grad_norm_ppo = 0.5
