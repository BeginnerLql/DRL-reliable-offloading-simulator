#configuration.py

class parameters:
    # ======================================================================
    # experiment setting
    # ======================================================================
    model_summary = "ppo" # Options: "dqn", "ppo","ddpg"  
    total_episodes = 5  # 100

    # ======================================================================
    # Infrastructure: servers
    # ======================================================================
    # The formal simulator now models eight homogeneous Edge servers.
    NUM_EDGE_SERVERS = 8
    NUM_CLOUD_SERVERS = 0
    serverNo = NUM_EDGE_SERVERS + NUM_CLOUD_SERVERS  # 8
    # ======================================================================
    # Workload: tasks
    # ======================================================================
    TASK_ARRIVAL_RATE = 0.5  # Poisson task-arrival rate, unit: tasks/s
    TASK_SIZE_RANGE = (10, 100)  # heter
    Low_demand, High_demand = 1, 100 # MI (Normal(mean=50, std=16) implied)
    taskno = 200
    
    # ======================================================================
    # Network model
    # ======================================================================
    # RSU-to-cloud backhaul bandwidth (Mb/s).
    rsu_to_cloud_bandwidth = 8  # Mb/s

    # ======================================================================
    # Reliability model parameters for the homogeneous Edge infrastructure.
    # ======================================================================
    FIXED_EDGE_PROCESSING_FREQUENCIES = (10, 11, 12, 14, 15, 17, 18, 20)  # MIPS
    LAMBDA_REF = 0.003  # 1/s, normal-environment reference rate
    FAILURE_RATE_OMEGA = 0.9
    FAILURE_RATE_FMIN = 10.0
    FAILURE_RATE_FMAX = 20.0
    # Compatibility ranges for legacy observation normalization and tools.
    EDGE_PROCESSING_FREQ_RANGE = (10, 20)
    CLOUD_PROCESSING_FREQ_RANGE = (10, 20)
    EDGE_FAILURE_RATE_RANGE = (
        LAMBDA_REF * 10 ** FAILURE_RATE_OMEGA,
        LAMBDA_REF,
    )
    CLOUD_FAILURE_RATE_RANGE = EDGE_FAILURE_RATE_RANGE
    # Fixed backlog-time normalization scale, unit: seconds.
    # Approximately one representative Edge task service time.
    BACKLOG_TIME_SCALE_SEC = 4.0

    # Episode-level quasi-static physical spatial risk field.
    # Episode-level quasi-static spatial risk field used by runtime reliability.
    SPATIAL_RISK_ENABLED = True
    SPATIAL_CORRELATION_LENGTH_KM = 0.5
    SPATIAL_RISK_BETA_P = 0.8
    SPATIAL_RISK_SEED = 2026
    # Independent random streams for environment arrivals and PPO minibatches.
    TASK_ARRIVAL_SEED = 2027
    PPO_MINIBATCH_SEED = 2028
    # Weight applied to the logarithmic failure-budget violation penalty.
    RELIABILITY_VIOLATION_WEIGHT = 10.0

    # ======================================================================
    # RL hyperparameters
    # ======================================================================
    # ----------- DDPG ----------------
    std_dev_ddpg = 0.25
    critic_lr_ddpg = 0.001   #0.002  
    actor_lr_ddpg = 0.0003 #  0.0005  # 0.0005
    gamma_ddpg = 0.85 # 0.9
    tau_ddpg = 0.005 #0.01
    buffer_capacity_ddpg = 100000 #50000
    batch_size_ddpg = 256 #64
    activation_function_ddpg ="softmax" 
    # ------------- DQN --------------
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
    # ----------- PPO --------------
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
