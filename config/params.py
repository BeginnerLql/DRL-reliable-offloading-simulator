"""config.params

Lightweight 'params' holder used across the simulator.
It mirrors values from config.configuration.parameters.

Important:
- Keep this file mostly as-is to avoid breaking logic.
- Any path-related values are handled in config.paths (DATA_DIR/RESULTS_DIR).
"""

from config.configuration import parameters

class params:

    # experiment setting
    model_summary = parameters.model_summary  # Options: "dqn", "ppo", "ddpg"
    total_episodes = parameters.total_episodes

    # Infrastructure: servers
    NUM_EDGE_SERVERS =parameters.NUM_EDGE_SERVERS
    NUM_CLOUD_SERVERS = parameters.NUM_CLOUD_SERVERS
    serverNo = NUM_EDGE_SERVERS + NUM_CLOUD_SERVERS  

    # Workload: tasks
    TASK_SIZE_RANGE = parameters.TASK_SIZE_RANGE
    Low_demand, High_demand = parameters.Low_demand, parameters.High_demand
    taskno = parameters.taskno
    TASK_ARRIVAL_RATE = parameters.TASK_ARRIVAL_RATE
    # Network model
    rsu_to_cloud_bandwidth = parameters.rsu_to_cloud_bandwidth

    # Server capabilities
    EDGE_PROCESSING_FREQ_RANGE = parameters.EDGE_PROCESSING_FREQ_RANGE
    CLOUD_PROCESSING_FREQ_RANGE = parameters.CLOUD_PROCESSING_FREQ_RANGE
    # Fixed base failure rates (1/s)
    EDGE_FAILURE_RATE_RANGE = parameters.EDGE_FAILURE_RATE_RANGE
    CLOUD_FAILURE_RATE_RANGE = parameters.CLOUD_FAILURE_RATE_RANGE
    BACKLOG_TIME_SCALE_SEC = parameters.BACKLOG_TIME_SCALE_SEC

    # RL hyperparameters
    num_states = 3 * serverNo + 2  # observable failure rate, frequency, backlog time per server + task profile
    num_actions = (serverNo*serverNo)+(serverNo*(serverNo-1))//2 # 92 actions for 8 servers
    
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
    gamma_ppo = parameters.gamma_ppo
    clip_eps_ppo = parameters.clip_eps_ppo
    k_epochs_ppo = parameters.k_epochs_ppo
    batch_size_ppo = parameters.batch_size_ppo
    entropy_coef_ppo = parameters.entropy_coef_ppo
    reward_scale_ppo = parameters.reward_scale_ppo
    gae_lambda_ppo = parameters.gae_lambda_ppo
    value_loss_coef_ppo = parameters.value_loss_coef_ppo
    max_grad_norm_ppo = parameters.max_grad_norm_ppo


    
    
    