# mainLoop.py  (multi-model: DDPG / DQN / PPO)
# - MainLoop signature simplified: no external buffer
# - DDPG uses model.policy() -> continuous scores -> argmax -> pair, and trains via model.buffer
# - DQN/PPO use model.select_action(state, epsilon) -> discrete action index -> pair
# - PPO trains once at end of episode; DQN trains online

from core.server import Server
from core.task import Task
from core.env_state import EnvironmentState
from config.params import params
from config.paths import DATA_DIR
from itertools import combinations
import operator

from core.spatial_risk import (
    build_distance_matrix,
    build_spatial_correlation_matrix,
    map_spatial_risk_to_effective_failure_rates,
    sample_spatial_risk_field,
    validate_correlation_matrix,
)

import simpy
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math


class MainLoop:
    def __init__(self, model, total_episodes, maxtaskno, num_states, num_actions):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.total_episodes = total_episodes

        self.model_name = str(getattr(params, "model_summary", "ddpg")).strip().lower()

        self.rewardsAll = []
        self.ep_reward_list = []
        self.ep_delay_list = []
        self.avg_reward_list = []
        self.this_episode = 0

        self.G_state = []
        self.G_action = None  # DDPG: score vector | DQN/PPO: pair action index
        self.action_pairs = self.generate_combinations()
        self.index_of_actions = self.action_pairs
        if num_actions != len(self.action_pairs):
            raise ValueError(
                f"num_actions must equal the {len(self.action_pairs)} available server pairs"
            )

        self.episodic_reward = 0
        self.episodic_delay = 0

        # Legacy DQN/DDPG buffer: tempbuffer[taskCounter] = (s, a, r, s')
        self.tempbuffer = {}
        self.taskCounter = 1
        self.pendingList = []
        self.maxTask = maxtaskno

        self.env = None
        self.env_state = None
        self.spatial_risk_rng = np.random.default_rng(params.SPATIAL_RISK_SEED)
        self.arrival_rng = np.random.default_rng(params.TASK_ARRIVAL_SEED)
        self.log_data = []
        self.task_Assignments_info = []
        self.episode_spatial_risk_log = []
        self.replica_completion_log = []

        # PPO-only arrival-driven SMDP bookkeeping.
        self.ppo_last_decision_state = None
        self.ppo_last_decision_action = None
        self.ppo_last_decision_time = None
        self.ppo_last_decision_task_id = None
        self.ppo_last_resolved_outcome_time = None


    # ---------------------------
    # EPISODE LOOP
    # ---------------------------
    def EP(self):
        while self.this_episode < self.total_episodes:
            self.this_episode += 1
            self.episodic_reward = 0
            self.episodic_delay = 0
            self.tempbuffer = {}
            self.taskCounter = 1
            self.pendingList = []
            self.ppo_last_decision_state = None
            self.ppo_last_decision_action = None
            self.ppo_last_decision_time = None
            self.ppo_last_decision_task_id = None
            self.ppo_last_resolved_outcome_time = None

            self.env = simpy.Environment()
            self.env_state = EnvironmentState()
            self.env_state.reset()

            self.setServers()
            self._initialize_episode_failure_rates()
            self.env.process(self.Iteration())
            self.env.run()

            # EnvironmentState owns only the current episode's log. Copy each
            # entry into the experiment-level log exactly once after SimPy has
            # drained all replica processes.
            for entry in self.env_state.replica_completion_log:
                entry_copy = dict(entry)
                entry_copy["episode"] = self.this_episode
                self.replica_completion_log.append(entry_copy)


    def _initialize_episode_failure_rates(self):
        """Initialize episode hazards from base rates and one spatial field."""
        server_objects = [
            server_info["server_object"]
            for server_info in self.env_state.servers.values()
        ]
        server_ids = sorted(server.server_id for server in server_objects)
        base_failure_rates = np.array([
            self.env_state.get_server_by_id(server_id).base_failure_rate
            for server_id in server_ids
        ], dtype=float)
        # Base rates already represent lambda_j^0 in the formal model.

        if not params.SPATIAL_RISK_ENABLED:
            self.env_state.set_episode_effective_failure_rates(
                server_ids,
                base_failure_rates,
            )
            return

        if params.SPATIAL_RISK_BETA_P is None:
            raise ValueError(
                "beta_p must be explicitly configured when spatial risk is enabled."
            )
        server_ids, distance_matrix = build_distance_matrix(server_objects)
        correlation_matrix = build_spatial_correlation_matrix(
            distance_matrix,
            params.SPATIAL_CORRELATION_LENGTH_KM,
        )
        validate_correlation_matrix(correlation_matrix)
        spatial_risk_field = sample_spatial_risk_field(
            correlation_matrix,
            rng=self.spatial_risk_rng,
        )
        effective_failure_rates = map_spatial_risk_to_effective_failure_rates(
            base_failure_rates,
            spatial_risk_field,
            params.SPATIAL_RISK_BETA_P,
        )
        self.env_state.set_episode_spatial_risk_context(
            server_ids,
            distance_matrix,
            correlation_matrix,
            spatial_risk_field,
            effective_failure_rates,
        )

        for index, server_id in enumerate(server_ids):
            base_failure_rate = float(
                self.env_state.get_server_by_id(server_id).base_failure_rate
            )
            effective_failure_rate = float(
                self.env_state.effective_failure_rates[server_id]
            )
            if base_failure_rate == 0.0:
                if effective_failure_rate != 0.0:
                    raise RuntimeError(
                        "effective failure rate must remain zero when "
                        f"base failure rate is zero for server_id {server_id}"
                    )
                spatial_hazard_multiplier = 1.0
            else:
                spatial_hazard_multiplier = (
                    effective_failure_rate / base_failure_rate
                )
            if not np.isfinite(spatial_hazard_multiplier):
                raise RuntimeError(
                    f"spatial hazard multiplier is non-finite for server_id {server_id}"
                )
            self.episode_spatial_risk_log.append({
                "episode": self.this_episode,
                "server_id": server_id,
                "spatial_risk_enabled": True,
                "correlation_length_km": float(
                    params.SPATIAL_CORRELATION_LENGTH_KM
                ),
                "beta_p": float(params.SPATIAL_RISK_BETA_P),
                "spatial_risk_seed": params.SPATIAL_RISK_SEED,
                "base_failure_rate": base_failure_rate,
                # Compatibility audit field; formal runtime scaling is 1.
                "failure_rate_scale": 1.0,
                # Historical audit column retained as an alias of the base rate.
                "scaled_base_failure_rate": base_failure_rate,
                "z_phy": float(self.env_state.spatial_risk_field[index]),
                "spatial_hazard_multiplier": float(spatial_hazard_multiplier),
                # Legacy alias: this is the spatial-only multiplier relative to
                # the normal-environment base rate.
                "hazard_multiplier": float(spatial_hazard_multiplier),
                "effective_failure_rate": effective_failure_rate,
            })

    def _initialize_episode_spatial_risk(self):
        """Backward-compatible entry point for episode hazard initialization."""
        return self._initialize_episode_failure_rates()

    def _sample_interarrival_time(self):
        """Sample one inter-arrival time for the common Poisson workload."""
        arrival_rate = float(params.TASK_ARRIVAL_RATE)
        if arrival_rate <= 0:
            raise ValueError(
                "TASK_ARRIVAL_RATE must be positive and measured in tasks/s"
            )
        return float(self.arrival_rng.exponential(scale=1.0 / arrival_rate))

    # ---------------------------
    # epsilon schedule (DQN only; PPO ignores epsilon in its select_action signature)
    # ---------------------------
    def get_epsilon(self, episode):
        if self.model_name != "dqn":
            return 0.0
        eps_start = getattr(params, "epsilon_start_dqn", 1.0)
        eps_end = getattr(params, "epsilon_end_dqn", 0.01)
        eps_decay = getattr(params, "epsilon_decay_dqn", 300)
        # linear decay 
        return max(eps_end, eps_start - (episode / float(eps_decay)))

    # ---------------------------
    # MAIN SIMULATION ITERATION
    # ---------------------------
    def Iteration(self):
        while self.taskCounter <= self.maxTask:
            yield self.env.timeout(self._sample_interarrival_time())
            current_time = float(self.env.now)

            # PPO closes the previous arrival-to-arrival interval before
            # observing the new task. Other algorithms keep their original
            # task-centric bookkeeping path below.
            if self.model_name == "ppo":
                self._collect_resolved_task_outcomes()

            task = Task(self.env, self.env_state, self.taskCounter)
            self.env_state.add_task(task)
            self.G_state = self.env_state.get_state(task)

            if self.model_name == "ppo":
                if self.ppo_last_decision_state is not None:
                    delta_t = current_time - self.ppo_last_decision_time
                    if delta_t < -1e-8:
                        raise ValueError(f"Negative PPO decision interval: {delta_t}")
                    self.model.store_transition(
                        self.ppo_last_decision_state,
                        self.ppo_last_decision_action,
                        None,
                        self.G_state,
                        delta_t=max(float(delta_t), 0.0),
                        done=False,
                        task_id=self.ppo_last_decision_task_id,
                    )
            elif self.taskCounter > 1:
                # Complete s' for the previous transition and train on any
                # resolved tasks using the legacy DQN/DDPG path.
                prev = list(self.tempbuffer[self.taskCounter - 1])
                prev[3] = self.G_state
                self.tempbuffer[self.taskCounter - 1] = tuple(prev)
                self.add_train()

            # -------- pair action selection --------
            if self.model_name == "ddpg":
                # DDPG outputs continuous scores over the shared pair actions.
                action_scores = self.model.policy(self.G_state)
                self.G_action = action_scores.numpy().tolist()
                self.G_action = self.model.addNoise(
                    self.G_action, self.this_episode, self.total_episodes
                )
                action_index = self.action_index_from_scores(self.G_action)
            else:
                # DQN/PPO output a discrete action index over the same pairs.
                eps = self.get_epsilon(self.this_episode)
                action_index = self._validate_action_index(
                    self.model.select_action(self.G_state, eps)
                )
                self.G_action = action_index

            X, Y = self.extract_parameters_from_index(action_index)
            task.action_index = action_index

            # Reliability is evaluated once from the selected pair and the
            # episode-effective hazards, before any replica process starts.
            task.initialize_reliability_evaluation(X, Y)

            if self.model_name == "ppo":
                self.ppo_last_decision_state = self.G_state
                self.ppo_last_decision_action = self.G_action
                self.ppo_last_decision_time = current_time
                self.ppo_last_decision_task_id = task.id
            else:
                # Store the legacy task-centric transition for DQN/DDPG.
                self.tempbuffer[self.taskCounter] = (self.G_state, self.G_action, None, [])

            self.env.process(task.execute_task(X, Y))
            self.pendingList.append(self.taskCounter)
            self.taskCounter += 1

        if self.model_name != "ppo":
            # Preserve the legacy final next-state placeholder for DQN/DDPG.
            if self.taskCounter > 1:
                last = list(self.tempbuffer[self.taskCounter - 1])
                last[3] = self.G_state
                self.tempbuffer[self.taskCounter - 1] = tuple(last)

        # Drain pending tasks by waiting for real task-level resolution events.
        yield from self._drain_pending_tasks()

        if self.model_name == "ppo":
            # Close the final arrival-to-terminal interval explicitly.
            if self.ppo_last_decision_state is not None:
                terminal_next_state = np.zeros_like(self.ppo_last_decision_state)
                terminal_time = self._get_ppo_terminal_time()
                delta_t = terminal_time - self.ppo_last_decision_time
                if delta_t < -1e-8:
                    raise ValueError(f"Negative PPO terminal interval: {delta_t}")
                self.model.store_transition(
                    self.ppo_last_decision_state,
                    self.ppo_last_decision_action,
                    None,
                    terminal_next_state,
                    delta_t=max(float(delta_t), 0.0),
                    done=True,
                    task_id=self.ppo_last_decision_task_id,
                )
            # PPO remains on-policy and updates once after the episode.
            self.model.train_step()

        # episode logs
        self.ep_reward_list.append(self.episodic_reward)
        self.ep_delay_list.append(self.episodic_delay)

        avg_reward = np.mean(self.ep_reward_list[-40:])
        avg_delay = np.mean(self.ep_delay_list[-40:])
        self.log_data.append((self.this_episode, avg_reward, self.episodic_reward, avg_delay))
        self.avg_reward_list.append(avg_reward)

        print(f"Episode {self.this_episode} | Avg Reward: {avg_reward:.3f} | This Episode: {self.episodic_reward:.3f}")

    def _drain_pending_tasks(self):
        """Resolve pending tasks using task-level SimPy events, not polling."""
        while len(self.pendingList) > 0:
            if self.model_name == "ppo":
                self._collect_resolved_task_outcomes()
            else:
                self.add_train()

            if len(self.pendingList) == 0:
                break

            resolution_events = []
            for task_id in self.pendingList:
                task = self.env_state.get_task_by_id(task_id)
                if task is None:
                    raise RuntimeError(
                        f"Pending task {task_id} is missing from the environment state"
                    )
                resolution_event = getattr(task, "resolution_event", None)
                if resolution_event is None:
                    raise RuntimeError(
                        f"Pending task {task_id} has no resolution event"
                    )
                if resolution_event.triggered:
                    raise RuntimeError(
                        f"Pending task {task_id} has a triggered event but no resolved reward"
                    )
                resolution_events.append(resolution_event)

            if not resolution_events:
                raise RuntimeError("No resolution events remain for pending tasks")

            yield self.env.any_of(resolution_events)

    @staticmethod
    def _reliability_diagnostic_metrics(task):
        """Return task-level diagnostic metrics without changing reward logic."""
        requirement = getattr(task, "reliability_requirement", None)
        execution_reliability = getattr(task, "execution_reliability", None)
        if requirement is None or execution_reliability is None:
            return (None, None, None)
        margin = float(execution_reliability) - float(requirement)
        shortfall = max(float(requirement) - float(execution_reliability), 0.0)
        excess = max(float(execution_reliability) - float(requirement), 0.0)
        return (margin, shortfall, excess)

    @staticmethod
    def _calculate_reliability_violation(task):
        """Return log10 failure-budget violation for a resolved task."""
        try:
            requirement = float(task.reliability_requirement)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Reliability_Requirement must be a finite value in (0, 1)") from exc
        if not math.isfinite(requirement) or not 0.0 < requirement < 1.0:
            raise ValueError("Reliability_Requirement must be a finite value in (0, 1)")

        try:
            joint_failure_probability = float(task.joint_failure_probability)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Joint failure probability must be a finite value in [0, 1]") from exc
        if not math.isfinite(joint_failure_probability) or not 0.0 <= joint_failure_probability <= 1.0:
            raise ValueError("Joint failure probability must be a finite value in [0, 1]")
        if joint_failure_probability <= 0.0:
            return 0.0

        failure_budget = 1.0 - requirement
        ratio = joint_failure_probability / failure_budget
        if ratio <= 1.0:
            return 0.0
        return math.log10(ratio)

    def _mark_task_resolution(self, task):
        """Mark reward bookkeeping complete exactly once for a resolved task."""
        recorder = getattr(self.env_state, "record_task_resolution", None)
        if callable(recorder):
            return recorder(task)
        # Compatibility for lightweight test registries and legacy callers.
        if getattr(task, "resolution_bookkeeping_done", False):
            return False
        task.resolution_bookkeeping_done = True
        return True

    def _finalize_resolved_task(self, task_counter, reward, delay):
        """Record metrics, then defer state cleanup until both replicas finish."""
        task = self.env_state.get_task_by_id(task_counter)
        self.episodic_reward += reward
        self.episodic_delay += delay
        self.rewardsAll.append(reward)
        reliability_margin, reliability_shortfall, reliability_excess = (
            self._reliability_diagnostic_metrics(task)
        )
        self.task_Assignments_info.append(
            (
                self.this_episode,
                task.id,
                task.primaryNode.server_id,
                task.primaryStarted,
                task.primaryFinished,
                task.primaryStat,
                task.backupNode.server_id,
                task.backupStarted,
                task.backupFinished,
                task.backupStat,
                None,
                getattr(task, "reliability_requirement", None),
                getattr(task, "primary_effective_failure_rate", None),
                getattr(task, "backup_effective_failure_rate", None),
                getattr(task, "primary_service_time_for_reliability", None),
                getattr(task, "backup_service_time_for_reliability", None),
                getattr(task, "primary_failure_probability", None),
                getattr(task, "backup_failure_probability", None),
                getattr(task, "joint_failure_probability", None),
                getattr(task, "execution_reliability", None),
                getattr(task, "reliability_satisfied", None),
                reward,
                delay,
                reliability_margin,
                reliability_shortfall,
                reliability_excess,
                getattr(task, "base_reward", None),
                getattr(task, "reliability_violation", None),
                getattr(task, "reliability_penalty", None),
                getattr(task, "action_index", None),
                task.primaryNode.server_id,
                task.backupNode.server_id,
            )
        )
        self._mark_task_resolution(task)
        self.pendingList.remove(task_counter)
        try_finalize = getattr(task, "try_finalize_lifecycle", None)
        if callable(try_finalize):
            try_finalize()

    def _get_task_outcome_time(self, task):
        """Return the first actual replica completion timestamp."""
        finish_times = [
            timestamp for timestamp in (task.primaryFinished, task.backupFinished)
            if timestamp is not None
        ]
        return float(min(finish_times)) if finish_times else None

    def _get_ppo_terminal_time(self):
        """Use the latest actual outcome timestamp for PPO terminal timing."""
        decision_time = float(self.ppo_last_decision_time)
        if self.ppo_last_resolved_outcome_time is None:
            return decision_time
        return max(decision_time, float(self.ppo_last_resolved_outcome_time))

    def _collect_resolved_task_outcomes(self):
        """Backfill each resolved task reward into its origin transition."""
        for task_counter in list(self.pendingList):
            task = self.env_state.get_task_by_id(task_counter)
            if task is None:
                raise RuntimeError(
                    f"Pending task {task_counter} is missing from the environment state"
                )
            task_reward, delay = self.calcReward(task_counter)
            resolution_event = getattr(task, "resolution_event", None)
            if task_reward is None:
                if resolution_event is not None and resolution_event.triggered:
                    raise RuntimeError(
                        f"Task {task_counter} resolution event fired before calcReward resolved"
                    )
                continue
            if resolution_event is not None and not resolution_event.triggered:
                raise RuntimeError(
                    f"Task {task_counter} has a reward but its resolution event is not triggered"
                )

            task_outcome_time = self._get_task_outcome_time(task)
            if task_outcome_time is None:
                raise RuntimeError(
                    "Resolved PPO task has no valid outcome timestamp"
                )

            self.model.assign_task_reward(task_counter, task_reward)
            if self.ppo_last_resolved_outcome_time is None:
                self.ppo_last_resolved_outcome_time = task_outcome_time
            else:
                self.ppo_last_resolved_outcome_time = max(
                    self.ppo_last_resolved_outcome_time, task_outcome_time
                )
            self._finalize_resolved_task(task_counter, task_reward, delay)

    # ---------------------------
    # REWARD CALCULATION
    # ---------------------------
    def calcReward(self, taskID):
        """Return final reward and delay for a resolved task once only."""
        task = self.env_state.get_task_by_id(taskID)
        if getattr(task, "resolution_bookkeeping_done", False):
            return None, None
        primary_started = task.primaryStarted
        finish_times = [
            timestamp for timestamp in (task.primaryFinished, task.backupFinished)
            if timestamp is not None
        ]
        if not finish_times or primary_started is None:
            return None, None
        delay = min(finish_times) - primary_started

        if task.reliability_satisfied is None:
            return None, None
        if task.reliability_satisfied:
            success_reward_weight = 1.0
            base_reward = success_reward_weight * (
                math.log(1 - (1 / math.exp(math.sqrt(delay))))
                / math.log(0.995)
            )
        else:
            failure_penalty_weight = 3.0
            base_reward = -failure_penalty_weight * delay
            if base_reward > -3:
                base_reward = -3

        reliability_violation = self._calculate_reliability_violation(task)
        try:
            violation_weight = float(params.RELIABILITY_VIOLATION_WEIGHT)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("RELIABILITY_VIOLATION_WEIGHT must be a finite non-negative number") from exc
        if not math.isfinite(violation_weight) or violation_weight < 0.0:
            raise ValueError("RELIABILITY_VIOLATION_WEIGHT must be a finite non-negative number")
        reliability_penalty = violation_weight * reliability_violation
        reward = base_reward - reliability_penalty

        task.base_reward = base_reward
        task.reliability_violation = reliability_violation
        task.reliability_penalty = reliability_penalty
        return reward, delay

    # ---------------------------
    # TRAINING (multi-model)
    # ---------------------------
    def add_train(self):
        if self.model_name == "ppo":
            self._collect_resolved_task_outcomes()
            return

        removeList = []
        resolved_rewards = {}

        for task_counter in list(self.pendingList):
            reward, delay = self.calcReward(task_counter)
            if reward is None:
                continue

            self.episodic_reward += reward
            self.episodic_delay += delay
            self.rewardsAll.append(reward)

            temp = list(self.tempbuffer[task_counter])
            temp[2] = reward
            self.tempbuffer[task_counter] = tuple(temp)
            resolved_rewards[task_counter] = (reward, delay)

            s, a, r, s_ = self.tempbuffer[task_counter]

            if self.model_name == "ddpg":
                # a is the score-vector (len=num_actions) -> OK for ddpg buffer
                self.model.buffer.record((s, a, r, s_))
                self.model.buffer.learn()
                self.model.update_target(self.model.target_actor.variables, self.model.actor_model.variables)
                self.model.update_target(self.model.target_critic.variables, self.model.critic_model.variables)

            elif self.model_name == "dqn":
                self.model.store_transition((s, int(a), r, s_))
                self.model.train_step()

            removeList.append(task_counter)

        for t in removeList:
            self.pendingList.remove(t)
            task = self.env_state.get_task_by_id(t)
            task_reward, task_delay = resolved_rewards[t]
            reliability_margin, reliability_shortfall, reliability_excess = (
                self._reliability_diagnostic_metrics(task)
            )
            self.task_Assignments_info.append(
                (
                    self.this_episode,
                    task.id,
                    task.primaryNode.server_id,
                    task.primaryStarted,
                    task.primaryFinished,
                    task.primaryStat,
                    task.backupNode.server_id,
                    task.backupStarted,
                    task.backupFinished,
                    task.backupStat,
                    None,
                    getattr(task, "reliability_requirement", None),
                    getattr(task, "primary_effective_failure_rate", None),
                    getattr(task, "backup_effective_failure_rate", None),
                    getattr(task, "primary_service_time_for_reliability", None),
                    getattr(task, "backup_service_time_for_reliability", None),
                    getattr(task, "primary_failure_probability", None),
                    getattr(task, "backup_failure_probability", None),
                    getattr(task, "joint_failure_probability", None),
                    getattr(task, "execution_reliability", None),
                    getattr(task, "reliability_satisfied", None),
                    task_reward,
                    task_delay,
                    reliability_margin,
                    reliability_shortfall,
                    reliability_excess,
                    getattr(task, "base_reward", None),
                    getattr(task, "reliability_violation", None),
                    getattr(task, "reliability_penalty", None),
                    getattr(task, "action_index", None),
                    task.primaryNode.server_id,
                    task.backupNode.server_id,
                )
            )
            self._mark_task_resolution(task)
            try_finalize = getattr(task, "try_finalize_lifecycle", None)
            if callable(try_finalize):
                try_finalize()

    # ---------------------------
    # SERVERS 
    # ---------------------------
    def setServers(self):
        excel_file = os.path.join(DATA_DIR, "server_info.xlsx")
        server_info_df = pd.read_excel(excel_file)
        required_columns = {
            "Server_ID",
            "Processing_Frequency",
            "Uplink_Rate",
            "Latitude",
            "Longitude",
        }
        missing_columns = sorted(required_columns.difference(server_info_df.columns))
        if missing_columns:
            raise ValueError(
                "server_info.xlsx is missing required columns: "
                + ", ".join(missing_columns)
            )
        if "Base_Failure_Rate" in server_info_df.columns:
            rate_column = "Base_Failure_Rate"
        elif "Failure_Rate" in server_info_df.columns:
            # Read-only compatibility for older temporary fixtures; formal
            # server_info.xlsx uses Base_Failure_Rate.
            rate_column = "Failure_Rate"
        else:
            raise ValueError(
                "server_info.xlsx is missing required columns: Base_Failure_Rate"
            )

        for _, row in server_info_df.iterrows():
            server_id = int(row["Server_ID"])
            processing_frequency = float(row["Processing_Frequency"])
            base_failure_rate = float(row[rate_column])
            uplink_rate_mbps = float(row["Uplink_Rate"])
            latitude = float(row["Latitude"])
            longitude = float(row["Longitude"])

            # Server_Type is retained only as a compatibility column for the
            # unfinished communication/reporting code. All formal nodes are Edge.
            server = Server(
                self.env,
                "Edge",
                server_id,
                processing_frequency,
                base_failure_rate,
                latitude,
                longitude,
                uplink_rate_mbps,
            )
            self.env_state.add_server_and_init_environment(server)

    # ---------------------------
    # ACTION DECODING
    # ---------------------------
    def _validate_action_index(self, action_index):
        if isinstance(action_index, (bool, np.bool_)):
            raise ValueError("action_index must be an integer")
        try:
            index = operator.index(action_index)
        except TypeError as exc:
            raise ValueError("action_index must be an integer") from exc
        if not 0 <= index < len(self.action_pairs):
            raise IndexError(
                f"action_index {action_index} is out of range [0, {len(self.action_pairs) - 1}]"
            )
        return index

    def action_index_from_scores(self, action_scores):
        """Return the argmax index from a DDPG score vector."""
        scores = np.asarray(action_scores, dtype=float).reshape(-1)
        if scores.size != len(self.action_pairs):
            raise ValueError(
                f"action score vector must have length {len(self.action_pairs)}"
            )
        if not np.isfinite(scores).all():
            raise ValueError("action score vector must contain finite values")
        return int(np.argmax(scores))

    def extract_parameters_from_index(self, action_index: int):
        index = self._validate_action_index(action_index)
        server_j_id, server_k_id = self.action_pairs[index]
        server_j = self.env_state.get_server_by_id(server_j_id)
        server_k = self.env_state.get_server_by_id(server_k_id)
        if server_j is None or server_k is None:
            raise RuntimeError(f"Action {index} references an unknown server pair")
        return server_j, server_k

    def extract_parameters_from_action(self, action_scores_list):
        """Decode a DDPG score vector into its selected distinct pair."""
        return self.extract_parameters_from_index(
            self.action_index_from_scores(action_scores_list)
        )

    # ---------------------------
    # ACTION INDEX LIST
    # ---------------------------
    @staticmethod
    def generate_combinations():
        """Return deterministic unordered pairs of distinct server IDs."""
        return list(combinations(range(1, params.serverNo + 1), 2))
