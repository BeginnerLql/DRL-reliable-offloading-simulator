# mainLoop.py  (multi-model: DDPG / DQN / PPO)
# - MainLoop signature simplified: no external buffer
# - DDPG uses model.policy() -> continuous scores -> argmax -> (X,Y,Z), and trains via model.buffer
# - DQN/PPO use model.select_action(state, epsilon) -> discrete action index -> (X,Y,Z)
# - PPO trains once at end of episode; DQN trains online

from core.server import Server
from core.task import Task
from core.env_state import EnvironmentState
from config.params import params
from config.paths import DATA_DIR
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
        self.G_action = None  # DDPG: list of floats | DQN/PPO: int action index
        self.index_of_actions = self.generate_combinations()

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
        self.log_data = []
        self.task_Assignments_info = []
        self.episode_spatial_risk_log = []

        # PPO-only arrival-driven SMDP bookkeeping.
        self.ppo_interval_reward = 0.0
        self.ppo_last_decision_state = None
        self.ppo_last_decision_action = None
        self.ppo_last_decision_time = None
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
            self.ppo_interval_reward = 0.0
            self.ppo_last_decision_state = None
            self.ppo_last_decision_action = None
            self.ppo_last_decision_time = None
            self.ppo_last_resolved_outcome_time = None

            self.env = simpy.Environment()
            self.env_state = EnvironmentState()
            self.env_state.reset()

            self.setServers()
            self._initialize_episode_spatial_risk()
            self.env.process(self.Iteration())
            self.env.run()


    def _initialize_episode_spatial_risk(self):
        """Sample and store one quasi-static spatial risk field for this episode."""
        if not params.SPATIAL_RISK_ENABLED:
            return
        if params.SPATIAL_RISK_BETA_P is None:
            raise ValueError(
                "beta_p must be explicitly configured when spatial risk is enabled."
            )

        server_objects = [
            server_info["server_object"]
            for server_info in self.env_state.servers.values()
        ]
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
        base_failure_rates = np.array([
            self.env_state.get_server_by_id(server_id).failure_rate
            for server_id in server_ids
        ], dtype=float)
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

        for index, server_id in enumerate(
            self.env_state.spatial_risk_server_ids
        ):
            base_failure_rate = float(
                self.env_state.get_server_by_id(server_id).failure_rate
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
                hazard_multiplier = 1.0
            else:
                hazard_multiplier = effective_failure_rate / base_failure_rate
            if not np.isfinite(hazard_multiplier):
                raise RuntimeError(
                    f"hazard multiplier is non-finite for server_id {server_id}"
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
                "z_phy": float(self.env_state.spatial_risk_field[index]),
                "hazard_multiplier": float(hazard_multiplier),
                "effective_failure_rate": effective_failure_rate,
            })


    def _sample_interarrival_time(self):
        """Sample one inter-arrival time for the common Poisson workload."""
        arrival_rate = float(params.TASK_ARRIVAL_RATE)
        if arrival_rate <= 0:
            raise ValueError(
                "TASK_ARRIVAL_RATE must be positive and measured in tasks/s"
            )
        return float(
            np.random.exponential(scale=1.0 / arrival_rate)
        )

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
                        self.ppo_interval_reward,
                        self.G_state,
                        delta_t=max(float(delta_t), 0.0),
                        done=False,
                    )
                    self.ppo_interval_reward = 0.0
            elif self.taskCounter > 1:
                # Complete s' for the previous transition and train on any
                # resolved tasks using the legacy DQN/DDPG path.
                prev = list(self.tempbuffer[self.taskCounter - 1])
                prev[3] = self.G_state
                self.tempbuffer[self.taskCounter - 1] = tuple(prev)
                self.add_train()

            # -------- action selection --------
            if self.model_name == "ddpg":
                # DDPG outputs continuous scores over actions.
                action_scores = self.model.policy(self.G_state)
                self.G_action = action_scores.numpy().tolist()
                self.G_action = self.model.addNoise(
                    self.G_action, self.this_episode, self.total_episodes
                )
                X, Y, Z = self.extract_parameters_from_action(self.G_action)
            else:
                # DQN/PPO output a discrete action index.
                eps = self.get_epsilon(self.this_episode)
                action_index = self.model.select_action(self.G_state, eps)
                self.G_action = int(action_index)
                X, Y, Z = self.extract_parameters_from_index(self.G_action)

            if self.model_name == "ppo":
                self.ppo_last_decision_state = self.G_state
                self.ppo_last_decision_action = self.G_action
                self.ppo_last_decision_time = current_time
            else:
                # Store the legacy task-centric transition for DQN/DDPG.
                self.tempbuffer[self.taskCounter] = (self.G_state, self.G_action, None, [])

            self.env.process(task.execute_task(X, Y, Z))
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
                    self.ppo_interval_reward,
                    terminal_next_state,
                    delta_t=max(float(delta_t), 0.0),
                    done=True,
                )
                self.ppo_interval_reward = 0.0
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

    def _finalize_resolved_task(self, task_counter, reward, delay):
        """Record common episode metrics and remove one resolved task."""
        task = self.env_state.get_task_by_id(task_counter)
        self.episodic_reward += reward
        self.episodic_delay += delay
        self.rewardsAll.append(reward)
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
                task.z,
            )
        )
        self.pendingList.remove(task_counter)
        self.env_state.remove_task(task_counter)

    def _get_task_outcome_time(self, task):
        """Return the timestamp when ``task`` became finally resolved."""
        primary_stat = task.primaryStat
        backup_stat = task.backupStat
        primary_finished = task.primaryFinished
        backup_finished = task.backupFinished

        if task.z == 0:
            if (
                primary_stat == "success"
                and backup_stat is None
                and primary_finished is not None
            ):
                return float(primary_finished)
            if (
                primary_stat == "failure"
                and backup_stat == "success"
                and backup_finished is not None
            ):
                return float(backup_finished)
            if (
                primary_stat == "failure"
                and backup_stat == "failure"
                and backup_finished is not None
            ):
                return float(backup_finished)
            return None

        # Parallel first-result mode: one successful replica resolves the
        # task immediately, while two failures require both timestamps.
        if (
            primary_stat == "success"
            and backup_stat == "success"
            and primary_finished is not None
            and backup_finished is not None
        ):
            return float(min(primary_finished, backup_finished))
        if (
            primary_stat == "success"
            and backup_stat == "failure"
            and primary_finished is not None
        ):
            return float(primary_finished)
        if (
            primary_stat == "failure"
            and backup_stat == "success"
            and backup_finished is not None
        ):
            return float(backup_finished)
        if (
            primary_stat == "failure"
            and backup_stat == "failure"
            and primary_finished is not None
            and backup_finished is not None
        ):
            return float(max(primary_finished, backup_finished))
        if (
            primary_stat == "success"
            and backup_stat is None
            and primary_finished is not None
        ):
            return float(primary_finished)
        if (
            primary_stat is None
            and backup_stat == "success"
            and backup_finished is not None
        ):
            return float(backup_finished)
        return None

    def _get_ppo_terminal_time(self):
        """Use the latest actual outcome timestamp for PPO terminal timing."""
        decision_time = float(self.ppo_last_decision_time)
        if self.ppo_last_resolved_outcome_time is None:
            return decision_time
        return max(decision_time, float(self.ppo_last_resolved_outcome_time))

    def _collect_resolved_task_outcomes(self):
        """Accumulate completed task rewards into the current PPO interval."""
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

            self.ppo_interval_reward += task_reward
            if self.ppo_last_resolved_outcome_time is None:
                self.ppo_last_resolved_outcome_time = task_outcome_time
            else:
                self.ppo_last_resolved_outcome_time = max(
                    self.ppo_last_resolved_outcome_time, task_outcome_time
                )
            self._finalize_resolved_task(task_counter, task_reward, delay)

    # ---------------------------
    # REWARD CALCULATION (unchanged)
    # ---------------------------
    def calcReward(self, taskID):
        """Reward completed replica executions and task-level outcomes.

        A ``failure`` status means the corresponding replica execution was
        defeated by a transient fault. It does not mean that a server is down.
        A final failure means that all required replicas failed.
        """
        task = self.env_state.get_task_by_id(taskID)
        z = task.z
        primaryStat = task.primaryStat
        backupStat = task.backupStat
        primaryFinished = task.primaryFinished
        primaryStarted = task.primaryStarted
        backupFinished = task.backupFinished
        backupStarted = task.backupStarted

        flag = "s"
        delay = None

        if z == 0:
            if primaryStat == 'success' and backupStat is None and primaryFinished is not None:
                delay = primaryFinished - primaryStarted
            elif primaryStat == 'failure' and backupStat == 'success' and backupFinished is not None:
                delay = backupFinished - primaryStarted
            elif primaryStat == 'failure' and backupStat == 'failure':
                delay = backupFinished - primaryStarted
                flag = "f"
            else:
                flag = "n"
        else:
            if primaryStat == 'success' and backupStat == 'success' and primaryFinished is not None and backupFinished is not None:
                delay = min(primaryFinished, backupFinished) - primaryStarted
            elif primaryStat == 'success' and backupStat == 'failure' and primaryFinished is not None:
                delay = primaryFinished - primaryStarted
            elif primaryStat == 'failure' and backupStat == 'success' and backupFinished is not None:
                delay = backupFinished - backupStarted
            elif primaryStat == 'failure' and backupStat == 'failure':
                delay = max(backupFinished - backupStarted, primaryFinished - primaryStarted)
                flag = "f"
            elif primaryStat == 'success' and backupStat is None and primaryFinished is not None:
                delay = primaryFinished - primaryStarted
            elif primaryStat is None and backupStat == 'success' and backupFinished is not None:
                delay = backupFinished - backupStarted
            else:
                flag = "n"

        if flag == "f":
            failure_penalty_weight = 3.0
            reward = -failure_penalty_weight * delay
            if reward > -3:
                reward = -3
        elif flag == "s":
            success_reward_weight = 1.0
            reward = success_reward_weight * (math.log(1 - (1 / math.exp(math.sqrt(delay)))) / math.log(0.995))
        else:
            reward = None

        return reward, delay

    # ---------------------------
    # TRAINING (multi-model)
    # ---------------------------
    def add_train(self):
        if self.model_name == "ppo":
            self._collect_resolved_task_outcomes()
            return

        removeList = []

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
                    task.z,
                )
            )
            self.env_state.remove_task(t)

    # ---------------------------
    # SERVERS 
    # ---------------------------
    def setServers(self):
        excel_file = os.path.join(DATA_DIR, "server_info.xlsx")
        server_info_df = pd.read_excel(excel_file)
        required_columns = {
            "Server_ID",
            "Server_Type",
            "Processing_Frequency",
            "Failure_Rate",
            "Latitude",
            "Longitude",
        }
        missing_columns = sorted(required_columns.difference(server_info_df.columns))
        if missing_columns:
            raise ValueError(
                "server_info.xlsx is missing required columns: "
                + ", ".join(missing_columns)
            )

        for _, row in server_info_df.iterrows():
            server_id = int(row["Server_ID"])
            server_type = str(row["Server_Type"])
            processing_frequency = float(row["Processing_Frequency"])
            failure_rate = float(row["Failure_Rate"])
            latitude = float(row["Latitude"])
            longitude = float(row["Longitude"])

            server = Server(
                self.env,
                server_type,
                server_id,
                processing_frequency,
                failure_rate,
                latitude,
                longitude,
            )
            self.env_state.add_server_and_init_environment(server)

    # ---------------------------
    # ACTION DECODING
    # ---------------------------
    def extract_parameters_from_index(self, action_index: int):
        primary_server_id, backup_server_id, z_parameter = self.index_of_actions[int(action_index)]
        primary_server = self.env_state.get_server_by_id(primary_server_id)
        backup_server = self.env_state.get_server_by_id(backup_server_id)
        return primary_server, backup_server, z_parameter

    def extract_parameters_from_action(self, action_scores_list):
        # DDPG: choose argmax index from continuous scores
        if not action_scores_list:
            raise ValueError("Action scores list is empty")
        max_index = int(action_scores_list.index(max(action_scores_list)))
        return self.extract_parameters_from_index(max_index)

    # ---------------------------
    # ACTION INDEX LIST 
    # ---------------------------
    @staticmethod
    def generate_combinations():
        numberOfServers = params.serverNo
        index_of_actions = []

        # z=0: ordered pairs (including i==j)
        for i in range(1, numberOfServers + 1):
            for j in range(1, numberOfServers + 1):
                index_of_actions.append((i, j, 0))

        # z=1: unique pairs (i<j)
        for i in range(1, numberOfServers + 1):
            for j in range(i + 1, numberOfServers + 1):
                index_of_actions.append((i, j, 1))

        return index_of_actions
