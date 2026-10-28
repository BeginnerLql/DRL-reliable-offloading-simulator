from itertools import combinations

from core.server import Server
from core.task import Task
from core.env_state import EnvironmentState
from config.params import params
from config.paths import SERVER_INFO_PATH, TRAIN_TASK_PATH
from pathlib import Path
from core.spatial_risk import (
    GaussHermiteReliability,
    build_distance_matrix,
    build_spatial_correlation_matrix,
    compute_baseline_correlation_length_km,
)

import simpy
import numpy as np
import pandas as pd


class MainLoop:
    def __init__(
        self, model, total_episodes, maxtaskno, num_states, num_actions,
        *, task_parameters_path=TRAIN_TASK_PATH,
    ):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.total_episodes = total_episodes
        self.task_parameters_path = Path(task_parameters_path)
        self.server_profiles, self.task_profiles = self._load_run_profiles()

        self.model_name = str(params.model_summary).strip().lower()

        self.this_episode = 0

        self.G_state = []
        self.G_action = None  # DDPG: list of floats | DQN/PPO: int action index
        self.index_of_actions = self.generate_combinations()

        self.episodic_reward = 0
        self.episodic_delay = 0

        # DQN/DDPG task transitions: tempbuffer[taskCounter] = (s, a, r, s')
        self.tempbuffer = {}
        self.taskCounter = 1
        self.pendingList = []
        self.maxTask = maxtaskno

        self.env = None
        self.env_state = None
        self.arrival_rng = np.random.default_rng(params.EXPERIMENT_SEED)
        self.reliability_integrator = GaussHermiteReliability(
            params.SPATIAL_RISK_BETA_P, params.SPATIAL_RISK_GH_ORDER,
        )
        self.pair_risk_correlations = None
        self.pair_reliability_cache = {}  # Fixed profiles and CPU-only exposures.
        self.log_data = []
        self.task_results = []

        # PPO-only arrival-driven SMDP bookkeeping.
        self.ppo_last_decision_state = None
        self.ppo_last_decision_action = None
        self.ppo_last_decision_time = None
        self.ppo_last_decision_action_mask = None
        self.ppo_last_resolved_outcome_time = None

    def _load_run_profiles(self):
        """Load fixed profiles once; every episode reuses their attributes/order."""
        server_df = pd.read_excel(
            SERVER_INFO_PATH,
            sheet_name="Servers",
            dtype={"Site_ID": "string"},
        )
        task_df = pd.read_excel(self.task_parameters_path)
        return (
            self._build_server_profiles(server_df),
            self._build_task_profiles(task_df),
        )

    @staticmethod
    def _build_server_profiles(server_df):
        frame = server_df.sort_values("Server_ID").reset_index(drop=True)
        return {
            int(row["Server_ID"]): {
                "Server_ID": int(row["Server_ID"]),
                "Site_ID": str(row["Site_ID"]),
                "Processing_Frequency": float(row["Processing_Frequency"]),
                "Base_Failure_Rate": float(row["Base_Failure_Rate"]),
                "Transmission_Rate": float(row["Transmission_Rate"]),
                "Latitude": float(row["Latitude"]),
                "Longitude": float(row["Longitude"]),
            }
            for _, row in frame.iterrows()
        }

    @staticmethod
    def _build_task_profiles(task_df):
        frame = task_df.sort_values("Task_ID").reset_index(drop=True)
        return {
            int(row["Task_ID"]): {
                "Task_ID": int(row["Task_ID"]),
                "Task_Size": float(row["Task_Size"]),
                "Computation_Demand": float(row["Computation_Demand"]),
                "Reliability_Requirement": float(row["Reliability_Requirement"]),
            }
            for _, row in frame.iterrows()
        }

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
            self.ppo_last_decision_action_mask = None
            self.ppo_last_resolved_outcome_time = None

            self.env = simpy.Environment()
            self.env_state = EnvironmentState()

            self.setServers()
            if self.pair_risk_correlations is None:
                self._initialize_spatial_correlations()
            self.env.process(self.Iteration())
            self.env.run()
            for task in sorted(self.env_state.tasks.values(), key=lambda item: item.id):
                self.task_results.append(self.get_task_outcome_info(task))

    def _initialize_spatial_correlations(self):
        """Build fixed pair correlations once per run, without sampling risk."""
        server_objects = [
            server_info["server_object"]
            for server_info in self.env_state.servers.values()
        ]
        server_ids, distance_matrix = build_distance_matrix(server_objects)
        if params.SPATIAL_RISK_ENABLED:
            correlation_length_km = compute_baseline_correlation_length_km(distance_matrix)
            correlation_matrix = build_spatial_correlation_matrix(
                distance_matrix, correlation_length_km,
            )
        else:
            correlation_matrix = np.eye(len(server_ids))
        self.pair_risk_correlations = {
            (server_a, server_b): float(correlation_matrix[row, column])
            for row, server_a in enumerate(server_ids)
            for column, server_b in enumerate(server_ids)
        }

    def _get_pair_reliability(self, task, server_a, server_b):
        """Reuse GH results across episodes with the same fixed profiles."""
        key = (task.computation_demand, server_a.server_id, server_b.server_id)
        if key in self.pair_reliability_cache:
            return self.pair_reliability_cache[key]
        exposure_a = (
            server_a.base_failure_rate * task.computation_demand / server_a.processing_frequency
        )
        exposure_b = (
            server_b.base_failure_rate * task.computation_demand / server_b.processing_frequency
        )
        result = self.reliability_integrator.compute_pair(
            exposure_a, exposure_b,
            self.pair_risk_correlations[(server_a.server_id, server_b.server_id)],
        )
        self.pair_reliability_cache[key] = result
        return result

    def _get_reliability_action_mask(self, task):
        """Keep feasible pairs; if none qualify, keep all exact reliability maxima.

        Correlations already reflect SPATIAL_RISK_ENABLED. Disabling the mask
        still records the original feasible count, but never triggers fallback.
        """
        reliabilities = np.array([
            self._get_pair_reliability(task, *self.extract_parameters_from_index(index)).correlated
            for index in range(self.num_actions)
        ])
        mask = reliabilities >= task.reliability_requirement
        task.feasible_action_count = int(mask.sum())
        task.mask_fallback_used = (
            params.RELIABILITY_ACTION_MASK_ENABLED and task.feasible_action_count == 0
        )
        if not params.RELIABILITY_ACTION_MASK_ENABLED:
            return None
        if task.mask_fallback_used:
            mask = reliabilities == reliabilities.max()
        return mask

    def _set_task_reliabilities(self, task, server_a, server_b):
        """Store the selected pair's marginal and joint reliability results."""
        result = self._get_pair_reliability(task, server_a, server_b)
        task.replica_A_reliability = result.replica_a
        task.replica_B_reliability = result.replica_b
        task.pair_reliability_correlated = result.correlated
        task.pair_reliability_independent = result.independent

    def _sample_interarrival_time(self):
        """Sample one inter-arrival time for the common Poisson workload."""
        arrival_rate = float(params.TASK_ARRIVAL_RATE)
        return float(
            self.arrival_rng.exponential(scale=1.0 / arrival_rate)
        )

    # epsilon schedule (DQN only; PPO ignores epsilon in its select_action signature)

    def get_epsilon(self, episode):
        if self.model_name != "dqn":
            return 0.0
        eps_start = params.epsilon_start_dqn
        eps_end = params.epsilon_end_dqn
        eps_decay = params.epsilon_decay_dqn

        return max(eps_end, eps_start - (episode / float(eps_decay)))

    def Iteration(self):
        while self.taskCounter <= self.maxTask:
            yield self.env.timeout(self._sample_interarrival_time())
            current_time = float(self.env.now)

            # PPO closes the previous arrival-to-arrival interval before
            # observing the new task. DQN/DDPG use
            # task-centric bookkeeping below.
            if self.model_name == "ppo":
                self._collect_resolved_task_outcomes()

            task_id = self.taskCounter
            task_profile = self.task_profiles[task_id]
            task = Task(self.env, self.env_state, task_id, task_profile)
            self.env_state.add_task(task)
            self.G_state = self.env_state.get_state(task)

            if self.model_name == "ppo":
                if self.ppo_last_decision_state is not None:
                    delta_t = current_time - self.ppo_last_decision_time
                    self.model.store_transition(
                        self.ppo_last_decision_state,
                        self.ppo_last_decision_action,
                        0.0,  # Filled with the task outcome before updating.
                        self.G_state,
                        delta_t=float(delta_t),
                        done=False,
                        action_mask=self.ppo_last_decision_action_mask,
                    )
            elif self.taskCounter > 1:
                # Complete s' for the previous transition and train on any
                # resolved DQN/DDPG task transitions.
                prev = list(self.tempbuffer[self.taskCounter - 1])
                prev[3] = self.G_state
                self.tempbuffer[self.taskCounter - 1] = tuple(prev)
                self.add_train()

            if self.model_name == "ddpg":
                action_scores = self.model.policy(self.G_state)
                self.G_action = action_scores.numpy().tolist()
                self.G_action = self.model.addNoise(
                    self.G_action, self.this_episode, self.total_episodes
                )
                server_A, server_B = self.extract_parameters_from_action(self.G_action)
            else:
                eps = self.get_epsilon(self.this_episode)
                if self.model_name == "ppo":
                    action_mask = self._get_reliability_action_mask(task)
                    action_index = self.model.select_action(self.G_state, eps, action_mask=action_mask)
                else:
                    action_index = self.model.select_action(self.G_state, eps)
                self.G_action = int(action_index)
                server_A, server_B = self.extract_parameters_from_index(self.G_action)

            self._set_task_reliabilities(task, server_A, server_B)

            if self.model_name == "ppo":
                self.ppo_last_decision_state = self.G_state
                self.ppo_last_decision_action = self.G_action
                self.ppo_last_decision_time = current_time
                self.ppo_last_decision_action_mask = action_mask
            else:
                # Store the DQN/DDPG task transition.
                self.tempbuffer[self.taskCounter] = (self.G_state, self.G_action, None, [])

            self.env.process(task.execute_task(server_A, server_B))
            self.pendingList.append(self.taskCounter)
            self.taskCounter += 1

        if self.model_name != "ppo":
            # The final DQN/DDPG transition uses the last observed state.
            if self.taskCounter > 1:
                last = list(self.tempbuffer[self.taskCounter - 1])
                last[3] = self.G_state
                self.tempbuffer[self.taskCounter - 1] = tuple(last)

        # Drain pending tasks by waiting for first-replica task completion events.
        yield from self._drain_pending_tasks()

        if self.model_name == "ppo":
            # Close the final arrival-to-terminal interval explicitly.
            if self.ppo_last_decision_state is not None:
                terminal_next_state = np.zeros_like(self.ppo_last_decision_state)
                terminal_time = self._get_ppo_terminal_time()
                delta_t = terminal_time - self.ppo_last_decision_time
                self.model.store_transition(
                    self.ppo_last_decision_state,
                    self.ppo_last_decision_action,
                    0.0,  # Filled with the task outcome before updating.
                    terminal_next_state,
                    delta_t=float(delta_t),
                    done=True,
                    action_mask=self.ppo_last_decision_action_mask,
                )
            # Each arrival-ordered decision receives its own task outcome.
            self.model.rewards[:] = [
                self.compute_reward(self.get_task_outcome_info(task))
                for task in self.env_state.tasks.values()
            ]
            # PPO remains on-policy and updates once after the episode.
            self.model.train_step()

        task_avg_delay = self.episodic_delay / self.maxTask
        episode_rsr_correlated = sum(
            task.pair_reliability_correlated >= task.reliability_requirement
            for task in self.env_state.tasks.values()
        ) / self.maxTask
        episode_rsr_independent = sum(
            task.pair_reliability_independent >= task.reliability_requirement
            for task in self.env_state.tasks.values()
        ) / self.maxTask
        mask_fallback_rate = sum(
            task.mask_fallback_used for task in self.env_state.tasks.values()
        ) / self.maxTask
        self.log_data.append((
            self.this_episode,
            self.episodic_reward,
            task_avg_delay,
            episode_rsr_correlated,
            episode_rsr_independent,
            mask_fallback_rate,
        ))

        print(
            f"Episode {self.this_episode} | "
            f"Episode Reward: {self.episodic_reward:.3f} | "
            f"task_Avg_Delay: {task_avg_delay:.3f} s | "
            f"RSR (correlated): {episode_rsr_correlated:.2%} | "
            f"RSR (independent): {episode_rsr_independent:.2%} | "
            f"Mask Fallback: {mask_fallback_rate:.2%}"
        )

    def _drain_pending_tasks(self):
        """Wait for pending tasks' first-replica completion events."""
        while len(self.pendingList) > 0:
            if self.model_name == "ppo":
                self._collect_resolved_task_outcomes()
            else:
                self.add_train()

            if len(self.pendingList) == 0:
                break

            completion_events = [
                self.env_state.tasks[task_id].task_completion_event
                for task_id in self.pendingList
            ]
            yield self.env.any_of(completion_events)

    def _finalize_resolved_task(self, task_counter, reward, delay):
        """Record common episode metrics and remove a task from pendingList."""
        self.episodic_reward += reward
        self.episodic_delay += delay
        self.pendingList.remove(task_counter)

    def get_task_outcome_info(self, task):
        """Build task-level outcome data from its lifecycle and saved snapshot."""
        first = task.first_completed_replica
        return {
            "episode": self.this_episode,
            "task_id": task.id,
            "task_size": task.task_size,
            "computation_demand": task.computation_demand,
            "reliability_requirement": task.reliability_requirement,
            "feasible_action_count": task.feasible_action_count,
            "mask_fallback_used": task.mask_fallback_used,
            "arrival_time": task.arrival_time,
            "server_A_id": task.replica_A["server_id"],
            "server_B_id": task.replica_B["server_id"],
            "replica_A_queue_enter_time": task.replica_A["queue_enter_time"],
            "replica_A_cpu_start_time": task.replica_A["cpu_start_time"],
            "replica_A_finish_time": task.replica_A["finish_time"],
            "replica_B_queue_enter_time": task.replica_B["queue_enter_time"],
            "replica_B_cpu_start_time": task.replica_B["cpu_start_time"],
            "replica_B_finish_time": task.replica_B["finish_time"],
            "task_completion_time": task.task_completion_time,
            "task_delay": task.task_completion_time - task.arrival_time,
            "task_upload_delay": first["queue_enter_time"] - task.arrival_time,
            "task_queue_delay": first["cpu_start_time"] - first["queue_enter_time"],
            "task_execution_delay": first["finish_time"] - first["cpu_start_time"],
            "replica_A_reliability": task.replica_A_reliability,
            "replica_B_reliability": task.replica_B_reliability,
            "pair_reliability_correlated": task.pair_reliability_correlated,
            "pair_reliability_independent": task.pair_reliability_independent,
            "requirement_satisfied_correlated": bool(
                task.pair_reliability_correlated >= task.reliability_requirement
            ),
            "requirement_satisfied_independent": bool(
                task.pair_reliability_independent >= task.reliability_requirement
            ),
        }

    def compute_reward(self, outcome_info):
        return -float(outcome_info["task_delay"])

    def _get_ppo_terminal_time(self):
        """Use the latest actual outcome timestamp for PPO terminal timing."""
        decision_time = float(self.ppo_last_decision_time)
        if self.ppo_last_resolved_outcome_time is None:
            return decision_time
        return max(decision_time, float(self.ppo_last_resolved_outcome_time))

    def _collect_resolved_task_outcomes(self):
        """Record completed task outcomes and the latest completion timestamp."""
        for task_counter in list(self.pendingList):
            task = self.env_state.tasks[task_counter]
            if not task.task_completion_event.triggered:
                continue

            outcome_info = self.get_task_outcome_info(task)
            task_reward = self.compute_reward(outcome_info)
            task_outcome_time = task.task_completion_time
            if self.ppo_last_resolved_outcome_time is None:
                self.ppo_last_resolved_outcome_time = task_outcome_time
            else:
                self.ppo_last_resolved_outcome_time = max(
                    self.ppo_last_resolved_outcome_time, task_outcome_time
                )
            self._finalize_resolved_task(
                task_counter, task_reward, outcome_info["task_delay"]
            )

    def add_train(self):
        for task_counter in list(self.pendingList):
            task = self.env_state.tasks[task_counter]
            if not task.task_completion_event.triggered:
                continue

            outcome_info = self.get_task_outcome_info(task)
            reward = self.compute_reward(outcome_info)
            delay = outcome_info["task_delay"]

            temp = list(self.tempbuffer[task_counter])
            temp[2] = reward
            self.tempbuffer[task_counter] = tuple(temp)

            s, a, r, s_ = self.tempbuffer[task_counter]

            if self.model_name == "ddpg":
                self.model.buffer.record((s, a, r, s_))
                self.model.buffer.learn()
                self.model.update_target(self.model.target_actor.variables, self.model.actor_model.variables)
                self.model.update_target(self.model.target_critic.variables, self.model.critic_model.variables)

            elif self.model_name == "dqn":
                self.model.store_transition((s, int(a), r, s_))
                self.model.train_step()

            self._finalize_resolved_task(task_counter, reward, delay)

    def setServers(self):
        """Create fresh episode-level servers from run-level profiles."""
        for server_id, profile in self.server_profiles.items():
            server = Server(
                self.env,
                server_id,
                profile["Site_ID"],
                profile["Processing_Frequency"],
                profile["Base_Failure_Rate"],
                profile["Transmission_Rate"],
                profile["Latitude"],
                profile["Longitude"],
            )
            self.env_state.add_server_and_init_environment(server)

    def extract_parameters_from_index(self, action_index: int):
        server_A_id, server_B_id = self.index_of_actions[int(action_index)]
        server_A = self.env_state.get_server_by_id(server_A_id)
        server_B = self.env_state.get_server_by_id(server_B_id)
        return server_A, server_B

    def extract_parameters_from_action(self, action_scores_list):
        max_index = int(action_scores_list.index(max(action_scores_list)))
        return self.extract_parameters_from_index(max_index)

    @staticmethod
    def generate_combinations():
        index_of_actions = list(
            combinations(range(1, params.NUM_SERVERS + 1), 2)
        )
        return index_of_actions
