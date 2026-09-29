# mainLoop.py  (multi-model: DDPG / DQN / PPO)
# - MainLoop signature simplified: no external buffer
# - DDPG uses model.policy() -> scores -> argmax -> (server_A, server_B)
# - DQN/PPO use model.select_action(state, epsilon) -> pair action index
# - PPO trains once at end of episode; DQN trains online

from itertools import combinations

from core.server import Server
from core.task import Task
from core.env_state import EnvironmentState
from config.params import params
from config.paths import DATA_DIR
from pathlib import Path
from core.spatial_risk import (
    build_distance_matrix,
    build_spatial_correlation_matrix,
    map_spatial_risk_to_effective_failure_rates,
    sample_spatial_risk_field,
    validate_correlation_matrix,
)

import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class MainLoop:
    def __init__(self, model, total_episodes, maxtaskno, num_states, num_actions):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.total_episodes = total_episodes
        self.server_profiles, self.task_profiles = self._load_run_profiles()

        self.model_name = str(getattr(params, "model_summary", "ddpg")).strip().lower()

        self.rewardsAll = []
        self.ep_reward_list = []
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
        self.task_results = []

        # PPO-only arrival-driven SMDP bookkeeping.
        self.ppo_interval_reward = 0.0
        self.ppo_last_decision_state = None
        self.ppo_last_decision_action = None
        self.ppo_last_decision_time = None
        self.ppo_last_resolved_outcome_time = None


    def _load_run_profiles(self):
        """Load and validate the immutable server/task profiles for this run."""
        server_path = Path(DATA_DIR) / "server_info.xlsx"
        task_path = Path(DATA_DIR) / "task_parameters.xlsx"
        missing_paths = [path for path in (server_path, task_path) if not path.is_file()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Required run profile workbook not found: {missing}")

        server_df = pd.read_excel(
            server_path,
            sheet_name="Servers",
            dtype={"Site_ID": "string"},
        )
        task_df = pd.read_excel(task_path)
        return (
            self._validate_server_profiles(server_df, server_path),
            self._validate_task_profiles(task_df, task_path),
        )

    @staticmethod
    def _validate_server_profiles(server_df, source_path):
        required_columns = {
            "Server_ID",
            "Site_ID",
            "Processing_Frequency",
            "Base_Failure_Rate",
            "Latitude",
            "Longitude",
        }
        missing = sorted(required_columns.difference(server_df.columns))
        if missing:
            raise ValueError(
                f"{source_path} is missing required columns: " + ", ".join(missing)
            )

        server_count = int(params.NUM_SERVERS)
        if len(server_df) != server_count:
            raise ValueError(
                f"{source_path} must contain {server_count} server rows; found {len(server_df)}."
            )

        frame = server_df.copy()
        server_ids = pd.to_numeric(frame["Server_ID"], errors="coerce")
        if (
            server_ids.isna().any()
            or not server_ids.map(math.isfinite).all()
            or not server_ids.map(lambda value: float(value).is_integer()).all()
        ):
            raise ValueError(f"{source_path} contains invalid Server_ID values.")
        frame["Server_ID"] = server_ids.astype(int)
        expected_ids = list(range(1, server_count + 1))
        if (
            frame["Server_ID"].duplicated().any()
            or sorted(frame["Server_ID"].tolist()) != expected_ids
        ):
            raise ValueError(f"{source_path} must have unique Server_ID values 1..N.")

        site_ids = frame["Site_ID"].astype("string").str.strip()
        if site_ids.isna().any() or site_ids.eq("").any():
            raise ValueError(f"{source_path} contains an empty Site_ID.")
        if site_ids.duplicated().any():
            raise ValueError(f"{source_path} contains duplicate Site_ID values.")
        frame["Site_ID"] = site_ids

        numeric_ranges = (
            ("Processing_Frequency", 0.0, None),
            ("Base_Failure_Rate", 0.0, None),
            ("Latitude", -90.0, 90.0),
            ("Longitude", -180.0, 180.0),
        )
        for column, minimum, maximum in numeric_ranges:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or not values.map(math.isfinite).all():
                raise ValueError(f"{source_path} contains invalid {column} values.")
            if column == "Processing_Frequency" and not values.gt(0).all():
                raise ValueError(f"{source_path} requires positive Processing_Frequency values.")
            if column == "Base_Failure_Rate" and not values.ge(0).all():
                raise ValueError(f"{source_path} requires non-negative Base_Failure_Rate values.")
            if maximum is not None and not values.between(minimum, maximum).all():
                raise ValueError(f"{source_path} contains out-of-range {column} values.")
            frame[column] = values.astype(float)

        frame = frame.sort_values("Server_ID").reset_index(drop=True)
        profiles = {}
        for _, row in frame.iterrows():
            server_id = int(row["Server_ID"])
            profiles[server_id] = {
                "Server_ID": server_id,
                "Site_ID": str(row["Site_ID"]),
                "Processing_Frequency": float(row["Processing_Frequency"]),
                "Base_Failure_Rate": float(row["Base_Failure_Rate"]),
                "Latitude": float(row["Latitude"]),
                "Longitude": float(row["Longitude"]),
            }
        return profiles

    @staticmethod
    def _validate_task_profiles(task_df, source_path):
        required_columns = {
            "Task_ID",
            "Task_Size",
            "Computation_Demand",
            "Reliability_Requirement",
        }
        missing = sorted(required_columns.difference(task_df.columns))
        if missing:
            raise ValueError(
                f"{source_path} is missing required columns: " + ", ".join(missing)
            )

        task_count = int(params.taskno)
        if len(task_df) != task_count:
            raise ValueError(
                f"{source_path} must contain {task_count} task rows; found {len(task_df)}."
            )

        frame = task_df.copy()
        task_ids = pd.to_numeric(frame["Task_ID"], errors="coerce")
        if (
            task_ids.isna().any()
            or not task_ids.map(math.isfinite).all()
            or not task_ids.map(lambda value: float(value).is_integer()).all()
        ):
            raise ValueError(f"{source_path} contains invalid Task_ID values.")
        frame["Task_ID"] = task_ids.astype(int)
        expected_ids = list(range(1, task_count + 1))
        if (
            frame["Task_ID"].duplicated().any()
            or sorted(frame["Task_ID"].tolist()) != expected_ids
        ):
            raise ValueError(f"{source_path} must have unique Task_ID values 1..T.")

        task_sizes = pd.to_numeric(frame["Task_Size"], errors="coerce")
        if (
            task_sizes.isna().any()
            or not task_sizes.map(math.isfinite).all()
            or not task_sizes.map(lambda value: float(value).is_integer()).all()
        ):
            raise ValueError(f"{source_path} contains invalid Task_Size values.")
        minimum_size, maximum_size = params.TASK_SIZE_RANGE
        if not task_sizes.between(minimum_size, maximum_size).all():
            raise ValueError(f"{source_path} contains out-of-range Task_Size values.")
        frame["Task_Size"] = task_sizes.astype(int)

        computation = pd.to_numeric(frame["Computation_Demand"], errors="coerce")
        if computation.isna().any() or not computation.map(math.isfinite).all():
            raise ValueError(f"{source_path} contains invalid Computation_Demand values.")
        if not computation.gt(0).all():
            raise ValueError(f"{source_path} requires positive Computation_Demand values.")
        frame["Computation_Demand"] = computation.astype(float)

        reliability = pd.to_numeric(frame["Reliability_Requirement"], errors="coerce")
        if (
            reliability.isna().any()
            or not reliability.map(math.isfinite).all()
            or not ((reliability > 0.0) & (reliability <= 1.0)).all()
        ):
            raise ValueError(
                f"{source_path} requires finite Reliability_Requirement values in (0, 1]."
            )
        frame["Reliability_Requirement"] = reliability.astype(float)

        frame = frame.sort_values("Task_ID").reset_index(drop=True)
        profiles = {}
        for _, row in frame.iterrows():
            task_id = int(row["Task_ID"])
            profiles[task_id] = {
                "Task_ID": task_id,
                "Task_Size": int(row["Task_Size"]),
                "Computation_Demand": float(row["Computation_Demand"]),
                "Reliability_Requirement": float(row["Reliability_Requirement"]),
            }
        return profiles

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
            for task in sorted(self.env_state.tasks.values(), key=lambda item: item.id):
                self.task_results.append(self.get_task_outcome_info(task))


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
            self.env_state.get_server_by_id(server_id).base_failure_rate
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

            task_id = self.taskCounter
            task_profile = self.task_profiles[task_id]
            task = Task(self.env, self.env_state, task_id, task_profile)
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
                server_A, server_B = self.extract_parameters_from_action(self.G_action)
            else:
                # DQN/PPO output a discrete action index.
                eps = self.get_epsilon(self.this_episode)
                action_index = self.model.select_action(self.G_state, eps)
                self.G_action = int(action_index)
                server_A, server_B = self.extract_parameters_from_index(self.G_action)

            task.replica_A_reliability = (
                self.env_state.compute_replica_reliability(task, server_A)
            )
            task.replica_B_reliability = (
                self.env_state.compute_replica_reliability(task, server_B)
            )
            task.pair_reliability = (
                self.env_state.compute_pair_reliability(task, server_A, server_B)
            )

            if self.model_name == "ppo":
                self.ppo_last_decision_state = self.G_state
                self.ppo_last_decision_action = self.G_action
                self.ppo_last_decision_time = current_time
            else:
                # Store the legacy task-centric transition for DQN/DDPG.
                self.tempbuffer[self.taskCounter] = (self.G_state, self.G_action, None, [])

            self.env.process(task.execute_task(server_A, server_B))
            self.pendingList.append(self.taskCounter)
            self.taskCounter += 1

        if self.model_name != "ppo":
            # Preserve the legacy final next-state placeholder for DQN/DDPG.
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

        avg_reward = np.mean(self.ep_reward_list[-40:])
        episode_avg_delay = self.episodic_delay / self.maxTask
        self.log_data.append((
            self.this_episode,
            avg_reward,
            self.episodic_reward,
            episode_avg_delay,
        ))
        self.avg_reward_list.append(avg_reward)

        print(f"Episode {self.this_episode} | Avg Reward: {avg_reward:.3f} | This Episode: {self.episodic_reward:.3f}")

    def _drain_pending_tasks(self):
        """Wait for pending tasks' first-replica completion events."""
        while len(self.pendingList) > 0:
            if self.model_name == "ppo":
                self._collect_resolved_task_outcomes()
            else:
                self.add_train()

            if len(self.pendingList) == 0:
                break

            completion_events = []
            for task_id in self.pendingList:
                task = self.env_state.get_task_by_id(task_id)
                if task is None:
                    raise RuntimeError(
                        f"Pending task {task_id} is missing from the environment state"
                    )
                completion_event = getattr(task, "task_completion_event", None)
                if completion_event is None:
                    raise RuntimeError(
                        f"Pending task {task_id} has no task completion event"
                    )
                if completion_event.triggered:
                    raise RuntimeError(
                        f"Pending task {task_id} has a triggered task completion event "
                        "but is still pending"
                    )
                completion_events.append(completion_event)

            if not completion_events:
                raise RuntimeError("No task completion events remain for pending tasks")

            yield self.env.any_of(completion_events)

    def _finalize_resolved_task(self, task_counter, reward, delay):
        """Record common episode metrics and remove a task from pendingList."""
        self.episodic_reward += reward
        self.episodic_delay += delay
        self.rewardsAll.append(reward)
        self.pendingList.remove(task_counter)

    def get_task_outcome_info(self, task):
        """Build task-level outcome data from its lifecycle and saved snapshot."""
        return {
            "episode": self.this_episode,
            "task_id": task.id,
            "task_size": task.task_size,
            "computation_demand": task.computation_demand,
            "reliability_requirement": task.reliability_requirement,
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
            "task_latency": task.task_completion_time - task.arrival_time,
            "replica_A_reliability": task.replica_A_reliability,
            "replica_B_reliability": task.replica_B_reliability,
            "pair_reliability": task.pair_reliability,
            "requirement_satisfied": bool(
                task.pair_reliability >= task.reliability_requirement
            ),
        }

    def compute_reward(self, outcome_info):
        return -float(outcome_info["task_latency"])

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
            if not task.task_completion_event.triggered:
                continue

            outcome_info = self.get_task_outcome_info(task)
            task_reward = self.compute_reward(outcome_info)
            self.ppo_interval_reward += task_reward
            task_outcome_time = task.task_completion_time
            if self.ppo_last_resolved_outcome_time is None:
                self.ppo_last_resolved_outcome_time = task_outcome_time
            else:
                self.ppo_last_resolved_outcome_time = max(
                    self.ppo_last_resolved_outcome_time, task_outcome_time
                )
            self._finalize_resolved_task(
                task_counter, task_reward, outcome_info["task_latency"]
            )

    # ---------------------------
    # TRAINING (multi-model)
    # ---------------------------
    def add_train(self):
        if self.model_name == "ppo":
            self._collect_resolved_task_outcomes()
            return

        for task_counter in list(self.pendingList):
            task = self.env_state.get_task_by_id(task_counter)
            if not task.task_completion_event.triggered:
                continue

            outcome_info = self.get_task_outcome_info(task)
            reward = self.compute_reward(outcome_info)
            delay = outcome_info["task_latency"]

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

            self._finalize_resolved_task(task_counter, reward, delay)

    # ---------------------------
    # SERVERS 
    # ---------------------------
    def setServers(self):
        """Create fresh episode-level servers from run-level profiles."""
        for server_id, profile in self.server_profiles.items():
            server = Server(
                self.env,
                server_id,
                profile["Site_ID"],
                profile["Processing_Frequency"],
                profile["Base_Failure_Rate"],
                profile["Latitude"],
                profile["Longitude"],
            )
            self.env_state.add_server_and_init_environment(server)

    # ---------------------------
    # ACTION DECODING
    # ---------------------------
    def extract_parameters_from_index(self, action_index: int):
        server_A_id, server_B_id = self.index_of_actions[int(action_index)]
        server_A = self.env_state.get_server_by_id(server_A_id)
        server_B = self.env_state.get_server_by_id(server_B_id)
        return server_A, server_B

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
        index_of_actions = list(
            combinations(range(1, params.NUM_SERVERS + 1), 2)
        )
        if len(index_of_actions) != params.num_actions:
            raise RuntimeError(
                f"Expected {params.num_actions} pair actions, "
                f"generated {len(index_of_actions)}"
            )
        return index_of_actions
