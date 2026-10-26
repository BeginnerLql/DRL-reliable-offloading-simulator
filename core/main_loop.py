from itertools import combinations

from core.server import Server
from core.task import Task, compute_task_qos, compute_task_utility
from core.env_state import EnvironmentState, select_latency_greedy_action
from config.params import params
from config.paths import SERVER_INFO_PATH, TRAIN_TASK_PATH
from pathlib import Path

import simpy
import numpy as np
import pandas as pd


class MainLoop:
    def __init__(
        self, model, total_episodes, maxtaskno, num_states, num_actions,
        *, task_parameters_path=TRAIN_TASK_PATH,
        use_safety_mask=True, failure_rate_scale=1.0, reliability_model="correlated",
    ):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.total_episodes = total_episodes
        self.task_parameters_path = Path(task_parameters_path)
        self.use_safety_mask = bool(use_safety_mask)
        self.failure_rate_scale = float(failure_rate_scale)
        self.reliability_model = reliability_model
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
        self.log_data = []
        self.task_results = []

        # PPO-only SMDP intervals span valid policy decisions, including intervening fallback.
        self.ppo_last_decision_state = None
        self.ppo_last_decision_action = None
        self.ppo_last_decision_time = None
        self.ppo_last_resolved_outcome_time = None
        self.ppo_last_decision_mask = None
        self.ppo_decision_task_ids = []
        self.ppo_interval_reward = 0.0

    def reliability_feasibility(self):
        """Inspect fixed profiles without advancing the workload RNG or changing caches."""
        env = simpy.Environment()
        state = EnvironmentState(failure_rate_scale=self.failure_rate_scale,
                                 reliability_model=self.reliability_model)
        for sid, profile in self.server_profiles.items():
            state.add_server_and_init_environment(Server(
                env, sid, profile["Site_ID"], profile["Processing_Frequency"],
                profile["Base_Failure_Rate"], profile["Transmission_Rate"],
                profile["Latitude"], profile["Longitude"],
            ))
        counts = []
        for tid in range(1, self.maxTask + 1):
            profile = {key: value for key, value in self.task_profiles[tid].items()
                       if key not in ("safe_mask", "reliability_snapshot")}
            task = Task(env, state, tid, profile)
            counts.append(int(state.get_safe_action_mask(task).sum()))
        counts = np.asarray(counts)
        return dict(task_count=len(counts), no_safe_action_count=int((counts == 0).sum()),
                    single_safe_action_count=int((counts == 1).sum()),
                    multi_safe_action_count=int((counts > 1).sum()),
                    mean_safe_pair_count=float(counts.mean()))

    def _load_run_profiles(self):
        """Load fixed profiles once; every episode reuses their attributes/order."""
        server_df = pd.read_excel(
            SERVER_INFO_PATH,
            sheet_name="Servers",
            dtype={"Site_ID": "string"},
        )
        task_df = pd.read_excel(self.task_parameters_path)
        if "Deadline" not in task_df:
            raise ValueError("QoS reward requires a Deadline column in the task workbook")
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
        if "Deadline" in frame:
            deadlines = frame.Deadline.to_numpy(dtype=float)
            if not np.isfinite(deadlines).all() or np.any(deadlines <= 0):
                raise ValueError("Task deadlines must be positive finite seconds")
        return {
            int(row["Task_ID"]): {
                "Task_ID": int(row["Task_ID"]),
                "Task_Size": float(row["Task_Size"]),
                "Computation_Demand": float(row["Computation_Demand"]),
                "Reliability_Requirement": float(row["Reliability_Requirement"]),
                **({"Deadline": float(row["Deadline"])} if "Deadline" in frame else {}),
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
            self.ppo_last_resolved_outcome_time = None
            self.ppo_last_decision_mask = None
            self.ppo_decision_task_ids = []
            self.ppo_interval_reward = 0.0

            self.env = simpy.Environment()
            self.env_state = EnvironmentState(failure_rate_scale=self.failure_rate_scale,
                                              reliability_model=self.reliability_model)

            self.setServers()
            self._initialize_episode_spatial_risk()
            self.env.process(self.Iteration())
            self.env.run()
            for task in sorted(self.env_state.tasks.values(), key=lambda item: item.id):
                self.task_results.append(self.get_task_outcome_info(task))

    def _initialize_episode_spatial_risk(self):
        """Initialize unknown-field GH reliability without an observed risk sample."""
        self.env_state.initialize_spatial_reliability()

    def _store_ppo_interval(self, next_state, next_mask, next_task_id=None, *, done=False):
        """Close an interval only at another PPO decision or the terminal drain."""
        if self.ppo_last_decision_state is None:
            return
        task_id = self.ppo_decision_task_ids[-1]
        task = self.env_state.tasks[task_id]
        end_time = self._get_ppo_terminal_time() if done else float(self.env.now)
        task.ppo_delta_t = end_time - self.ppo_last_decision_time
        task.ppo_next_task_id = next_task_id
        task.ppo_interval_reward = self.ppo_interval_reward
        self.model.store_transition(
            self.ppo_last_decision_state, self.ppo_last_decision_action, self.ppo_interval_reward,
            next_state, delta_t=task.ppo_delta_t, done=done,
            action_mask=self.ppo_last_decision_mask, next_action_mask=next_mask, task_id=task_id,
        )
        self.ppo_interval_reward = 0.0

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

            # Collect all task outcomes, including deterministic fallback tasks.
            if self.model_name == "ppo":
                self._collect_resolved_task_outcomes()

            task_id = self.taskCounter
            task_profile = self.task_profiles[task_id]
            task = Task(self.env, self.env_state, task_id, task_profile)
            self.env_state.add_task(task)
            self.G_state = self.env_state.get_state(task)

            task.safe_mask = self.env_state.get_safe_action_mask(task)
            if self.model_name != "ppo" and self.taskCounter > 1:
                # Complete s' for the previous transition and train on any
                # resolved DQN/DDPG task transitions.
                prev = list(self.tempbuffer[self.taskCounter - 1])
                prev[3] = self.G_state
                self.tempbuffer[self.taskCounter - 1] = tuple(prev)
                self.add_train()

            if self.model_name == "ppo":
                task.is_fallback = not bool(task.safe_mask.any())
                if task.is_fallback:
                    self.G_action = select_latency_greedy_action(self.G_state, self.index_of_actions)
                else:
                    mask = task.safe_mask if self.use_safety_mask else np.ones(self.num_actions, dtype=bool)
                    self._store_ppo_interval(self.G_state, mask, task.id)
                    self.G_action = self.model.select_action(self.G_state, 0.0, action_mask=mask)
                    task.ppo_decision = True
                    task.ppo_action_mask = np.array(mask, dtype=bool, copy=True)
                    task.ppo_transition_index = len(self.ppo_decision_task_ids)
                    self.ppo_decision_task_ids.append(task.id)
                    self.ppo_last_decision_state = self.G_state
                    self.ppo_last_decision_action = self.G_action
                    self.ppo_last_decision_time = current_time
                    self.ppo_last_decision_mask = np.array(mask, copy=True)
                server_A, server_B = self.extract_parameters_from_index(self.G_action)
            elif self.model_name == "ddpg":
                action_scores = self.model.policy(self.G_state)
                self.G_action = action_scores.numpy().tolist()
                self.G_action = self.model.addNoise(
                    self.G_action, self.this_episode, self.total_episodes
                )
                server_A, server_B = self.extract_parameters_from_action(self.G_action)
            else:
                eps = self.get_epsilon(self.this_episode)
                action_index = self.model.select_action(self.G_state, eps)
                self.G_action = int(action_index)
                server_A, server_B = self.extract_parameters_from_index(self.G_action)
            task.action_index = int(self.G_action) if self.model_name != "ddpg" else None

            # Freeze the decision-time reliability snapshot before execution.
            task.replica_A_reliability = (
                self.env_state.compute_replica_reliability(task, server_A)
            )
            task.replica_B_reliability = (
                self.env_state.compute_replica_reliability(task, server_B)
            )
            task.pair_reliability = (
                self.env_state.compute_pair_reliability(task, server_A, server_B)
            )

            if self.model_name != "ppo":
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
            if self.ppo_decision_task_ids:
                self._store_ppo_interval(
                    np.zeros_like(self.ppo_last_decision_state),
                    np.zeros(self.num_actions, dtype=bool), done=True,
                )
            # Rewards were collected at completion and assigned to decision intervals.
            if not np.isclose(sum(self.model.rewards), self.episodic_reward, rtol=0, atol=1e-9):
                raise RuntimeError("PPO interval rewards must equal total effective task QoS")
            # PPO remains on-policy and updates once after the episode.
            if self.ppo_decision_task_ids:
                self.model.train_step()

        task_avg_delay = self.episodic_delay / self.maxTask
        self.log_data.append((
            self.this_episode,
            self.episodic_reward,
            task_avg_delay,
        ))

        print(
            f"Episode {self.this_episode} | "
            f"Episode Reward: {self.episodic_reward:.3f} | "
            f"task_Avg_Delay: {task_avg_delay:.3f} s"
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
        outcome = {
            "episode": self.this_episode,
            "task_id": task.id,
            "task_size": task.task_size,
            "computation_demand": task.computation_demand,
            "reliability_requirement": task.reliability_requirement,
            "arrival_time": task.arrival_time,
            "deadline_s": task.deadline,
            "absolute_deadline_s": task.absolute_deadline,
            "action_index": task.action_index,
            "ppo_action": task.action_index if task.ppo_decision else None,
            "safe_mask": "".join("1" if flag else "0" for flag in task.safe_mask),
            "action_mask": ("".join("1" if flag else "0" for flag in task.ppo_action_mask)
                            if task.ppo_decision else None),
            "is_fallback": task.is_fallback,
            "ppo_decision": task.ppo_decision,
            "ppo_transition_index": task.ppo_transition_index,
            "ppo_delta_t": task.ppo_delta_t,
            "ppo_next_task_id": task.ppo_next_task_id,
            "ppo_reward_owner_task_id": task.ppo_reward_owner_task_id,
            "ppo_interval_reward": task.ppo_interval_reward,
            "safe_pair_count": int(np.count_nonzero(task.safe_mask)),
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
        outcome["task_qos"] = compute_task_qos(
            outcome["task_latency"], outcome["deadline_s"],
            grace_multiplier=params.QOS_GRACE_MULTIPLIER)
        outcome["task_utility"] = MainLoop.compute_reward(outcome)
        if hasattr(task, "rasoul_decision"):
            decision = task.rasoul_decision
            outcome.update(
                selection_reliability_model="marginal",
                selection_reliability=task.replica_A_reliability,
                selection_requirement_satisfied=outcome["requirement_satisfied"],
                predicted_qos=decision.predicted_qos,
                predicted_latency_s=decision.predicted_latency_s,
                fallback_reason=decision.fallback_reason,
                safe_node_mask="".join("1" if flag else "0" for flag in decision.safe_node_mask),
                safe_node_count=sum(decision.safe_node_mask),
                admitted_node_mask="".join("1" if flag else "0" for flag in decision.admitted_node_mask),
                admitted_node_count=sum(decision.admitted_node_mask),
                rasoul_score=decision.selection_score,
                rasoul_cold_start=decision.feedback_count == 0,
                rasoul_history_mean_error=decision.history_mean_error,
                rasoul_confidence_bonus=decision.confidence_bonus,
                rasoul_feedback_count_before=decision.feedback_count,
                rasoul_decision_count=decision.decision_count,
                rasoul_selected_load=decision.selected_load,
                rasoul_selected_load_after_admission=decision.selected_load_after_admission,
            )
            outcome.update({f"rasoul_{key}": value for key, value in getattr(task, "rasoul_feedback", {}).items()})
        if hasattr(task, "tore_decision"):
            decision = task.tore_decision
            outcome.update(
                selection_reliability_model="independent",
                selection_reliability=decision.independent_reliability,
                selection_requirement_satisfied=bool(
                    decision.independent_reliability >= task.reliability_requirement),
                predicted_qos=decision.predicted_qos,
                predicted_latency_s=decision.predicted_latency_s,
                greedy_server_ids=decision.greedy_server_ids,
                greedy_reliability=decision.greedy_reliability,
                fallback_reason="replica_limit" if decision.is_fallback else None,
                independent_feasible_pair_exists=bool(task.safe_mask.any()),
                greedy_missed_feasible_pair=bool(decision.is_fallback and task.safe_mask.any()),
            )
        return outcome

    @staticmethod
    def compute_reward(outcome_info):
        """Per-task objective contribution; PPO sums these over completion intervals."""
        qos = compute_task_qos(outcome_info["task_latency"], outcome_info["deadline_s"],
                               grace_multiplier=params.QOS_GRACE_MULTIPLIER)
        return compute_task_utility(qos, outcome_info["requirement_satisfied"])

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
            if self.ppo_last_decision_state is not None:
                task.ppo_reward_owner_task_id = self.ppo_decision_task_ids[-1]
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

    def add_train(self):
        for task_counter in list(self.pendingList):
            task = self.env_state.tasks[task_counter]
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
