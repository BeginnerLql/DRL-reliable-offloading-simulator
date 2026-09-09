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

        # tempbuffer[taskCounter] = (s, a, r, s')
        self.tempbuffer = {}
        self.taskCounter = 1
        self.pendingList = []
        self.maxTask = maxtaskno

        self.env = None
        self.env_state = None
        self.log_data = []
        self.task_Assignments_info = []


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

            self.env = simpy.Environment()
            self.env_state = EnvironmentState()
            self.env_state.reset()

            self.setServers()
            self.env.process(self.Iteration())
            self.env.run()


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
            yield self.env.timeout(np.random.poisson(1 / params.TASK_ARRIVAL_RATE))

            task = Task(self.env, self.env_state, self.taskCounter)
            self.env_state.add_task(task)

            # build state vector
            self.G_state = self.env_state.get_state(task)

            # complete s' for previous transition and train on any resolved tasks
            if self.taskCounter > 1:
                prev = list(self.tempbuffer[self.taskCounter - 1])
                prev[3] = self.G_state
                self.tempbuffer[self.taskCounter - 1] = tuple(prev)
                self.add_train()

            # -------- action selection --------
            if self.model_name == "ddpg":
                # DDPG outputs continuous scores over actions
                action_scores = self.model.policy(self.G_state)  # torch tensor (CPU)
                self.G_action = action_scores.numpy().tolist()
                self.G_action = self.model.addNoise(self.G_action, self.this_episode, self.total_episodes)

                X, Y, Z = self.extract_parameters_from_action(self.G_action)

            else:
                # DQN/PPO output a discrete action index
                eps = self.get_epsilon(self.this_episode)
                action_index = self.model.select_action(self.G_state, eps)
                self.G_action = int(action_index)

                X, Y, Z = self.extract_parameters_from_index(self.G_action)

            # store pending transition (reward filled later)
            self.tempbuffer[self.taskCounter] = (self.G_state, self.G_action, None, [])
            self.env.process(task.execute_task(X, Y, Z))
            self.pendingList.append(self.taskCounter)

            self.taskCounter += 1

        # finalize last transition next-state placeholder
        if self.taskCounter > 1:
            last = list(self.tempbuffer[self.taskCounter - 1])
            last[3] = self.G_state
            self.tempbuffer[self.taskCounter - 1] = tuple(last)

        # drain pending tasks until all rewards resolved
        while len(self.pendingList) > 0:
            yield_time = self.env_state.get_min_computation_demand()
            yield self.env.timeout(yield_time)
            self.add_train()

        # PPO: update policy at end of episode (on-policy)
        if self.model_name == "ppo":
            self.model.train_step()

        # episode logs
        self.ep_reward_list.append(self.episodic_reward)
        self.ep_delay_list.append(self.episodic_delay)

        avg_reward = np.mean(self.ep_reward_list[-40:])
        avg_delay = np.mean(self.ep_delay_list[-40:])
        self.log_data.append((self.this_episode, avg_reward, self.episodic_reward, avg_delay))
        self.avg_reward_list.append(avg_reward)

        print(f"Episode {self.this_episode} | Avg Reward: {avg_reward:.3f} | This Episode: {self.episodic_reward:.3f}")

    # ---------------------------
    # REWARD CALCULATION (unchanged)
    # ---------------------------
    def calcReward(self, taskID):
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

            elif self.model_name == "ppo":
                self.model.store_transition(s, int(a), r, s_, done=False)

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

        for _, row in server_info_df.iterrows():
            server_id = int(row["Server_ID"])
            server_type = str(row["Server_Type"])
            processing_frequency = float(row["Processing_Frequency"])
            failure_rate = float(row["Failure_Rate"])

            server = Server(self.env, server_type, server_id, processing_frequency, failure_rate)
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
