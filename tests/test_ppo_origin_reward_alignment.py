import unittest

import numpy as np

from agents.ppo_agent import PPOAgent


class PPOOriginRewardAlignmentTests(unittest.TestCase):
    def _agent(self, min_rollout=8):
        return PPOAgent(
            num_states=3,
            num_actions=2,
            hidden_layers=[8],
            min_rollout=min_rollout,
            k_epochs=0,
        )

    @staticmethod
    def _store(agent, task_id, reward=None, done=False):
        state = np.full(3, float(task_id))
        agent.store_transition(
            state,
            0,
            reward,
            state + 0.5,
            delta_t=1.0,
            done=done,
            task_id=task_id,
        )

    def test_reward_after_transition_backfills_origin(self):
        agent = self._agent()
        self._store(agent, 1)
        agent.assign_task_reward(1, 5.0)
        self.assertEqual(agent.rewards, [5.0])
        self.assertEqual(agent.pending_task_rewards, {})

    def test_reward_before_transition_is_consumed_by_store(self):
        agent = self._agent()
        agent.assign_task_reward(1, 5.0)
        self._store(agent, 1)
        self.assertEqual(agent.rewards, [5.0])
        self.assertEqual(agent.pending_task_rewards, {})

    def test_out_of_order_completion_keeps_arrival_order(self):
        agent = self._agent()
        for task_id in (1, 2, 3):
            self._store(agent, task_id)
        for task_id, reward in ((3, 7.0), (1, 10.0), (2, -4.0)):
            agent.assign_task_reward(task_id, reward)
        self.assertEqual(agent.task_ids, [1, 2, 3])
        self.assertEqual(agent.rewards, [10.0, -4.0, 7.0])

    def test_delayed_high_reliability_task_stays_with_origin(self):
        agent = self._agent()
        self._store(agent, 1)
        self._store(agent, 2)
        agent.assign_task_reward(2, -1.0)
        agent.assign_task_reward(1, 10.0)
        self.assertEqual(agent.rewards, [10.0, -1.0])

    def test_multiple_completed_tasks_are_not_summed(self):
        agent = self._agent()
        self._store(agent, 1)
        self._store(agent, 2)
        agent.assign_task_reward(1, 3.0)
        agent.assign_task_reward(2, 4.0)
        self.assertEqual(agent.rewards, [3.0, 4.0])

    def test_duplicate_reward_assignment_raises(self):
        agent = self._agent()
        self._store(agent, 1)
        agent.assign_task_reward(1, 5.0)
        with self.assertRaises(RuntimeError):
            agent.assign_task_reward(1, 5.0)

    def test_duplicate_pending_reward_assignment_raises(self):
        agent = self._agent()
        agent.assign_task_reward(1, 5.0)
        with self.assertRaises(RuntimeError):
            agent.assign_task_reward(1, 5.0)

    def test_duplicate_task_id_transition_raises(self):
        agent = self._agent()
        self._store(agent, 1)
        with self.assertRaises(RuntimeError):
            self._store(agent, 1)

    def test_train_step_rejects_unresolved_reward_and_clears_rollout(self):
        agent = self._agent(min_rollout=1)
        self._store(agent, 1, done=True)
        with self.assertRaises(RuntimeError):
            agent.train_step()
        self.assertEqual(agent.states, [])
        self.assertEqual(agent.task_ids, [])
        self.assertEqual(agent.task_id_to_transition_index, {})
        self.assertEqual(agent.pending_task_rewards, {})

    def test_terminal_reward_before_transition_is_backfilled(self):
        agent = self._agent()
        agent.assign_task_reward(200, 12.0)
        self._store(agent, 200, done=True)
        self.assertEqual(agent.rewards, [12.0])
        self.assertTrue(agent.dones[0])

    def test_clear_rollout_clears_identity_and_pending_state(self):
        agent = self._agent()
        agent.assign_task_reward(1, 2.0)
        self._store(agent, 2)
        agent.clear_rollout()
        self.assertEqual(agent.states, [])
        self.assertEqual(agent.actions, [])
        self.assertEqual(agent.rewards, [])
        self.assertEqual(agent.next_states, [])
        self.assertEqual(agent.dones, [])
        self.assertEqual(agent.old_log_probs, [])
        self.assertEqual(agent.delta_times, [])
        self.assertEqual(agent.task_ids, [])
        self.assertEqual(agent.task_id_to_transition_index, {})
        self.assertEqual(agent.pending_task_rewards, {})

    def test_synthetic_three_task_async_completion_integration(self):
        agent = self._agent()
        for task_id in (1, 2, 3):
            self._store(agent, task_id, done=task_id == 3)
        for task_id, reward in ((2, -4.0), (3, 7.0), (1, 10.0)):
            agent.assign_task_reward(task_id, reward)
        self.assertEqual(
            list(zip(agent.task_ids, agent.rewards)),
            [(1, 10.0), (2, -4.0), (3, 7.0)],
        )


if __name__ == "__main__":
    unittest.main()
