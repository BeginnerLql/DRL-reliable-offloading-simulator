"""Objective conservation and causal completion intervals in the real SimPy loop."""

import contextlib
import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from agents.ppo_agent import PPOAgent
from compare_tore import all_task_qos_metrics
from config.params import params
from core.main_loop import MainLoop
from core.task import compute_task_utility
from io_utils.save_parameters_and_logs import summarize_task_results


class RecordingAgent:
    def __init__(self, action=0):
        self.action = action
        self.rewards = []
        self.transitions = []
        self.episodes = []

    def select_action(self, state, epsilon, action_mask):
        assert action_mask[self.action]
        return self.action

    def store_transition(self, state, action, reward, next_state, **metadata):
        self.rewards.append(reward)
        self.transitions.append(dict(state=state.copy(), action=action, reward=reward,
                                     next_state=next_state.copy(), **metadata))

    def train_step(self):
        self.episodes.append(copy.deepcopy(self.transitions))
        self.rewards.clear()
        self.transitions.clear()


class FixtureLoop(MainLoop):
    def __init__(self, agent, requirements, gaps, *, episodes=1, masked=True, slow_loser=False):
        self.requirements = requirements
        self.gaps = gaps
        self.slow_loser = slow_loser
        super().__init__(agent, episodes, len(requirements), params.num_states, params.num_actions,
                         use_safety_mask=masked)

    def _load_run_profiles(self):
        servers = {sid: dict(Site_ID=str(sid), Processing_Frequency=4e9,
            Base_Failure_Rate=.01, Transmission_Rate=20., Latitude=0., Longitude=sid/100.)
            for sid in range(1, 9)}
        if self.slow_loser:
            servers[2]['Processing_Frequency'] = 1e9
        pair = np.full(28, .2)
        pair[0] = .99
        tasks = {tid: dict(Task_Size=5., Computation_Demand=2e9,
            Reliability_Requirement=req, Deadline=10.,
            reliability_snapshot=(np.full(8, .9), pair.copy(), pair.copy()))
            for tid, req in enumerate(self.requirements, 1)}
        return servers, tasks

    def _sample_interarrival_time(self):
        return self.gaps[self.taskCounter-1]


def run(loop):
    with contextlib.redirect_stdout(io.StringIO()):
        loop.EP()
    return pd.DataFrame(loop.task_results)


class ObjectiveAlignmentTests(unittest.TestCase):
    def test_completion_rewards_move_to_later_decisions_and_include_terminal_tail(self):
        agent = RecordingAgent()
        loop = FixtureLoop(agent, [.9, .9, 1., .9], [.1, .1, 1., .1], episodes=2)
        records = run(loop)
        self.assertEqual(len(agent.episodes), 2)
        for transitions in agent.episodes:
            self.assertEqual([row['task_id'] for row in transitions], [1, 2, 4])
            # Task 1 finishes after decision 2, before decision 4; own-task backfill fails this.
            self.assertEqual([row['reward'] for row in transitions], [0., 1., 2.])
            self.assertEqual([row['done'] for row in transitions], [False, False, True])
            np.testing.assert_allclose([row['delta_t'] for row in transitions[:2]], [.1, 1.1])
        self.assertTrue(records.task_qos.eq(1.).all())
        self.assertTrue(records.loc[records.is_fallback, 'task_utility'].eq(0.).all())
        for episode, group in records.groupby('episode'):
            transitions = agent.episodes[episode-1]
            by_id = group.set_index('task_id')
            self.assertEqual(by_id.loc[1, 'ppo_reward_owner_task_id'], 2)
            self.assertEqual(by_id.loc[2, 'ppo_reward_owner_task_id'], 4)
            self.assertTrue(pd.isna(by_id.loc[3, 'ppo_interval_reward']))
            self.assertEqual(sum(row['reward'] for row in transitions), group.task_utility.sum())
            self.assertEqual(loop.log_data[episode-1][1], 3.)
            self.assertEqual(all_task_qos_metrics(group)['mean_qos'], .75)
            summary, _, loads = summarize_task_results(group)
            self.assertEqual(summary.mean_qos.iloc[0], .75)
            self.assertEqual(summary.mean_raw_qos.iloc[0], 1.)
            self.assertEqual(loads.replica_count.sum(), 8)

    def test_all_fallback_has_zero_effective_return_and_no_policy_update(self):
        agent = RecordingAgent()
        loop = FixtureLoop(agent, [1., 1.], [.1, .1])
        records = run(loop)
        self.assertFalse(agent.episodes)
        self.assertFalse(agent.transitions)
        self.assertTrue(records.task_qos.eq(1.).all())
        self.assertTrue(records.task_utility.eq(0.).all())
        self.assertEqual(loop.log_data[0][1], 0.)

    def test_prefix_fallback_and_losing_replica_drain_do_not_create_extra_rewards(self):
        agent = RecordingAgent()
        loop = FixtureLoop(agent, [1., .9], [.1, 1.], slow_loser=True)
        records = run(loop)
        transitions = agent.episodes[0]
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]['reward'], 1.)
        terminal = records.task_completion_time.max()
        self.assertAlmostEqual(transitions[0]['delta_t'], terminal-records.arrival_time.iloc[1])
        self.assertGreater(records.replica_B_finish_time.max(), terminal)
        self.assertEqual(records.task_utility.sum(), 1.)

    def test_unmasked_unsafe_choice_gets_zero_despite_nonempty_safe_set(self):
        loop = FixtureLoop(RecordingAgent(action=1), [.9], [.1], masked=False)
        records = run(loop)
        self.assertFalse(records.is_fallback.iloc[0])
        self.assertFalse(records.requirement_satisfied.iloc[0])
        self.assertEqual(records.task_qos.iloc[0], 1.)
        self.assertEqual(loop.model.episodes[0][0]['reward'], 0.)

    def test_effective_utility_preserves_reliable_tasks_and_indexes(self):
        records = pd.DataFrame(dict(task_qos=[1., .6, .8],
            requirement_satisfied=[False, True, True], is_fallback=[False, True, False]), index=[8, 3, 9])
        np.testing.assert_array_equal(compute_task_utility(records.task_qos, records.requirement_satisfied),
                                      [0., .6, .8])
        self.assertAlmostEqual(all_task_qos_metrics(records)['mean_qos'], 1.4/3)
        with self.assertRaises(ValueError):
            compute_task_utility(float('nan'), False)

    def test_real_ppo_updates_with_interval_rewards_and_gamma_one(self):
        torch.manual_seed(7)
        agent = PPOAgent(params.num_states, params.num_actions, [8], k_epochs=1, batch_size=2)
        loop = FixtureLoop(agent, [.9, .9, 1., .9], [.1, .1, 1., .1])
        captured = []
        original = agent.train_step
        def update():
            captured.append(list(agent.rewards))
            original()
        agent.train_step = update
        records = run(loop)
        self.assertEqual(agent.gamma, 1.)
        self.assertEqual(captured, [[0., 1., 2.]])
        self.assertTrue(agent.optimizer_value.state)
        self.assertFalse(agent.rewards)
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.value_net.parameters()))
        self.assertEqual(sum(captured[0]), records.task_utility.sum())

    def test_preflight_rebuilds_stale_cache_without_mutating_inputs_or_rng(self):
        loop = FixtureLoop(None, [.9, 1.], [.1, .1])
        before = copy.deepcopy(loop.arrival_rng.bit_generator.state)
        snapshots = [p['reliability_snapshot'] for p in loop.task_profiles.values()]
        result = loop.reliability_feasibility()
        self.assertEqual(result['no_safe_action_count'], 1)
        self.assertEqual(result['multi_safe_action_count'], 1)
        self.assertEqual(before, loop.arrival_rng.bit_generator.state)
        self.assertTrue(all(p['reliability_snapshot'] is snapshot
                            for p, snapshot in zip(loop.task_profiles.values(), snapshots)))

    def test_main_rejects_all_fallback_before_building_a_gpu_model(self):
        import Project_main
        loop = FixtureLoop(None, [1.], [.1])
        with patch.object(Project_main, 'MainLoop', return_value=loop), \
             patch.object(Project_main, 'build_model') as build, \
             patch.object(params, 'taskno', 1), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'no feasible decisions'):
                Project_main.run_simulation()
            build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
