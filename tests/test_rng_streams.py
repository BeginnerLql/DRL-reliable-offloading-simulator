import unittest
from types import SimpleNamespace

import numpy as np

from agents.ppo_agent import PPOAgent
from config.params import params
from core.main_loop import MainLoop


class IndependentRngStreamTests(unittest.TestCase):
    def _loop(self):
        return MainLoop(
            model=SimpleNamespace(),
            total_episodes=0,
            maxtaskno=0,
            num_states=params.num_states,
            num_actions=params.num_actions,
        )

    def _agent(self, seed):
        return PPOAgent(
            params.num_states,
            params.num_actions,
            params.hidden_layers_ppo,
            activation=params.af_ppo,
            minibatch_seed=seed,
        )

    def test_same_arrival_seed_reproduces_sequence(self):
        first = self._loop()
        second = self._loop()
        first_samples = [first._sample_interarrival_time() for _ in range(20)]
        second_samples = [second._sample_interarrival_time() for _ in range(20)]
        np.testing.assert_array_equal(first_samples, second_samples)

    def test_global_numpy_rng_does_not_change_arrivals(self):
        first = self._loop()
        second = self._loop()
        expected = [first._sample_interarrival_time() for _ in range(20)]

        np.random.random(1000)
        np.random.shuffle(np.arange(1000))
        actual = [second._sample_interarrival_time() for _ in range(20)]

        np.testing.assert_array_equal(expected, actual)

    def test_ppo_shuffle_does_not_change_arrivals(self):
        first = self._loop()
        second = self._loop()
        agent = self._agent(params.PPO_MINIBATCH_SEED)
        expected = [first._sample_interarrival_time() for _ in range(20)]

        actual = []
        for _ in range(20):
            agent._shuffled_indices(128)
            actual.append(second._sample_interarrival_time())

        np.testing.assert_array_equal(expected, actual)

    def test_minibatch_rng_reproducibility(self):
        first = self._agent(params.PPO_MINIBATCH_SEED)
        second = self._agent(params.PPO_MINIBATCH_SEED)
        for _ in range(5):
            np.testing.assert_array_equal(
                first._shuffled_indices(128),
                second._shuffled_indices(128),
            )

    def test_different_minibatch_seeds_change_permutation(self):
        first = self._agent(params.PPO_MINIBATCH_SEED)
        second = self._agent(params.PPO_MINIBATCH_SEED + 1)
        self.assertFalse(
            np.array_equal(first._shuffled_indices(128), second._shuffled_indices(128))
        )

    def test_minibatch_shuffle_does_not_consume_global_numpy_rng(self):
        agent = self._agent(params.PPO_MINIBATCH_SEED)
        np.random.seed(314159)
        expected = np.random.random(20)

        np.random.seed(314159)
        agent._shuffled_indices(128)
        actual = np.random.random(20)

        np.testing.assert_array_equal(expected, actual)

    def test_matched_environment_streams_are_isolated(self):
        first = self._loop()
        second = self._loop()
        agent = self._agent(params.PPO_MINIBATCH_SEED)
        first_arrivals = []
        second_arrivals = []
        first_spatial = []
        second_spatial = []
        for _ in range(10):
            first_arrivals.append(first._sample_interarrival_time())
            first_spatial.append(first.spatial_risk_rng.normal())
            agent._shuffled_indices(64)
            second_arrivals.append(second._sample_interarrival_time())
            second_spatial.append(second.spatial_risk_rng.normal())
        np.testing.assert_array_equal(first_arrivals, second_arrivals)
        np.testing.assert_array_equal(first_spatial, second_spatial)

    def test_configuration_and_dimensions_remain_fixed(self):
        self.assertEqual(params.TASK_ARRIVAL_SEED, 2027)
        self.assertEqual(params.PPO_MINIBATCH_SEED, 2028)
        self.assertEqual(params.SPATIAL_RISK_SEED, 2026)
        self.assertEqual(params.num_states, 35)
        self.assertEqual(params.num_actions, 28)


if __name__ == "__main__":
    unittest.main()
