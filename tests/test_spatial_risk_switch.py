"""The runtime switch controls both live and precomputed reliabilities."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from config.params import params
from compare_ppo_greedy import make_environment, precompute_masked_profiles, MASKED_ALPHA
from compare_tore import cache_evaluation_profiles
from core.spatial_risk import compute_marginal_failure_probability, compute_pair_reliability_gh
from core.task import Task


class SpatialRiskSwitchTests(unittest.TestCase):
    def setUp(self):
        self.profiles = {sid: dict(Site_ID=str(sid), Processing_Frequency=(1+sid/4)*1e9,
            Base_Failure_Rate=.1+sid/25, Transmission_Rate=10., Latitude=0., Longitude=sid/100.)
            for sid in range(1, 9)}
        self.profile = dict(Task_Size=5., Computation_Demand=3e9,
                            Reliability_Requirement=.99, Deadline=3.)

    def test_disabled_matches_base_rate_formula_in_both_selection_models(self):
        for model in ('correlated', 'independent'):
            with self.subTest(model=model), patch.object(params, 'SPATIAL_RISK_ENABLED', False):
                env, state = make_environment(self.profiles, failure_rate_scale=.2,
                                               reliability_model=model)
                task = Task(env, state, 1, self.profile)
                marginal, corr, ind = state.compute_task_reliabilities(task)
                expected = np.array([np.exp(-.2*p['Base_Failure_Rate']*3e9/p['Processing_Frequency'])
                                     for p in self.profiles.values()])
                pair_expected = np.array([1-(1-expected[a-1])*(1-expected[b-1])
                                          for a, b in state.server_pairs])
                np.testing.assert_allclose(marginal, expected, rtol=0, atol=1e-14)
                np.testing.assert_allclose(corr, pair_expected, rtol=0, atol=1e-14)
                np.testing.assert_allclose(ind, pair_expected, rtol=0, atol=1e-14)
                np.testing.assert_array_equal(state.get_safe_action_mask(task), pair_expected >= .99)

    def test_enabled_preserves_original_gh_computation(self):
        with patch.object(params, 'SPATIAL_RISK_ENABLED', True):
            env, state = make_environment(self.profiles)
            task = Task(env, state, 1, self.profile)
            actual = state.compute_task_reliabilities(task)
            frequencies = np.array([p['Processing_Frequency'] for p in self.profiles.values()])
            rates = np.array([p['Base_Failure_Rate'] for p in self.profiles.values()])
            pairs = np.array(state.server_pairs)-1
            expected_marginal = 1-compute_marginal_failure_probability(3e9, frequencies, rates,
                                                                      params.SPATIAL_RISK_BETA_P)
            expected_corr, expected_ind = compute_pair_reliability_gh(3e9, frequencies[pairs], rates[pairs],
                state.spatial_correlation_matrix[pairs[:, 0], pairs[:, 1]], params.SPATIAL_RISK_BETA_P)
            for observed, expected in zip(actual, (expected_marginal, expected_corr, expected_ind)):
                np.testing.assert_array_equal(observed, expected)
            self.assertTrue(np.any(actual[1] < actual[2]-1e-6))

    def test_disabled_four_method_cache_rebuild_and_batched_cache_agree(self):
        tasks = {tid: dict(self.profile) for tid in range(1, 201)}
        with patch.object(params, 'SPATIAL_RISK_ENABLED', True):
            old = cache_evaluation_profiles(self.profiles, tasks,
                failure_rate_scale=MASKED_ALPHA, ell_ratio=1.)
        with patch.object(params, 'SPATIAL_RISK_ENABLED', False), tempfile.TemporaryDirectory() as directory:
            rebuilt = cache_evaluation_profiles(self.profiles, old,
                failure_rate_scale=MASKED_ALPHA, ell_ratio=1.)
            batch = precompute_masked_profiles(self.profiles, {'train': tasks}, Path(directory))
            self.assertFalse(np.array_equal(old[1]['reliability_snapshot'][0],
                                           rebuilt[1]['reliability_snapshot'][0]))
            for threshold, profiles in batch['train'].items():
                for tid, profile in profiles.items():
                    for actual, expected in zip(profile['reliability_snapshot'], rebuilt[tid]['reliability_snapshot']):
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
                    np.testing.assert_array_equal(profile['safe_mask'],
                                                   rebuilt[tid]['reliability_snapshot'][1] >= threshold)


if __name__ == '__main__':
    unittest.main()
