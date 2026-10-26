"""TORE selection guarantees and paired single/dual-replica integration."""

from itertools import combinations
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import simpy

from config.params import params
from core.env_state import EnvironmentState, select_latency_greedy_action
from core.main_loop import MainLoop
from core.server import Server
from core.task import Task
from core.tore import select_tore_action
from io_utils.save_parameters_and_logs import summarize_task_results


PAIRS = list(combinations(range(1, params.NUM_SERVERS + 1), 2))


def observation(times, deadline=5.):
    """Construct observable work with known completion times, without mocking the predictor."""
    n = params.NUM_SERVERS
    service = params.TASK_DEMAND_RANGE[0] / params.SERVER_PROCESSING_FREQ_RANGE[1]
    return np.r_[np.zeros(n), np.ones(n), np.ones(n), np.asarray(times) - service,
                 np.zeros(n), 0., 0., deadline]


def independent_pairs(marginals):
    return np.array([1 - (1 - marginals[a-1]) * (1 - marginals[b-1]) for a, b in PAIRS])


class TORESelectionTests(unittest.TestCase):
    def test_cap_failure_does_not_search_other_feasible_pairs(self):
        marginal = np.array([.8, .8, .99, .99, .8, .8, .8, .8])
        result = select_tore_action(observation(range(1, 9)), PAIRS, marginal,
                                    independent_pairs(marginal), .999, 5.)
        self.assertGreater(independent_pairs(marginal)[PAIRS.index((3, 4))], .999)
        self.assertEqual(result.greedy_server_ids, (1, 2))
        self.assertAlmostEqual(result.greedy_reliability, .96)
        self.assertTrue(result.is_fallback)

    def test_node_qos_ties_use_each_nodes_completion_time(self):
        state = observation([1., 8., 2., 9., 10., 11., 12., 13.])
        marginal = np.full(8, .9)
        result = select_tore_action(state, PAIRS, marginal, independent_pairs(marginal), .98, 5.)
        self.assertEqual(result.greedy_server_ids, (1, 3))
        self.assertEqual(result.server_ids, (1, 3))

    def test_first_completion_breaks_qos_plateau_before_fixed_order(self):
        state = observation([3., 4., 2., 1., 5., 6., 7., 8.], deadline=10.)
        marginal = np.full(8, .9)
        result = select_tore_action(state, PAIRS, marginal, independent_pairs(marginal), .98, 10.)
        self.assertEqual(result.greedy_server_ids, (4, 3))
        self.assertEqual(result.server_ids, (3, 4))
        self.assertEqual(result.predicted_latency_s, 1.)

    def test_one_replica_stops_selection_when_reliability_suffices(self):
        marginal = np.full(8, .9)
        marginal[0] = .9995
        result = select_tore_action(observation(range(1, 9)), PAIRS, marginal,
                                    independent_pairs(marginal), .999, 5.)
        self.assertEqual(result.server_ids, (1,))
        self.assertIsNone(result.action_index)
        self.assertFalse(result.is_fallback)

    def test_lower_ranked_feasible_single_node_is_added_as_second_replica(self):
        marginal = np.full(8, .9)
        marginal[1] = .9995
        result = select_tore_action(observation(range(1, 9), deadline=1.5), PAIRS, marginal,
                                    independent_pairs(marginal), .999, 1.5)
        # Original incremental logic retains node 1 and adds node 2, even
        # though node 2 could have satisfied the requirement on its own.
        self.assertGreater(independent_pairs(marginal)[0], .999)
        self.assertEqual(result.server_ids, (1, 2))
        self.assertEqual(result.predicted_qos, 1.)
        self.assertFalse(result.is_fallback)

    def test_infeasible_search_uses_existing_unmasked_sml_fallback(self):
        state = observation([1., 8., 2., 9., 10., 11., 12., 13.])
        marginal = np.full(8, .9)
        result = select_tore_action(state, PAIRS, marginal, independent_pairs(marginal), 1., 5.)
        self.assertTrue(result.is_fallback)
        self.assertEqual(result.action_index, select_latency_greedy_action(state, PAIRS))
        self.assertEqual(result.server_ids, (1, 3))

    def test_stop_after_current_prefix_meets_requirement_without_visiting_third_node(self):
        marginal = np.array([.98, .98, .9995, .9, .9, .9, .9, .9])
        result = select_tore_action(observation(range(1, 9)), PAIRS, marginal,
                                    independent_pairs(marginal), .999, 5.)
        self.assertEqual(result.server_ids, (1, 2))
        self.assertAlmostEqual(result.independent_reliability, .9996)
        self.assertFalse(result.is_fallback)

    def test_equal_node_scores_and_completion_times_use_fixed_ids(self):
        marginal = np.full(8, .98)
        result = select_tore_action(observation(np.ones(8)), PAIRS, marginal,
                                    independent_pairs(marginal), .999, 5.)
        self.assertEqual(result.greedy_server_ids, (1, 2))


class TOREIntegrationTests(unittest.TestCase):
    def test_requirement_override_rebuilds_cache_without_mutating_original_tasks(self):
        from compare_tore import cache_evaluation_profiles
        profiles = {sid: dict(Site_ID=str(sid), Processing_Frequency=4e9,
                             Base_Failure_Rate=.01, Transmission_Rate=20.,
                             Latitude=0., Longitude=sid/100.)
                    for sid in range(1, 9)}
        stale = (np.zeros(8), np.zeros(28), np.zeros(28))
        original_mask = np.zeros(28, dtype=bool)
        tasks = {1: dict(Task_Size=5., Computation_Demand=2e9, Deadline=3.,
                         Reliability_Requirement=.9, reliability_snapshot=stale, safe_mask=original_mask)}
        effective = cache_evaluation_profiles(profiles, tasks, failure_rate_scale=1., ell_ratio=1.,
                                             reliability_requirement=.999)
        self.assertEqual(effective[1]['Reliability_Requirement'], .999)
        self.assertEqual(tasks[1]['Reliability_Requirement'], .9)
        self.assertIs(tasks[1]['reliability_snapshot'], stale)
        self.assertIs(tasks[1]['safe_mask'], original_mask)
        self.assertTrue(effective[1]['safe_mask'].any())
        np.testing.assert_array_equal(effective[1]['safe_mask'],
                                      effective[1]['reliability_snapshot'][1] >= .999)

    def test_all_single_replica_episode_summarizes_without_missing_values_as_work(self):
        from compare_ppo_greedy import masked_evaluation
        from compare_tore import cache_evaluation_profiles
        profiles = {sid: dict(Site_ID=str(sid), Processing_Frequency=4e9,
                             Base_Failure_Rate=.01, Transmission_Rate=20.,
                             Latitude=0., Longitude=sid/100.)
                    for sid in range(1, 9)}
        tasks = {tid: dict(Task_Size=5., Computation_Demand=2e9,
                           Reliability_Requirement=.9999, Deadline=3.) for tid in (1, 2)}
        tasks = cache_evaluation_profiles(profiles, tasks, failure_rate_scale=0., ell_ratio=0.)
        context = SimpleNamespace(this_episode=1, use_safety_mask=True, num_actions=params.num_actions)
        records = []
        with patch.object(params, 'taskno', 2):
            values, loads = masked_evaluation('TORE', None, profiles, tasks, np.full(2, .1), 1, 1,
                lambda task: MainLoop.get_task_outcome_info(context, task), task_records=records,
                failure_rate_scale=0., ell_ratio=0.)
        self.assertEqual(values['theoretical_rsr'], 1.)
        self.assertEqual(values['fallback_count'], 0)
        self.assertTrue(all(row['server_B_id'] is None for row in records))
        self.assertEqual(loads.replica_count.sum(), 2)
        self.assertAlmostEqual(loads.cpu_busy_s.sum(), 1.)

    def test_single_replica_has_no_phantom_execution(self):
        env = simpy.Environment()
        state = EnvironmentState()
        server = Server(env, 1, 'a', 2e9, .01, 5, 0, 0)
        state.add_server_and_init_environment(server)
        task = Task(env, state, 1, {'Task_Size': 5., 'Computation_Demand': 2e9,
                                   'Reliability_Requirement': .9, 'Deadline': 3.})
        env.process(task.execute_task(server))
        env.run()
        self.assertEqual(len(task.active_replicas), 1)
        self.assertIsNone(task.replica_B['server_id'])
        self.assertIsNone(task.replica_B['finish_time'])
        self.assertEqual(task.task_completion_time, 2.)

    def test_cached_correlated_mask_does_not_control_tore_and_actual_work_reconciles(self):
        from compare_ppo_greedy import masked_evaluation
        profiles = {sid: dict(Site_ID=str(sid), Processing_Frequency=4e9,
                             Base_Failure_Rate=.01, Transmission_Rate=20.,
                             Latitude=0., Longitude=sid/100.)
                    for sid in range(1, 9)}
        tasks = {}
        for tid, (r, corr) in enumerate(((.9995, .99999), (.99, .998), (.8, .95)), start=1):
            marginal = np.full(8, r)
            correlated = np.full(28, corr)
            mask = correlated >= .999
            for value in (marginal, correlated, mask):
                value.setflags(write=False)
            tasks[tid] = dict(Task_Size=5., Computation_Demand=2e9,
                Reliability_Requirement=.999, Deadline=3.,
                reliability_snapshot=(marginal, correlated, independent_pairs(marginal)), safe_mask=mask)
        context = SimpleNamespace(this_episode=1, use_safety_mask=True, num_actions=params.num_actions)
        records = []
        original_masks = [p['safe_mask'].copy() for p in tasks.values()]
        with patch.object(params, 'taskno', 3):
            values, loads = masked_evaluation('TORE', None, profiles, tasks, np.ones(3), 1, 1,
                lambda task: MainLoop.get_task_outcome_info(context, task), task_records=records)
        data = pd.DataFrame(records)
        self.assertIsNone(records[0]['server_B_id'])
        self.assertFalse(records[1]['is_fallback'])
        self.assertTrue(records[1]['selection_requirement_satisfied'])
        self.assertFalse(records[1]['requirement_satisfied'])
        self.assertTrue(records[2]['is_fallback'])
        self.assertEqual(values['fallback_count'], 1)
        self.assertEqual(loads.replica_count.sum(), 5)
        self.assertEqual(loads.execution_cycles.sum(), 5*2e9)
        self.assertAlmostEqual(loads.cpu_busy_s.sum(), 5*.5)
        episodes, _, _ = summarize_task_results(records)
        self.assertAlmostEqual(episodes.theoretical_rsr.iloc[0], 1/3)
        for original, profile in zip(original_masks, tasks.values()):
            np.testing.assert_array_equal(original, profile['safe_mask'])

    def test_fallback_can_occur_with_nonempty_independent_safe_set(self):
        from compare_ppo_greedy import masked_evaluation
        profiles = {sid: dict(Site_ID=str(sid), Processing_Frequency=4e9,
                             Base_Failure_Rate=.01, Transmission_Rate=20.,
                             Latitude=0., Longitude=sid/100.)
                    for sid in range(1, 9)}
        marginal = np.array([.8, .8, .99, .99, .8, .8, .8, .8])
        independent = independent_pairs(marginal)
        tasks = {1: dict(Task_Size=5., Computation_Demand=2e9,
            Reliability_Requirement=.999, Deadline=3.,
            reliability_snapshot=(marginal, independent.copy(), independent), safe_mask=independent >= .999)}
        context = SimpleNamespace(this_episode=1, use_safety_mask=True, num_actions=params.num_actions)
        records = []
        with patch.object(params, 'taskno', 1):
            values, _ = masked_evaluation('TORE', None, profiles, tasks, np.ones(1), 1, 1,
                lambda task: MainLoop.get_task_outcome_info(context, task), task_records=records)
        self.assertEqual(values['fallback_count'], 1)
        self.assertEqual(values['normal_decision_count'], 0)
        self.assertTrue(records[0]['independent_feasible_pair_exists'])
        self.assertTrue(records[0]['greedy_missed_feasible_pair'])
        self.assertEqual(records[0]['fallback_reason'], 'replica_limit')


if __name__ == '__main__':
    unittest.main()
