import io
import math
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import simpy

from agents.rasoul_delay_agent import RasoulDelayAgent
from config.params import params
from config.paths import EVALUATION_TASK_PATH
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.spatial_risk import GaussHermiteReliability
from core.task import Task
from tools.run_experiments import temporary_params


class RasoulDelayAgentTests(unittest.TestCase):
    def test_unobserved_priority_and_completed_feedback_change_selection(self):
        agent = RasoulDelayAgent([1, 2], exploration_c=0)
        bases = {1: 1, 2: 2}
        self.assertEqual(agent.select_action(bases, [1, 2]), 1)
        # Allocation without completion does not create an observation.
        self.assertEqual(agent.select_action(bases, [1, 2]), 1)
        self.assertEqual(agent.completed_counts, {1: 0, 2: 0})
        agent.observe(1, actual_delay=11, base_delay=1)
        self.assertEqual(agent.select_action(bases, [1, 2]), 2)
        agent.observe(2, actual_delay=2, base_delay=2)
        # Both are observed: historical discrepancy defeats basic-delay greedy.
        self.assertEqual(agent.select_action(bases, [1, 2]), 2)
        self.assertEqual(agent.decision_count, 4)

    def test_lcb_formula_negative_predictions_and_reset(self):
        agent = RasoulDelayAgent([1], exploration_c=10)
        agent.observe(1, actual_delay=3, base_delay=1)
        agent.observe(1, actual_delay=1, base_delay=1)
        agent.select_action({1: 1}, [1])
        agent.select_action({1: 1}, [1])
        expected = 1 + 1 - 10 * math.sqrt(2 * math.log(2) / 2)
        self.assertAlmostEqual(agent.predict_delay(1, 1), expected)
        self.assertLess(agent.predict_delay(1, 1), 0)
        self.assertEqual(agent.discrepancy_sums[1], 2)
        agent.reset()
        self.assertEqual(agent.completed_counts, {1: 0})
        self.assertEqual(agent.discrepancy_sums, {1: 0})
        self.assertEqual(agent.decision_count, 0)

    def test_ties_and_single_allowed_server(self):
        agent = RasoulDelayAgent([1, 2, 3], exploration_c=0)
        self.assertEqual(agent.select_action({1: 3, 2: 1, 3: 1}, [3, 2, 1]), 2)
        agent.observe(1, 2, 1)
        agent.observe(2, 2, 2)
        self.assertEqual(agent.select_action({1: 1, 2: 2}, [2, 1]), 1)
        self.assertEqual(agent.select_action({3: 100}, [3]), 3)

    def test_fallback_never_scores_and_counts_decisions(self):
        agent = RasoulDelayAgent([1, 2])
        with patch.object(agent, "predict_delay", side_effect=AssertionError("Fallback must not score")):
            self.assertEqual(agent.select_action({1: 2, 2: 1}, [1, 2], fallback=True), 2)
            self.assertEqual(agent.select_action({1: 1, 2: 1}, [2, 1], fallback=True), 1)
        self.assertEqual(agent.decision_count, 2)
        self.assertEqual(sum(agent.completed_counts.values()), 0)

    def test_invalid_exploration_coefficients(self):
        for coefficient in (-1, float("inf"), float("nan")):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                RasoulDelayAgent([1], coefficient)

    def test_marginal_matches_pair_gh_for_all_correlations(self):
        for beta in (0, .5, 1):
            gh = GaussHermiteReliability(beta, 32)
            for exposure in (0, .01, .1, 1):
                for rho in (0, .5, 1):
                    pair = gh.compute_pair(exposure, .2, rho)
                    self.assertEqual(gh.compute_marginal(exposure), pair.replica_a)
                    self.assertEqual(gh.compute_marginal(.2), pair.replica_b)
        for exposure in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                gh.compute_marginal(exposure)


class RasoulSimulationTests(unittest.TestCase):
    def make_loop(self, episodes=1, count=4):
        from Project_main import build_model

        loop = MainLoop(build_model(device="cpu"), episodes, count, params.num_states,
                        params.num_actions, task_parameters_path=EVALUATION_TASK_PATH)
        return loop

    @staticmethod
    def task(loop, task_id=1):
        return Task(loop.env, loop.env_state, task_id, dict(
            Task_Size=1, Computation_Demand=1, Reliability_Requirement=.95,
        ))

    def test_safe_set_and_maximum_marginal_fallback_ties(self):
        with temporary_params(model_summary="RASOUL-Delay"), redirect_stdout(io.StringIO()):
            loop = self.make_loop()
            self.assertEqual(loop.num_actions, 8)
            loop.env, loop.env_state = simpy.Environment(), EnvironmentState()
            loop.setServers()
            for server_id in loop.index_of_actions:
                server = loop.env_state.get_server_by_id(server_id)
                server.transmission_rate = server.processing_frequency = 1
                loop.marginal_reliability_cache[(1, server_id)] = .80
            loop.marginal_reliability_cache[(1, 3)] = .95
            task = self.task(loop)
            with patch.object(loop.env_state, "get_server_backlog_components", side_effect=AssertionError):
                self.assertEqual(loop._select_rasoul_server(task).server_id, 3)
            self.assertEqual(task.feasible_action_count, 1)
            self.assertFalse(task.mask_fallback_used)
            task.reliability_requirement = .99
            loop.marginal_reliability_cache[(1, 2)] = .95
            with patch.object(loop.model, "predict_delay", side_effect=AssertionError):
                self.assertEqual(loop._select_rasoul_server(task).server_id, 2)
                loop.env_state.get_server_by_id(3).transmission_rate = 2
                self.assertEqual(loop._select_rasoul_server(task).server_id, 3)
            self.assertTrue(task.mask_fallback_used)
            self.assertEqual(task.feasible_action_count, 0)

    def test_one_execution_for_single_two_for_dual_and_loser_continues(self):
        for single in (True, False):
            with self.subTest(single=single), temporary_params(model_summary="RASOUL-Delay"), \
                    redirect_stdout(io.StringIO()):
                loop = self.make_loop()
                loop.env, loop.env_state = simpy.Environment(), EnvironmentState()
                loop.setServers()
                first, second = (loop.env_state.get_server_by_id(i) for i in (1, 2))
                first.transmission_rate = second.transmission_rate = 1
                first.processing_frequency, second.processing_frequency = 1, .5
                task = self.task(loop)
                with patch.object(loop.env_state, "register_waiting_replica",
                                  wraps=loop.env_state.register_waiting_replica) as queue, \
                        patch.object(loop.env_state, "start_replica_execution",
                                     wraps=loop.env_state.start_replica_execution) as execute, \
                        patch.object(task.task_completion_event, "succeed",
                                     wraps=task.task_completion_event.succeed) as complete:
                    loop.env.process(task.execute_task(first, None if single else second))
                    loop.env.run(until=task.task_completion_event)
                    self.assertIsNone(task.replica_B["finish_time"])
                    loop.env.run()
                self.assertEqual(queue.call_count, 1 if single else 2)
                self.assertEqual(execute.call_count, 1 if single else 2)
                self.assertEqual(complete.call_count, 1)
                self.assertEqual(task.task_completion_time, 2)
                if single:
                    self.assertTrue(all(value is None for value in task.replica_B.values()))
                else:
                    self.assertEqual(task.replica_B["finish_time"], 3)

    def test_simultaneous_completions_update_once_and_never_early(self):
        with temporary_params(model_summary="RASOUL-Delay"), redirect_stdout(io.StringIO()):
            loop = self.make_loop()
            loop.env, loop.env_state = simpy.Environment(), EnvironmentState()
            loop.setServers()
            for server_id in (1, 2):
                server = loop.env_state.get_server_by_id(server_id)
                server.transmission_rate = server.processing_frequency = 1
                task = self.task(loop, server_id)
                task.base_delay_at_decision, task.replica_A_reliability = 2, .9
                loop.env_state.add_task(task)
                loop.pendingList.append(task.id)
                loop.env.process(task.execute_task(server))
                loop.env.process(loop._observe_rasoul_completion(task))
            with patch.object(loop.model, "observe", wraps=loop.model.observe) as observe:
                loop.env.run(until=1.9)
                self.assertEqual(observe.call_count, 0)
                for task in loop.env_state.tasks.values():
                    loop._update_rasoul_feedback(task)
                self.assertEqual(observe.call_count, 0)
                loop.env.run()
                self.assertEqual(observe.call_count, 2)
                loop._collect_resolved_task_outcomes()
                loop._collect_resolved_task_outcomes()
                for task in loop.env_state.tasks.values():
                    loop._update_rasoul_feedback(task)
                self.assertEqual(observe.call_count, 2)
            self.assertEqual(loop.pendingList, [])
            self.assertEqual(loop.model.completed_counts[1], 1)
            self.assertEqual(loop.model.completed_counts[2], 1)

    def test_online_learning_changes_real_choices_and_episode_reset(self):
        with temporary_params(model_summary="RASOUL-Delay", NUM_SERVERS=2,
                              RASOUL_DELAY_EXPLORATION_C=0), redirect_stdout(io.StringIO()):
            servers = {key: dict(Server_ID=key, Site_ID=str(key), Processing_Frequency=1,
                                 Transmission_Rate=1, Base_Failure_Rate=0, Latitude=0, Longitude=key)
                       for key in (1, 2)}
            tasks = {key: dict(Task_Size=1, Computation_Demand=1,
                               Reliability_Requirement=.9) for key in range(1, 5)}
            with patch.object(MainLoop, "_load_run_profiles", return_value=(servers, tasks)):
                loop = self.make_loop(episodes=2)
            arrivals = iter([0, .1, 2.1, 2.1] * 2)
            loop._sample_interarrival_time = lambda: next(arrivals)
            snapshots = []
            loop.episode_end_callback = lambda _: snapshots.append(dict(loop.model.completed_counts))
            with patch.object(EnvironmentState, "get_state", side_effect=AssertionError("No queue observation")):
                loop.EP()
            self.assertEqual(loop.num_actions, 2)
            self.assertEqual(snapshots, [{1: 2, 2: 2}] * 2)
            self.assertEqual(loop.model.decision_count, 4)
            for episode in (1, 2):
                rows = [row for row in loop.task_results if row["episode"] == episode]
                self.assertEqual([row["server_A_id"] for row in rows], [1, 1, 2, 2])
                self.assertEqual([row["observations_at_decision"] for row in rows], [0, 0, 0, 1])
                self.assertAlmostEqual(rows[1]["delay_discrepancy"], .9)
                self.assertTrue(all(row["requirement_satisfied"] for row in rows))
                self.assertTrue(all(row["task_delay"] >= row["base_delay_at_decision"] for row in rows))
            self.assertAlmostEqual(loop.model.discrepancy_sums[1], .9)

    def test_all_fallback_uses_controlled_profiles(self):
        with temporary_params(model_summary="RASOUL-Delay", NUM_SERVERS=2,
                              SPATIAL_RISK_BETA_P=.5, SPATIAL_RISK_GH_ORDER=32), \
                redirect_stdout(io.StringIO()):
            # Server 1 is faster; server 2 is more reliable. Neither meets .99.
            servers = {key: dict(Server_ID=key, Site_ID=str(key), Processing_Frequency=1,
                                 Transmission_Rate=2 if key == 1 else 1,
                                 Base_Failure_Rate=.1 if key == 1 else .05,
                                 Latitude=0, Longitude=key) for key in (1, 2)}
            tasks = {key: dict(Task_Size=1, Computation_Demand=1,
                               Reliability_Requirement=.99) for key in range(1, 5)}
            with patch.object(MainLoop, "_load_run_profiles", return_value=(servers, tasks)):
                loop = self.make_loop(episodes=2)
            with patch.object(loop.model, "predict_delay", side_effect=AssertionError("Fallback must not score")):
                loop.EP()
            self.assertEqual(len(loop.task_results), 8)
            self.assertTrue(all(row["server_A_id"] == 2 and row["mask_fallback_used"]
                                and row["feasible_action_count"] == 0
                                and not row["requirement_satisfied"] for row in loop.task_results))
            self.assertEqual(loop.model.completed_counts, {1: 0, 2: 4})
            for _, _, _, rsr_corr, rsr_ind, fallback in loop.log_data:
                self.assertEqual((rsr_corr, rsr_ind, fallback), (0, 0, 1))

    def test_profile_smoke_checks_safe_sets_and_preserves_arrivals(self):
        for requirement in (None, .95):
            with self.subTest(requirement=requirement), \
                    temporary_params(model_summary="RASOUL-Delay", EXPERIMENT_SEED=2027), \
                    redirect_stdout(io.StringIO()):
                loop = self.make_loop(episodes=2)
                loop.maxTask = min(params.taskno, len(loop.task_profiles))
                if requirement is not None:
                    # Change only the in-memory test workload, never the workbook.
                    for profile in loop.task_profiles.values():
                        profile["Reliability_Requirement"] = requirement
                loop.EP()
                self.assert_profile_outcomes(loop)

    def assert_profile_outcomes(self, loop):
        self.assertEqual(len(loop.task_results), loop.total_episodes * loop.maxTask)
        self.assertEqual(loop.pendingList, [])
        self.assertEqual(sum(loop.model.completed_counts.values()), loop.maxTask)
        expected_arrivals = np.cumsum(np.random.default_rng(2027).exponential(
            1 / params.TASK_ARRIVAL_RATE, (loop.total_episodes, loop.maxTask)), axis=1).ravel()
        np.testing.assert_array_equal([row["arrival_time"] for row in loop.task_results], expected_arrivals)
        for row in loop.task_results:
            # The established pair integrator supplies the marginal reference.
            reliabilities = {
                server: loop.reliability_integrator.compute_pair(
                    profile["Base_Failure_Rate"] * row["computation_demand"] / profile["Processing_Frequency"],
                    0, 0,
                ).replica_a for server, profile in loop.server_profiles.items()
            }
            safe = [server for server, reliability in reliabilities.items()
                    if reliability >= row["reliability_requirement"]]
            self.assertEqual(row["feasible_action_count"], len(safe))
            self.assertEqual(row["mask_fallback_used"], not safe)
            if safe:
                self.assertIn(row["server_A_id"], safe)
            else:
                expected = min(loop.server_profiles, key=lambda server: (
                    -reliabilities[server],
                    row["task_size"] / loop.server_profiles[server]["Transmission_Rate"]
                    + row["computation_demand"] / loop.server_profiles[server]["Processing_Frequency"],
                    server,
                ))
                self.assertEqual(row["server_A_id"], expected)
            self.assertEqual(row["task_reliability"], reliabilities[row["server_A_id"]])
            self.assertEqual(row["requirement_satisfied"],
                             row["task_reliability"] >= row["reliability_requirement"])
            self.assertIsNone(row["server_B_id"])
            self.assertIsNone(row["pair_reliability_correlated"])
            self.assertIsNone(row["pair_reliability_independent"])
            self.assertEqual(row["task_reliability"], row["replica_A_reliability"])
        for task in loop.env_state.tasks.values():
            self.assertTrue(task.rasoul_feedback_processed)
        for episode, _, _, rsr_corr, rsr_ind, fallback in loop.log_data:
            rows = [row for row in loop.task_results if row["episode"] == episode]
            expected_rsr = sum(row["requirement_satisfied"] for row in rows) / len(rows)
            expected_fallback = sum(row["mask_fallback_used"] for row in rows) / len(rows)
            self.assertEqual(rsr_corr, expected_rsr)
            self.assertEqual(rsr_ind, expected_rsr)
            self.assertEqual(fallback, expected_fallback)


if __name__ == "__main__":
    unittest.main()
