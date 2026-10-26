"""RASOUL selection, causal feedback, admission and warmup/reset integration."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import simpy

from config.params import params
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.rasoul import RASOULPolicy, single_core_loads
from core.server import Server
from core.task import Task


def environment():
    env, state = simpy.Environment(), EnvironmentState()
    for sid in range(1, 9):
        state.add_server_and_init_environment(Server(env, sid, str(sid), 4e9, .01, 20., 0., sid/100.))
    state.initialize_spatial_reliability()
    return env, state


def task_at(env, state, tid=1, *, marginal=None, deadline=3., size=5., demand=2e9):
    if marginal is None:
        marginal = np.full(8, .9999)
    return Task(env, state, tid, dict(Task_Size=size, Computation_Demand=demand,
        Reliability_Requirement=.999, Deadline=deadline,
        reliability_snapshot=(np.asarray(marginal), np.ones(28), np.ones(28))))


def learned_policy(counts, errors):
    return RASOULPolicy.from_snapshot(dict(server_ids=list(range(1, 9)),
        decision_count=sum(counts), feedback_counts=counts, error_sums=errors))


class RASOULSelectionTests(unittest.TestCase):
    def test_history_correction_changes_choice_and_exploration_score_is_not_clipped(self):
        env, state = environment()
        task = task_at(env, state)
        cold = RASOULPolicy(range(1, 9)).select(state.get_state(task), task)
        self.assertEqual(cold.server_id, 1)
        policy = learned_policy([100]*8, [40.] + [0.]*7)
        decision = policy.select(state.get_state(task), task)
        self.assertEqual(decision.server_id, 2)
        self.assertAlmostEqual(decision.confidence_bonus, np.sqrt(2*np.log(801)/100))
        self.assertGreater(decision.selection_score, 1.)
        underobserved = learned_policy([100, 1] + [100]*6, [0.]*8)
        self.assertEqual(underobserved.select(state.get_state(task), task).server_id, 2)

    def test_cold_start_priority_stays_within_reliability_and_admission(self):
        env, state = environment()
        task = task_at(env, state, marginal=[.9999, .9999] + [.9]*6)
        state.servers[2]['waiting_replicas'].append(dict(task=task, service_time=4.))
        policy = learned_policy([100] + [0]*7, [0.]*8)
        decision = policy.select(state.get_state(task), task)
        self.assertEqual(decision.server_id, 1)
        self.assertFalse(decision.is_fallback)
        self.assertEqual(sum(decision.safe_node_mask), 2)
        self.assertEqual(sum(decision.admitted_node_mask), 1)

    def test_admission_fallback_chooses_a_reliable_node_even_if_unsafe_node_is_faster(self):
        env, state = environment()
        task = task_at(env, state, marginal=[.9999] + [.9]*7)
        state.servers[1]['waiting_replicas'].append(dict(task=task, service_time=4.))
        decision = RASOULPolicy(range(1, 9)).select(state.get_state(task), task)
        self.assertEqual(decision.fallback_reason, 'admission')
        self.assertEqual(decision.server_id, 1)
        self.assertTrue(decision.safe_node_mask[0])
        self.assertFalse(any(decision.admitted_node_mask))

    def test_reliability_fallback_uses_fastest_node_and_fixed_id_tie(self):
        env, state = environment()
        task = task_at(env, state, marginal=[.9]*8)
        state.servers[1]['waiting_replicas'].append(dict(task=task, service_time=2.))
        decision = RASOULPolicy(range(1, 9)).select(state.get_state(task), task)
        self.assertEqual(decision.fallback_reason, 'no_reliable_node')
        self.assertEqual(decision.server_id, 2)
        self.assertFalse(any(decision.safe_node_mask))

    def test_admission_accepts_exact_single_core_boundary(self):
        env, state = environment()
        task = task_at(env, state, deadline=.5)
        decision = RASOULPolicy(range(1, 9)).select(state.get_state(task), task)
        self.assertFalse(decision.is_fallback)
        self.assertEqual(decision.selected_load_after_admission, 1.)

    def test_load_uses_full_execution_own_deadlines_and_excludes_uploads(self):
        env, state = environment()
        server = state.get_server_by_id(1)
        first = task_at(env, state, 1, size=20., deadline=1.)
        second = task_at(env, state, 2, size=20., deadline=2.)
        env.process(first.execute_task(server))
        env.process(second.execute_task(server))
        env.run(until=.5)
        self.assertEqual(single_core_loads(state, range(1, 9))[0], 0.)
        env.run(until=1.25)
        self.assertEqual(single_core_loads(state, range(1, 9))[0], .5/1. + .5/2.)
        env.run(until=1.75)
        self.assertEqual(single_core_loads(state, range(1, 9))[0], .5/2.)
        env.run()
        self.assertEqual(single_core_loads(state, range(1, 9))[0], 0.)


class RASOULFeedbackTests(unittest.TestCase):
    def test_feedback_waits_for_completion_uses_raw_prediction_and_is_recorded_once(self):
        env, state = environment()
        # This earlier upload is invisible in the CPU queue at decision time.
        blocker = task_at(env, state, 10, demand=8e9)
        env.process(blocker.execute_task(state.get_server_by_id(1)))
        task = task_at(env, state, marginal=[.9999] + [.9]*7, size=10., deadline=1.)
        policy = learned_policy([1]*8, [0.]*8)
        decision = policy.select(state.get_state(task), task)
        self.assertEqual(decision.predicted_qos, 1.)
        self.assertGreater(decision.selection_score, 1.)
        self.assertEqual(sum(policy.feedback_counts), 8)
        with self.assertRaises(ValueError):
            policy.observe(task)
        with self.assertRaises(ValueError):
            policy.snapshot()
        env.process(task.execute_task(state.get_server_by_id(decision.server_id)))

        def feedback():
            yield task.task_completion_event
            policy.observe(task)

        env.process(feedback())
        env.run(until=.6)
        self.assertEqual(sum(policy.feedback_counts), 8)
        env.run()
        self.assertEqual(task.rasoul_feedback['actual_qos'], 0.)
        self.assertEqual(task.rasoul_feedback['prediction_error'], 1.)
        self.assertEqual(task.rasoul_feedback['feedback_time'], task.task_completion_time)
        self.assertEqual(policy.snapshot()['feedback_counts'][0], 2)
        self.assertEqual(policy.snapshot()['error_sums'][0], 1.)
        with self.assertRaises(ValueError):
            policy.observe(task)

    def test_snapshots_are_independent_and_reject_undrained_or_invalid_statistics(self):
        history = learned_policy([2]*8, [0.]*8).snapshot()
        clone = RASOULPolicy.from_snapshot(history)
        clone.error_sums[0] = -.5
        self.assertEqual(history['error_sums'][0], 0.)
        for field, value in (('decision_count', 17), ('feedback_counts', [-1]*8),
                             ('error_sums', [3.]*8), ('error_sums', [float('nan')]*8)):
            with self.subTest(field=field, value=value):
                invalid = {**history, field: value}
                with self.assertRaises(ValueError):
                    RASOULPolicy.from_snapshot(invalid)


class RASOULIntegrationTests(unittest.TestCase):
    def test_admission_fallback_is_reported_separately_and_still_satisfies_reliability(self):
        from compare_ppo_greedy import masked_evaluation
        profiles = {sid: dict(Site_ID=str(sid), Processing_Frequency=4e9,
                             Base_Failure_Rate=.01, Transmission_Rate=20., Latitude=0., Longitude=sid/100.)
                    for sid in range(1, 9)}
        tasks = {tid: dict(Task_Size=5., Computation_Demand=2e9, Deadline=.6,
            Reliability_Requirement=.999, safe_mask=np.ones(28, dtype=bool),
            reliability_snapshot=(np.array([.9999] + [.9]*7), np.ones(28), np.ones(28)))
            for tid in (1, 2)}
        policy = RASOULPolicy(range(1, 9))
        context = SimpleNamespace(this_episode=1, use_safety_mask=True, num_actions=params.num_actions)
        records = []
        with patch.object(params, 'taskno', 2):
            values, loads = masked_evaluation('RASOUL', None, profiles, tasks, np.array([.1, .6]), 1, 0,
                lambda task: MainLoop.get_task_outcome_info(context, task), task_records=records,
                arrival_rate=1., failure_rate_scale=1., rasoul_policy=policy)
        self.assertEqual(values['theoretical_rsr'], 1.)
        self.assertEqual(values['admission_fallback_count'], 1)
        self.assertEqual(values['reliability_fallback_count'], 0)
        self.assertEqual(records[1]['fallback_reason'], 'admission')
        self.assertTrue(records[1]['requirement_satisfied'])
        self.assertEqual(loads.replica_count.sum(), 2)
        self.assertEqual(sum(policy.snapshot()['feedback_counts']), 2)

    def test_evaluation_uses_one_replica_updates_fallback_feedback_and_replays_from_snapshot(self):
        from compare_ppo_greedy import masked_evaluation
        from compare_tore import cache_evaluation_profiles
        profiles = {sid: dict(Site_ID=str(sid), Processing_Frequency=4e9,
                             Base_Failure_Rate=.01, Transmission_Rate=20., Latitude=0., Longitude=sid/100.)
                    for sid in range(1, 9)}
        tasks = {tid: dict(Task_Size=5., Computation_Demand=2e9, Deadline=3.,
                          Reliability_Requirement=requirement)
                 for tid, requirement in enumerate((.9, .9999), start=1)}
        tasks = cache_evaluation_profiles(profiles, tasks, failure_rate_scale=1., ell_ratio=1.)
        history = learned_policy([10]*8, [0.]*8).snapshot()
        digest = RASOULPolicy.history_digest(history)
        traces = []
        for _ in range(2):
            policy = RASOULPolicy.from_snapshot(history)
            records = []
            context = SimpleNamespace(this_episode=1, use_safety_mask=True, num_actions=params.num_actions)
            with patch.object(params, 'taskno', 2):
                values, loads = masked_evaluation('RASOUL', None, profiles, tasks, np.full(2, .2), 1, 0,
                    lambda task: MainLoop.get_task_outcome_info(context, task), task_records=records,
                    arrival_rate=2., failure_rate_scale=1., ell_ratio=1., rasoul_policy=policy)
            self.assertEqual(values['theoretical_rsr'], .5)
            self.assertEqual(values['reliability_fallback_count'], 1)
            self.assertEqual(loads.replica_count.sum(), 2)
            self.assertTrue(all(row['server_B_id'] is None and row['action_index'] is None for row in records))
            self.assertTrue(all(row['selection_reliability_model'] == 'marginal' for row in records))
            self.assertEqual(sum(policy.snapshot()['feedback_counts']), 82)
            self.assertEqual(RASOULPolicy.history_digest(history), digest)
            traces.append((values['trace_sha256'], policy.snapshot()))
        self.assertEqual(traces[0], traces[1])

    def test_comparison_warms_up_on_training_only_and_resets_each_evaluation_with_four_methods(self):
        import compare_tore
        from compare_ppo_greedy import build_agent
        with tempfile.TemporaryDirectory(prefix='rasoul-comparison-') as root:
            root = Path(root)
            servers = pd.DataFrame([dict(Server_ID=sid, Site_ID=str(sid), Processing_Frequency=4e9,
                Base_Failure_Rate=.01, Transmission_Rate=20., Latitude=0., Longitude=sid/100.)
                for sid in range(1, 9)])
            servers.to_excel(root/'servers.xlsx', sheet_name='Servers', index=False)
            for split, start, size in (('train', 101, 5.), ('eval', 201, 10.)):
                pd.DataFrame([dict(Task_ID=tid, Task_Size=size, Computation_Demand=2e9,
                    Reliability_Requirement=req, Deadline=3.)
                    for tid, req in zip(range(start, start+3), (.9, .999, .9999))]).to_excel(root/f'{split}.xlsx', index=False)
            build_agent().save_model(root/'ppo.pt')
            args = SimpleNamespace(ppo_checkpoint=root/'ppo.pt', device='cpu', failure_rate_scale=1.,
                ell_ratio=1., reliability_requirement=None, output_dir=root/'comparison', seeds=[1, 2],
                episodes=2, arrival_rate=2., rasoul_warmup_episodes=2)
            with patch.object(params, 'taskno', 3), \
                 patch.object(compare_tore, 'SERVER_INFO_PATH', root/'servers.xlsx'), \
                 patch.object(compare_tore, 'TRAIN_TASK_PATH', root/'train.xlsx'), \
                 patch.object(compare_tore, 'EVALUATION_TASK_PATH', root/'eval.xlsx'), \
                 contextlib.redirect_stdout(io.StringIO()):
                compare_tore.run_comparison(args)
            output = args.output_dir
            metrics = pd.read_csv(output/'eval_episode_metrics.csv')
            self.assertEqual(set(metrics.method), {'TORE', 'RASOUL', 'SML', 'PPO'})
            self.assertTrue(metrics.groupby(['seed_id', 'episode_id']).arrival_sha256.nunique().eq(1).all())
            raw = pd.read_csv(output/'eval_task_metrics.csv')
            self.assertEqual(set(raw.task_id), {201, 202, 203})
            self.assertTrue(np.isfinite(raw.task_qos).all())
            rasoul = metrics[metrics.method == 'RASOUL']
            self.assertTrue(rasoul.rasoul_initial_decision_count.eq(6).all())
            self.assertTrue(rasoul.rasoul_final_decision_count.eq(9).all())
            self.assertTrue(rasoul.groupby('seed_id').rasoul_initial_history_sha256.nunique().eq(1).all())
            self.assertTrue(raw[raw.method == 'RASOUL'].server_B_id.isna().all())
            for seed in args.seeds:
                warmup = pd.read_csv(output/f'rasoul_warmup_tasks_seed_{seed}.csv.gz')
                self.assertEqual(set(warmup.task_id), {101, 102, 103})
                self.assertEqual(len(warmup), 6)
                history = json.loads((output/f'rasoul_history_seed_{seed}.json').read_text())
                self.assertEqual(sum(history['feedback_counts']), 6)
                digest = RASOULPolicy.history_digest(history)
                self.assertTrue(rasoul[rasoul.seed_id == seed].rasoul_initial_history_sha256.eq(digest).all())
            manifest = json.loads((output/'output_manifest.json').read_text())
            self.assertTrue(all(hashlib.sha256((output/name).read_bytes()).hexdigest() == digest
                                for name, digest in manifest.items()))
            self.assertTrue(json.loads((output/'validation.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
