"""Seed-level inference and in-memory scene overrides for the formal runner."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.params import params
from tools import run_aligned_experiments as runner


class AlignedExperimentTests(unittest.TestCase):
    def test_paired_statistics_uses_seed_count_and_correct_metric_directions(self):
        rows = []
        for seed in (1,2):
            for method in ('PPO','SML','TORE','RASOUL'):
                metrics = {metric: 1. for metric in runner.METRICS}
                if method == 'PPO':
                    metrics['mean_qos'] += seed/10
                    metrics['mean_latency_s'] -= seed/10
                rows.append(dict(scenario='test', seed_id=seed, method=method, **metrics))
        stats = runner.paired_statistics(pd.DataFrame(rows))
        qos = stats[(stats.reference == 'SML') & (stats.metric == 'mean_qos')].iloc[0]
        delay = stats[(stats.reference == 'SML') & (stats.metric == 'mean_latency_s')].iloc[0]
        self.assertEqual(qos.seed_count, 2)
        self.assertAlmostEqual(qos.delta_ppo_minus_reference, .15)
        self.assertEqual(qos.ppo_better_seeds, 2)
        self.assertEqual(delay.ppo_better_seeds, 2)
        self.assertLess(qos.ci95_lower, qos.delta_ppo_minus_reference)
        self.assertGreater(qos.ci95_upper, qos.delta_ppo_minus_reference)

    def test_one_seed_does_not_fabricate_a_confidence_interval(self):
        rows = [dict(scenario='one', seed_id=1, method=method,
                     **{metric:1. for metric in runner.METRICS})
                for method in ('PPO','SML','TORE','RASOUL')]
        stats = runner.paired_statistics(pd.DataFrame(rows))
        self.assertTrue(stats.ci95_lower.isna().all())
        self.assertTrue(stats.ci95_upper.isna().all())

    def test_scenario_overrides_both_splits_without_changing_workbooks(self):
        servers = {sid:dict(Site_ID=str(sid), Processing_Frequency=4e9,
            Base_Failure_Rate=.5, Transmission_Rate=20., Latitude=0., Longitude=sid/100.)
            for sid in range(1,9)}
        tasks = {tid:dict(Task_ID=tid, Task_Size=5., Computation_Demand=2e9,
                         Reliability_Requirement=.9999, Deadline=2.) for tid in (1,2)}
        with patch.object(params, 'taskno', 2), patch.object(runner.pd, 'read_excel', return_value=pd.DataFrame()), \
             patch.object(runner.MainLoop, '_build_server_profiles', return_value=servers), \
             patch.object(runner.MainLoop, '_build_task_profiles', return_value=tasks):
            _, originals, cases, feasibility = runner.prepare_scenarios(['low_requirement'])
        self.assertEqual(len(feasibility), 2)
        for split in ('train','evaluation'):
            self.assertTrue(all(p['Reliability_Requirement'] == .9 for p in cases['low_requirement'][split].values()))
            self.assertTrue(all(p['Reliability_Requirement'] == .9999 for p in originals[split].values()))
            self.assertTrue(all('safe_mask' not in p for p in originals[split].values()))

    def test_missing_cuda_fails_before_creating_formal_output(self):
        args = SimpleNamespace(device='cuda', preflight_only=False, scenarios=['low_requirement'])
        prepared = ({}, {}, {'low_requirement': {'train': {1:{'safe_mask':np.ones(28,dtype=bool)}}}}, pd.DataFrame())
        with patch.object(runner, 'prepare_scenarios', return_value=prepared), \
             patch.object(runner.torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                runner.run(args)


if __name__ == '__main__':
    unittest.main()
