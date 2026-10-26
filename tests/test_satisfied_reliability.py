"""Conditional reliability uses actual selections, empty groups and task-weighted pooling."""

import unittest

import numpy as np
import pandas as pd

from compare_tore import aggregate_comparison_metrics, summarize_task_groups
from io_utils.save_parameters_and_logs import summarize_satisfied_reliability
from tools.run_aligned_experiments import METRICS, paired_statistics


class SatisfiedReliabilityTests(unittest.TestCase):
    def records(self):
        return pd.DataFrame(dict(seed_id=[1,2,2,2,2,1], method=['PPO']*5+['RASOUL'],
            pair_reliability=[.91,.99,.99,.99,.89,.89],
            selection_reliability=[1.]*6,
            requirement_satisfied=[True,True,True,True,False,False],
            is_fallback=[False,True,False,False,False,True],
            task_qos=[0.,.2,.3,.4,1.,1.], task_latency=[9.,8.,7.,6.,1.,1.]))

    def test_actual_selected_reliability_includes_reliable_fallback_and_late_tasks(self):
        summary = summarize_satisfied_reliability(self.records(), ['method']).set_index('method')
        self.assertEqual(summary.loc['PPO', 'satisfied_task_count'], 4)
        self.assertAlmostEqual(summary.loc['PPO', 'satisfied_reliability_sum'], 3.88)
        self.assertAlmostEqual(summary.loc['PPO', 'mean_satisfied_reliability'], .97)
        self.assertEqual(summary.loc['RASOUL', 'satisfied_task_count'], 0)
        self.assertTrue(np.isnan(summary.loc['RASOUL', 'mean_satisfied_reliability']))

    def test_seed_pooling_weights_by_satisfied_task_count(self):
        original = self.records()
        seeds = summarize_task_groups(original, ['seed_id','method'])
        pooled = aggregate_comparison_metrics(seeds, ['method'], ['mean_qos','mean_satisfied_reliability'])
        direct = summarize_task_groups(original, ['method'])
        pd.testing.assert_series_equal(pooled.set_index('method').mean_satisfied_reliability,
                                       direct.set_index('method').mean_satisfied_reliability)
        self.assertAlmostEqual(pooled.set_index('method').loc['PPO','mean_satisfied_reliability'], .97)

    def test_conditional_qos_scope_does_not_change_reliability_subset(self):
        records = self.records()
        all_tasks = summarize_task_groups(records, ['method'])
        conditional = summarize_task_groups(records, ['method'], only_satisfied=True)
        pd.testing.assert_series_equal(all_tasks.mean_satisfied_reliability,
                                       conditional.mean_satisfied_reliability)
        self.assertEqual(conditional.set_index('method').loc['RASOUL','satisfied_task_count'], 0)

    def test_paired_conditional_reliability_excludes_missing_pairs(self):
        rows = []
        for seed in (1,2):
            for method in ('PPO','SML','TORE','RASOUL'):
                metrics = {metric:1. for metric in METRICS}
                if method == 'RASOUL' or (method == 'SML' and seed == 2):
                    metrics['mean_satisfied_reliability'] = np.nan
                rows.append(dict(scenario='test', seed_id=seed, method=method, **metrics))
        stats = paired_statistics(pd.DataFrame(rows))
        stats = stats[stats.metric == 'mean_satisfied_reliability'].set_index('reference')
        self.assertEqual(stats.loc['SML','seed_count'], 1)
        self.assertEqual(stats.loc['RASOUL','seed_count'], 0)
        self.assertTrue(np.isnan(stats.loc['RASOUL','delta_ppo_minus_reference']))
        self.assertTrue(np.isnan(stats.loc['SML','ci95_lower']))


if __name__ == '__main__':
    unittest.main()
