"""Default summaries include all tasks; explicit conditional summaries remain supported."""
import unittest

import numpy as np
import pandas as pd

from compare_tore import (reliable_qos_metrics, aggregate_comparison_metrics,
                          all_task_qos_metrics, summarize_reliability_tiers)


class ComparisonQoSTests(unittest.TestCase):
    def test_tiers_include_unreliable_and_fallback_tasks_and_pool_all_rows(self):
        records = pd.DataFrame(dict(seed_id=[1, 1, 1, 2], episode=[1, 1, 1, 1],
            reliability_requirement=[.9, .9, .9999, .9], method=['PPO']*4,
            task_qos=[.2, .8, .1, .8], task_latency=[1., 2., 3., 4.],
            requirement_satisfied=[True, False, False, False], is_fallback=[False, True, True, True]))
        original = records.copy(deep=True)
        self.assertAlmostEqual(all_task_qos_metrics(records)['mean_qos'], .05)
        episodes, seeds, summary = summarize_reliability_tiers(records)
        low = summary.set_index('reliability_requirement').loc[.9]
        self.assertEqual(low.qos_task_count, 3)
        self.assertAlmostEqual(low.mean_qos, .2/3)
        self.assertAlmostEqual(low.mean_latency_s, 7/3)
        self.assertAlmostEqual(low.p95_latency_s, 3.8)
        high = summary.set_index('reliability_requirement').loc[.9999]
        self.assertEqual(high.qos_task_count, 1)
        self.assertEqual(high.mean_qos, 0.)
        self.assertEqual(high.mean_latency_s, 3.)
        self.assertEqual(high.theoretical_rsr, 0.)
        self.assertEqual(episodes.qos_task_count.sum(), len(records))
        pooled = aggregate_comparison_metrics(seeds, ['reliability_requirement','method'], ['mean_qos'])
        self.assertAlmostEqual(pooled.set_index('reliability_requirement').loc[.9].mean_qos, .2/3)
        pd.testing.assert_frame_equal(records, original)

    def test_zeroing_uses_actual_reliability_and_keeps_reliable_fallback_qos(self):
        records = pd.DataFrame(dict(seed_id=[1]*3, episode=[1]*3,
            reliability_requirement=[.99]*3, method=['TORE']*3,
            task_qos=[.2, .8, 1.], task_latency=[1., 3., 9.],
            requirement_satisfied=[True, True, False],
            selection_requirement_satisfied=[True]*3, is_fallback=[False, True, False]))
        _, _, summary = summarize_reliability_tiers(records)
        self.assertAlmostEqual(all_task_qos_metrics(records)['mean_qos'], 1/3)
        self.assertAlmostEqual(summary.mean_qos.iloc[0], 1/3)
        self.assertEqual(summary.qos_task_count.iloc[0], 3)
        self.assertAlmostEqual(summary.mean_latency_s.iloc[0], 13/3)

    def test_satisfied_tiers_keep_zero_sample_groups_and_use_actual_not_selection_reliability(self):
        records = pd.DataFrame(dict(seed_id=[1]*4, episode=[1]*4,
            reliability_requirement=[.9, .9, .9, .9999], method=['PPO']*4,
            task_qos=[.2, .8, 1., .9], task_latency=[1., 3., 9., 5.],
            requirement_satisfied=[True, True, False, False],
            selection_requirement_satisfied=[True]*4, is_fallback=[False, True, False, True]))
        episodes, seeds, summary = summarize_reliability_tiers(records, only_satisfied=True)
        low = summary.set_index('reliability_requirement').loc[.9]
        self.assertEqual(low.qos_task_count, 2)
        self.assertEqual(low.tier_task_count, 3)
        self.assertAlmostEqual(low.mean_qos, .5)
        self.assertAlmostEqual(low.mean_latency_s, 2.)
        self.assertAlmostEqual(low.p95_latency_s, 2.9)
        high = summary.set_index('reliability_requirement').loc[.9999]
        self.assertEqual(high.qos_task_count, 0)
        self.assertEqual(high.tier_task_count, 1)
        for metric in ('mean_qos', 'mean_latency_s', 'p95_latency_s'):
            self.assertTrue(np.isnan(high[metric]))
        self.assertEqual(episodes.qos_task_count.sum(), 2)

    def test_eligibility_uses_actual_reliability_including_reliable_fallback(self):
        data = pd.DataFrame(dict(task_qos=[.2, .8, 1.],
            requirement_satisfied=[True, True, False],
            selection_requirement_satisfied=[True, True, True],
            is_fallback=[False, True, False]))
        self.assertEqual(reliable_qos_metrics(data),
                         dict(mean_qos=.5, qos_eligible_count=2, qos_eligible_sum=1.))

    def test_empty_and_unequal_groups_pool_by_eligible_task_count(self):
        rows = []
        for seed, qos in ((1, [.2]), (1, []), (2, [.8, .8, .8])):
            data = pd.DataFrame({'task_qos': qos, 'requirement_satisfied': [True]*len(qos)})
            data['requirement_satisfied'] = data.requirement_satisfied.astype(bool)
            values = reliable_qos_metrics(data)
            if not qos:
                self.assertTrue(np.isnan(values['mean_qos']))
            rows.append(dict(seed_id=seed, method='PPO', **values))
        seeds = aggregate_comparison_metrics(pd.DataFrame(rows), ['seed_id', 'method'], ['mean_qos'])
        summary = aggregate_comparison_metrics(seeds, ['method'], ['mean_qos'])
        self.assertAlmostEqual(summary.mean_qos.iloc[0], .65)
        self.assertEqual(summary.qos_eligible_count.iloc[0], 4)


if __name__ == '__main__':
    unittest.main()
