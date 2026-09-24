"""Focused checks for the formal experiment's seed-level statistics."""
import unittest

import numpy as np
import pandas as pd

from diagnostics.aggregate_masked_pair_ppo_10seed import bootstrap_ci, metrics, paired_tables
from diagnostics.run_masked_pair_ppo_10seed import action_seed, formal_spec


class FormalMaskedPairStatisticsTests(unittest.TestCase):
    def test_formal_seed_plan_and_independent_action_seed(self):
        meta,plan=formal_spec()
        self.assertEqual((meta['train_episodes'],meta['eval_episodes'],meta['tasks_per_episode']),
                         (300,20,200))
        self.assertEqual(plan.Trial_ID.tolist(),list(range(10)))
        derived=[action_seed(row) for _,row in plan.iterrows()]
        self.assertEqual(len(set(derived)),10)
        self.assertEqual(derived,[action_seed(row) for _,row in plan.iterrows()])
        for value,row in zip(derived,plan.itertuples()):
            self.assertNotIn(value,(row.Eval_Arrival_Seed,row.Eval_Spatial_Seed))

    def test_seed_level_bootstrap_is_deterministic(self):
        values=np.arange(10,dtype=float)
        self.assertEqual(bootstrap_ci(values),bootstrap_ci(values))
        low,high=bootstrap_ci(values)
        self.assertLess(low,values.mean())
        self.assertGreater(high,values.mean())
        with self.assertRaises(RuntimeError):bootstrap_ci([1.0,2.0])

    def test_task_metrics_keep_feasibility_and_policy_failure_distinct(self):
        assignments=pd.DataFrame({
            'episode':[1,1,1], 'task_id':[1,2,3], 'action_index':[0,1,2],
            'Task_Reward':[10.,20.,30.], 'Task_Delay':[1.,2.,3.],
            'Reliability_Satisfied':[True,False,False]})
        decisions=pd.DataFrame({
            'episode':[1,1,1], 'task_id':[1,2,3], 'action_index':[0,1,2],
            'safe_set_empty':[False,False,True], 'selected_action_safe':[True,False,False],
            'reliability_deficit':[0.,0.,.0002], 'safe_set_size':[2,2,0],
            'masked_entropy':[.5,.4,0.]})
        result=metrics(assignments,decisions)
        self.assertAlmostEqual(result['feasibility_rate'],2/3)
        self.assertAlmostEqual(result['conditional_rsr'],.5)
        self.assertAlmostEqual(result['avoidable_violation_rate'],1/3)
        self.assertAlmostEqual(result['unavoidable_violation_rate'],1/3)
        self.assertAlmostEqual(result['mean_empty_deficit'],.0002)

    def test_latency_wins_are_directional(self):
        rows=[]
        for trial in range(10):
            for policy in ('pair','masked_stochastic','masked_greedy'):
                rows.append({'trial_id':trial,'policy':policy,
                    'mean_reward':1.+int(policy!='pair'),
                    'mean_latency':2.-int(policy!='pair'),
                    'overall_rsr':.9,'highest_rsr':.8,
                    'pair_selection_hhi':.5,'pair_selection_entropy':1.,
                    'top1_pair_frequency':.5,'maximum_server_selection_share':.5,
                    'maximum_server_utilization':.5,'maximum_mean_queue_length':.5,
                    'maximum_p95_queue_length':1.,'mean_server_queue_length':.5,
                    'mean_server_p95_queue_length':1.,'avoidable_violation_rate':0.,
                    'conditional_rsr':1.})
        paired,ci,wlt=paired_tables(pd.DataFrame(rows))
        self.assertEqual(len(paired),20)
        value=wlt[(wlt.comparison=='masked_minus_pair')&(wlt.metric=='mean_latency')].iloc[0]
        self.assertEqual((value.positive_seeds,value.negative_seeds,value.ties),(10,0,0))
        reward=ci[(ci.comparison=='masked_minus_pair')&(ci.metric=='mean_reward')].iloc[0]
        self.assertAlmostEqual(reward.mean_delta,1.)
        self.assertGreater(reward.ci95_low,0.)


if __name__=='__main__':unittest.main()
