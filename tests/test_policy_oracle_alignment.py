"""Focused checks for observation-only PPO instrumentation and metric summaries."""
import unittest

import numpy as np
import pandas as pd
import torch

from diagnostics.analyze_policy_oracle_alignment import summary_group, task_ids_by_tier, vector
from diagnostics.instrumented_pair_ppo_run import InstrumentedPairAgent
from tools.pair_policy_diagnostics import DiagnosticPPOAgent


class TestPolicyOracleAlignment(unittest.TestCase):
    def test_instrumentation_preserves_actions_and_parameter_updates(self):
        settings = dict(num_states=35, num_actions=28, hidden_layers=[16],
                        actor_mode="pair_scoring", num_servers=8,
                        pair_correlations=np.linspace(0.1, 0.8, 28),
                        min_rollout=1, batch_size=2, k_epochs=2, minibatch_seed=9)
        agents = []
        for cls in (DiagnosticPPOAgent, InstrumentedPairAgent):
            torch.manual_seed(31)
            agents.append(cls(**settings))
        actions = []
        for agent in agents:
            torch.manual_seed(99)
            chosen = []
            for task_id in range(1, 5):
                state = np.linspace(0, 1, 35, dtype=np.float32) + task_id / 10
                action = agent.select_action(state, epsilon=0.0)
                chosen.append(action)
                agent.store_transition(state, action, float(task_id - 2),
                                       state + 0.01, delta_t=task_id / 2,
                                       done=task_id == 4, task_id=task_id)
            actions.append(chosen)
            agent.train_step()
        self.assertEqual(actions[0], actions[1])
        for name in ("policy_net", "policy_old", "value_net"):
            original = getattr(agents[0], name).state_dict()
            recorded = getattr(agents[1], name).state_dict()
            for key in original:
                torch.testing.assert_close(original[key], recorded[key], rtol=0, atol=0)
        archive = agents[1].rollout_archive
        self.assertEqual(len(archive), 4)
        self.assertEqual([row["sampled_action"] for row in archive], actions[0])
        self.assertAlmostEqual(sum(row["normalized_advantage"] for row in archive), 0, places=5)
        for row in archive:
            self.assertEqual(len(row["probabilities"]), 28)
            self.assertAlmostEqual(sum(row["probabilities"]), 1, places=5)
            self.assertAlmostEqual(row["return"], row["value"] + row["raw_advantage"], places=5)

    def test_tier_selection_and_summary_quantiles(self):
        requirements = np.repeat([0.9, 0.99, 0.999, 0.9999], 5)
        tasks = pd.DataFrame({"Reliability_Requirement": requirements},
                             index=pd.Index(range(1, 21), name="Task_ID"))
        ids = task_ids_by_tier(tasks, 2)
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(sum(np.isclose(tasks.loc[ids, "Reliability_Requirement"], 0.9999)), 2)
        frame = pd.DataFrame({"R_req": [0.9] * 5, "optimal_set_probability_mass": [0, .25, .5, .75, 1]})
        row = summary_group(frame, ["optimal_set_probability_mass"]).iloc[0]
        self.assertEqual(row.optimal_set_probability_mass_mean, .5)
        self.assertEqual(row.optimal_set_probability_mass_p25, .25)
        self.assertEqual(row.optimal_set_probability_mass_p75, .75)
        self.assertEqual(row.optimal_set_probability_mass_p90, .9)

    def test_vector_rejects_non_finite_values(self):
        np.testing.assert_array_equal(vector("[0, 1]"), [0, 1])
        with self.assertRaises(ValueError):
            vector("[0, NaN]")


if __name__ == "__main__":
    unittest.main()
