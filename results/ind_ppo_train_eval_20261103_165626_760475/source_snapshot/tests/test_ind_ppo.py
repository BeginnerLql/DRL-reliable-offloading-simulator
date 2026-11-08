import copy
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Project_main import build_model
from config.params import params
from config.paths import EVALUATION_TASK_PATH
from core.main_loop import MainLoop
from tools.run_experiments import temporary_params


class IndependentPPOTests(unittest.TestCase):
    def test_same_initial_networks_and_hyperparameters_as_corr_ppo(self):
        models = []
        with redirect_stdout(io.StringIO()):
            for name in ("ppo", "ind-ppo"):
                with temporary_params(model_summary=name):
                    torch.manual_seed(123)
                    models.append(build_model(device="cpu"))
        correlated, independent = models
        self.assertEqual(type(correlated), type(independent))
        self.assertEqual(independent.num_actions, 28)
        for network in ("policy_net", "policy_old", "value_net"):
            left, right = (getattr(model, network).state_dict() for model in models)
            self.assertEqual(left.keys(), right.keys())
            self.assertTrue(all(torch.equal(left[key], right[key]) for key in left))
        for setting in ("gamma", "clip_eps", "k_epochs", "batch_size", "entropy_coef", "gae_lambda",
                        "reward_scale", "value_loss_coef", "max_grad_norm", "min_rollout"):
            self.assertEqual(getattr(correlated, setting), getattr(independent, setting))

    def test_independent_safe_set_fallback_ties_and_mask_switch(self):
        with temporary_params(model_summary="ind-ppo"), redirect_stdout(io.StringIO()):
            loop = MainLoop(build_model(device="cpu"), 1, 1, params.num_states, 3)
            loop.extract_parameters_from_index = lambda index: (index, None)
            loop._get_pair_reliability = lambda task, index, unused: SimpleNamespace(
                correlated=[.90, .96, .91][index], independent=[.97, .98, .98][index],
            )
            task = SimpleNamespace(reliability_requirement=.95)
            np.testing.assert_array_equal(loop._get_reliability_action_mask(task), [True] * 3)
            self.assertEqual(task.feasible_action_count, 3)
            loop.model_name = "ppo"
            np.testing.assert_array_equal(loop._get_reliability_action_mask(task), [False, True, False])
            loop.model_name = "ind-ppo"
            task.reliability_requirement = .99
            np.testing.assert_array_equal(loop._get_reliability_action_mask(task), [False, True, True])
            self.assertTrue(task.mask_fallback_used)
            self.assertEqual(task.feasible_action_count, 0)
            task.reliability_requirement = .98
            np.testing.assert_array_equal(loop._get_reliability_action_mask(task), [False, True, True])
            self.assertFalse(task.mask_fallback_used)
            self.assertEqual(task.feasible_action_count, 2)
            with patch.object(params, "RELIABILITY_ACTION_MASK_ENABLED", False):
                self.assertIsNone(loop._get_reliability_action_mask(task))

    def test_training_stores_independent_masks_and_updates_each_episode(self):
        torch.set_num_threads(1)
        with temporary_params(model_summary="ind-ppo", EXPERIMENT_SEED=2027), redirect_stdout(io.StringIO()):
            torch.manual_seed(2027)
            np.random.seed(2027)
            model = build_model(device="cpu")
            initial = copy.deepcopy(model.policy_net.state_dict())
            loop = MainLoop(model, 2, 200, params.num_states, params.num_actions,
                            task_parameters_path=EVALUATION_TASK_PATH)
            with patch.object(model, "train_step", wraps=model.train_step) as train, \
                    patch.object(model, "store_transition", wraps=model.store_transition) as store:
                loop.EP()
            self.assertEqual(train.call_count, 2)
            self.assertEqual(store.call_count, 400)
            for call in store.call_args_list:
                self.assertEqual(call.kwargs["action_mask"].shape, (28,))
                self.assertTrue(call.kwargs["action_mask"][call.args[1]])
            self.assertTrue(any(not torch.equal(initial[key], value)
                                for key, value in model.policy_net.state_dict().items()))
            self.assertTrue(model.optimizer_policy.state)
            self.assertEqual(loop.pendingList, [])
            self.assertEqual(loop.tempbuffer, {})
            for row in loop.task_results:
                self.assertEqual(row["requirement_satisfied_independent"], row["feasible_action_count"] > 0)
                self.assertEqual(row["mask_fallback_used"], row["feasible_action_count"] == 0)
                self.assertEqual(row["task_completion_time"],
                                 min(row["replica_A_finish_time"], row["replica_B_finish_time"]))
            self.assertTrue(any(row["requirement_satisfied_independent"]
                                and not row["requirement_satisfied_correlated"] for row in loop.task_results))


if __name__ == "__main__":
    unittest.main()
