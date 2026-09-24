import unittest

import torch

from diagnostics.run_pair_stochastic_deployment import (
    native_policy_probabilities,
    sample_native_action,
)


class NativePairSamplingTests(unittest.TestCase):
    def test_probabilities_are_unmasked_softmax_over_all_legal_actions(self):
        logits = torch.linspace(-2.0, 2.0, 28)
        actual = native_policy_probabilities(logits)
        expected = torch.softmax(logits, dim=-1)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(int((actual > 0).sum()), 28)
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=6)

    def test_private_generator_is_reproducible_and_does_not_change_global_rng(self):
        logits = torch.linspace(-1.0, 1.0, 28)
        generator_a = torch.Generator(device="cpu").manual_seed(20260924)
        generator_b = torch.Generator(device="cpu").manual_seed(20260924)
        global_state = torch.random.get_rng_state().clone()
        samples_a = [sample_native_action(logits, generator_a)[0] for _ in range(50)]
        samples_b = [sample_native_action(logits, generator_b)[0] for _ in range(50)]
        self.assertEqual(samples_a, samples_b)
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))

    def test_sampled_probability_is_the_native_policy_probability(self):
        logits = torch.tensor([-1.0, 0.0, 1.0] + [0.0] * 25)
        generator = torch.Generator(device="cpu").manual_seed(77)
        action, probabilities = sample_native_action(logits, generator)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 28)
        self.assertAlmostEqual(
            float(probabilities[action]),
            float(torch.softmax(logits, dim=-1)[action]),
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
