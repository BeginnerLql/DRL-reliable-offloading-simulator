import unittest

import numpy as np

from config.params import params
from core.main_loop import MainLoop


class TaskArrivalProcessTests(unittest.TestCase):
    def setUp(self):
        self.loop = MainLoop.__new__(MainLoop)
        self.original_rate = params.TASK_ARRIVAL_RATE

    def tearDown(self):
        params.TASK_ARRIVAL_RATE = self.original_rate

    def test_rate_semantics_and_mean_interval(self):
        self.assertAlmostEqual(float(params.TASK_ARRIVAL_RATE), 0.5)
        self.assertAlmostEqual(1.0 / params.TASK_ARRIVAL_RATE, 2.0)

    def test_samples_are_float_and_nonnegative(self):
        np.random.seed(12345)
        samples = [self.loop._sample_interarrival_time() for _ in range(100)]
        self.assertTrue(all(isinstance(sample, float) for sample in samples))
        self.assertTrue(all(sample >= 0.0 for sample in samples))
        self.assertTrue(any(not sample.is_integer() for sample in samples))

    def test_exponential_sample_mean(self):
        np.random.seed(12345)
        samples = np.random.exponential(scale=2.0, size=100000)
        self.assertLess(abs(float(samples.mean()) - 2.0), 0.05)

    def test_nonpositive_rate_is_rejected(self):
        for invalid_rate in (0.0, -0.5):
            params.TASK_ARRIVAL_RATE = invalid_rate
            with self.assertRaisesRegex(ValueError, "TASK_ARRIVAL_RATE"):
                self.loop._sample_interarrival_time()


if __name__ == "__main__":
    unittest.main()
