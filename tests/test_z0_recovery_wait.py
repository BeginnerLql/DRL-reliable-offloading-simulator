"""Regression coverage for removal of the former z=0 recovery mode."""

import unittest

import simpy

from core.env_state import EnvironmentState
from core.server import Server
from core.task import Task


class FormerZ0RecoveryWaitTests(unittest.TestCase):
    def test_same_server_is_not_a_valid_dual_replica_action(self):
        env = simpy.Environment()
        state = EnvironmentState()
        server = Server(env, "Edge", 1, 10.0, 0.001, -37.8, 144.9)
        task = Task.__new__(Task)
        task.env = env
        task.env_state = state
        task.resolution_event = env.event()
        generator = task.execute_task(server, server)
        with self.assertRaisesRegex(ValueError, "distinct Edge servers"):
            next(generator)


if __name__ == "__main__":
    unittest.main()
