import types
import unittest

import simpy

from core.task import Task


class Z0RecoveryWaitTests(unittest.TestCase):
    @staticmethod
    def _task(env):
        task = Task.__new__(Task)
        task.env = env
        task.env_state = None
        task.id = 1
        task.primaryNode = None
        task.backupNode = None
        task.z = None
        task.primaryStarted = None
        task.primaryFinished = None
        task.primaryStat = None
        task.primary_service_time = None
        task.backupStarted = None
        task.backupFinished = None
        task.backupStat = None
        task.resolution_event = env.event()
        task.teta = 10.0
        return task

    @staticmethod
    def _install_replica_stubs(task, primary_status, backup_starts):
        def primary(self):
            yield self.env.timeout(2.0)
            self.primaryStat = primary_status
            self.primaryFinished = self.env.now
            self._signal_resolution_if_ready()

        def backup(self):
            backup_starts.append(self.env.now)
            yield self.env.timeout(1.0)
            self.backupStat = "success"
            self.backupFinished = self.env.now
            self._signal_resolution_if_ready()

        task.primary = types.MethodType(primary, task)
        task.backup = types.MethodType(backup, task)

    def test_z0_primary_completion_keeps_backup_standby(self):
        env = simpy.Environment()
        task = self._task(env)
        backup_starts = []
        self._install_replica_stubs(task, "success", backup_starts)

        env.process(task.execute_task("primary", "backup", 0))
        env.run()

        self.assertEqual(backup_starts, [])
        self.assertIsNone(task.backupStarted)
        self.assertIsNone(task.backupFinished)
        self.assertTrue(task.resolution_event.triggered)
        self.assertEqual(task.teta, 10.0)

    def test_z0_primary_success_does_not_start_backup(self):
        env = simpy.Environment()
        task = self._task(env)
        backup_starts = []
        self._install_replica_stubs(task, "success", backup_starts)

        env.process(task.execute_task("primary", "backup", 0))
        env.run()

        self.assertEqual(backup_starts, [])
        self.assertIsNone(task.backupStarted)
        self.assertTrue(task.resolution_event.triggered)


if __name__ == "__main__":
    unittest.main()
