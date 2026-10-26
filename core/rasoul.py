"""Single-replica RASOUL adaptation with queue-aware QoS residual learning."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from config.params import params
from core.env_state import predict_server_completion_times
from core.task import compute_task_qos


@dataclass(frozen=True)
class RASOULDecision:
    server_id: int
    fallback_reason: str | None
    safe_node_mask: tuple
    admitted_node_mask: tuple
    predicted_latency_s: float
    predicted_qos: float
    selection_score: float | None  # None means an unobserved node's +inf priority.
    history_mean_error: float
    confidence_bonus: float | None
    feedback_count: int
    decision_count: int
    selected_load: float
    selected_load_after_admission: float

    @property
    def is_fallback(self):
        return self.fallback_reason is not None


def single_core_loads(env_state, server_ids):
    """Sum full execution/deadline for CPU-resident tasks, including the running one.

    This is not remaining work in seconds. Uploads in flight are excluded, and
    running work retains its full normalized contribution until completion.
    """
    loads = []
    for sid in server_ids:
        info = env_state.servers[sid]
        residents = list(info['waiting_replicas'])
        if info['running_replica'] is not None:
            residents.append(info['running_replica'])
        load = 0.
        for replica in residents:
            deadline = replica['task'].deadline
            if deadline is None or not np.isfinite(deadline) or deadline <= 0:
                raise ValueError('RASOUL admission requires positive finite resident deadlines')
            load += replica['service_time'] / deadline
        loads.append(load)
    return np.asarray(loads, dtype=float)


class RASOULPolicy:
    """Keep cumulative completed-feedback statistics; never learn from pending tasks."""

    def __init__(self, server_ids):
        ids = tuple(sorted(int(sid) for sid in server_ids))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('RASOUL requires unique server IDs')
        self.server_ids = ids
        self.decision_count = 0
        self.feedback_counts = np.zeros(len(ids), dtype=np.int64)
        self.error_sums = np.zeros(len(ids), dtype=float)
        self._pending = {}

    def snapshot(self):
        """Only drained histories can seed an independent evaluation episode."""
        if self._pending:
            raise ValueError('Drain pending RASOUL tasks before saving history')
        return dict(server_ids=list(self.server_ids), decision_count=self.decision_count,
                    feedback_counts=self.feedback_counts.tolist(), error_sums=self.error_sums.tolist())

    @classmethod
    def from_snapshot(cls, history):
        policy = cls(history['server_ids'])
        if list(policy.server_ids) != list(history['server_ids']):
            raise ValueError('RASOUL history must use sorted server IDs')
        total = history['decision_count']
        counts = np.asarray(history['feedback_counts'])
        errors = np.asarray(history['error_sums'], dtype=float)
        if (isinstance(total, bool) or not isinstance(total, int) or total < 0
                or counts.shape != (len(policy.server_ids),) or errors.shape != counts.shape
                or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0)
                or int(counts.sum()) != total or not np.isfinite(errors).all()
                or np.any(np.abs(errors) > counts + 1e-9)):
            raise ValueError('Invalid or undrained RASOUL history')
        policy.decision_count = total
        policy.feedback_counts = counts.astype(np.int64, copy=True)
        policy.error_sums = errors.copy()
        return policy

    @staticmethod
    def history_digest(history):
        return hashlib.sha256(json.dumps(history, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def select(self, observation, task):
        """Rank reliable nodes, apply single-core admission, then dispatch one node."""
        if task.id in self._pending:
            raise ValueError('RASOUL already has a pending task with this ID')
        state = task.env_state
        if tuple(sorted(state.servers)) != self.server_ids:
            raise ValueError('RASOUL history and environment servers must match')
        predicted = predict_server_completion_times(observation)
        marginal = np.asarray(state.compute_task_reliabilities(task)[0], dtype=float)
        if (predicted.shape != (len(self.server_ids),) or marginal.shape != predicted.shape
                or not np.isfinite(predicted).all() or np.any(predicted < 0)
                or not np.isfinite(marginal).all() or np.any((marginal < 0) | (marginal > 1))
                or not np.isfinite(task.reliability_requirement)
                or not 0 < task.reliability_requirement <= 1):
            raise ValueError('RASOUL requires valid predictions and marginal reliabilities')
        qos = np.asarray(compute_task_qos(predicted, task.deadline,
                                        grace_multiplier=params.QOS_GRACE_MULTIPLIER))
        loads = single_core_loads(state, self.server_ids)
        service = np.array([task.computation_demand / state.get_server_by_id(sid).processing_frequency
                            for sid in self.server_ids])
        proposed = loads + service / task.deadline
        safe = marginal >= task.reliability_requirement
        admitted = safe & (proposed <= 1.)
        decision_count = self.decision_count + 1  # Include the current allocation.
        means = np.divide(self.error_sums, self.feedback_counts,
                          out=np.zeros_like(self.error_sums), where=self.feedback_counts > 0)
        bonuses = np.full(len(self.server_ids), np.inf)
        observed = self.feedback_counts > 0
        bonuses[observed] = np.sqrt(2 * np.log(max(2, decision_count)) / self.feedback_counts[observed])
        scores = qos - means + bonuses
        fallback = None
        if admitted.any():
            index = min(np.flatnonzero(admitted), key=lambda i: (
                -scores[i], predicted[i], self.server_ids[i]))
        else:
            fallback = 'admission' if safe.any() else 'no_reliable_node'
            candidates = np.flatnonzero(safe) if safe.any() else range(len(self.server_ids))
            index = min(candidates, key=lambda i: (predicted[i], self.server_ids[i]))
        decision = RASOULDecision(
            server_id=self.server_ids[index], fallback_reason=fallback,
            safe_node_mask=tuple(bool(flag) for flag in safe),
            admitted_node_mask=tuple(bool(flag) for flag in admitted),
            predicted_latency_s=float(predicted[index]), predicted_qos=float(qos[index]),
            selection_score=float(scores[index]) if observed[index] else None,
            history_mean_error=float(means[index]),
            confidence_bonus=float(bonuses[index]) if observed[index] else None,
            feedback_count=int(self.feedback_counts[index]), decision_count=decision_count,
            selected_load=float(loads[index]), selected_load_after_admission=float(proposed[index]),
        )
        self.decision_count = decision_count
        self._pending[task.id] = (task, decision)
        return decision

    def observe(self, task):
        """Update the selected node exactly once, after actual task completion."""
        pending = self._pending.get(task.id)
        if pending is None or pending[0] is not task:
            raise ValueError('No matching pending RASOUL decision')
        decision = pending[1]
        if (not task.task_completion_event.triggered or task.task_completion_time is None
                or task.task_completion_time > task.env.now
                or len(task.active_replicas) != 1
                or task.replica_A['server_id'] != decision.server_id
                or task.replica_A['finish_time'] is None):
            raise ValueError('RASOUL feedback requires one completed replica on the selected node')
        actual = compute_task_qos(task.task_completion_time - task.arrival_time, task.deadline,
                                  grace_multiplier=params.QOS_GRACE_MULTIPLIER)
        error = decision.predicted_qos - actual  # Never use the corrected/exploration score.
        index = self.server_ids.index(decision.server_id)
        self.feedback_counts[index] += 1
        self.error_sums[index] += error
        del self._pending[task.id]
        task.rasoul_feedback = dict(prediction_error=error, actual_qos=actual,
                                   feedback_time=float(task.env.now),
                                   feedback_count_after=int(self.feedback_counts[index]))
        return task.rasoul_feedback
