"""Runtime task initialized from a fixed profile: size in MB, demand in cycles."""

import numpy as np


def compute_task_deadlines(task_sizes, computation_demands, processing_frequencies,
                           transmission_rates, *, kappa):
    """Return (reference latency, relative deadline), in seconds, without queues.

    Task sizes (MB) and demands (cycles) broadcast; server frequency (cycles/s)
    and upload rate (MB/s) are matching nonempty vectors. No policy, reliability
    requirement, arrival rate or future outcome enters the reference latency.
    """
    sizes, demands = np.broadcast_arrays(np.asarray(task_sizes, dtype=float),
                                        np.asarray(computation_demands, dtype=float))
    frequencies = np.asarray(processing_frequencies, dtype=float)
    rates = np.asarray(transmission_rates, dtype=float)
    if not np.isfinite(kappa) or kappa <= 1:
        raise ValueError("Deadline scale kappa must be finite and greater than one")
    if (frequencies.ndim != 1 or frequencies.size == 0 or rates.shape != frequencies.shape
            or not np.isfinite(frequencies).all() or not np.isfinite(rates).all()
            or np.any(frequencies <= 0) or np.any(rates <= 0)):
        raise ValueError("Server frequencies and upload rates must be positive finite vectors")
    if (not np.isfinite(sizes).all() or not np.isfinite(demands).all()
            or np.any(sizes < 0) or np.any(demands < 0)):
        raise ValueError("Task sizes and computation demands must be finite and nonnegative")
    reference = np.min(sizes[..., None] / rates + demands[..., None] / frequencies, axis=-1)
    deadlines = kappa * reference
    if not np.isfinite(deadlines).all() or np.any(deadlines <= 0):
        raise ValueError("Generated deadlines must be positive and finite")
    return reference, deadlines


def compute_task_qos(task_latency, deadline, *, grace_multiplier):
    """QoS is one through the deadline, then linear to zero at b * deadline."""
    latency, deadline = np.broadcast_arrays(np.asarray(task_latency, dtype=float),
                                          np.asarray(deadline, dtype=float))
    if not np.isfinite(grace_multiplier) or grace_multiplier <= 1:
        raise ValueError("QoS grace multiplier must be finite and greater than one")
    if not np.isfinite(latency).all() or np.any(latency < 0):
        raise ValueError("Task latency must be finite and nonnegative")
    if not np.isfinite(deadline).all() or np.any(deadline <= 0):
        raise ValueError("QoS requires a positive finite task deadline in seconds")
    qos = np.clip((grace_multiplier-latency/deadline)/(grace_multiplier-1), 0., 1.)
    return float(qos) if qos.ndim == 0 else qos


def compute_task_utility(task_qos, requirement_satisfied):
    """Effective QoS: keep latency QoS only when actual reliability meets its requirement."""
    qos, satisfied = np.broadcast_arrays(np.asarray(task_qos, dtype=float),
                                        np.asarray(requirement_satisfied, dtype=bool))
    if not np.isfinite(qos).all() or np.any((qos < 0) | (qos > 1)):
        raise ValueError("Task QoS must be finite and in [0, 1]")
    utility = np.where(satisfied, qos, 0.)
    return float(utility) if utility.ndim == 0 else utility


class Task:
    def __init__(self, env, state, task_id, task_profile):
        self.env = env
        self.env_state = state
        self.id = task_id
        self.task_size = task_profile["Task_Size"]
        self.computation_demand = task_profile["Computation_Demand"]
        self.reliability_requirement = task_profile["Reliability_Requirement"]
        self.deadline = task_profile.get("Deadline")  # Seconds relative to arrival.
        self.replica_A_reliability = None
        self.replica_B_reliability = None
        self.pair_reliability = None
        self.reliability_snapshot = task_profile.get("reliability_snapshot")
        self.safe_mask = task_profile.get("safe_mask")
        self.action_index = None
        self.is_fallback = False
        self.ppo_decision = False  # Policy inference is independent of training storage.
        self.ppo_action_mask = None
        self.ppo_transition_index = None
        self.ppo_delta_t = None
        self.ppo_next_task_id = None
        # Completion utility belongs to the decision interval in which it is observed.
        self.ppo_reward_owner_task_id = None
        self.ppo_interval_reward = None

        self.arrival_time = float(self.env.now)
        self.absolute_deadline = (self.arrival_time + self.deadline
                                  if self.deadline is not None else None)
        self.replica_A = {
            "server_id": None,
            "queue_enter_time": None,
            "cpu_start_time": None,
            "finish_time": None,
        }
        self.replica_B = {
            "server_id": None,
            "queue_enter_time": None,
            "cpu_start_time": None,
            "finish_time": None,
        }
        self.task_completion_time = None
        self.task_completion_event = self.env.event()

    def get_transmission_rate(self, server):
        """Return the server's fixed upload rate in MB/s."""
        return server.transmission_rate

    @property
    def active_replicas(self):
        """Only dispatched replicas; an unused TORE replica has no lifecycle."""
        return tuple(r for r in (self.replica_A, self.replica_B) if r["server_id"] is not None)

    def execute_task(self, server_A, server_B=None):
        if server_B is not None and server_A.server_id == server_B.server_id:
            raise ValueError("Replicas must run on distinct servers")
        self.replica_A["server_id"] = server_A.server_id
        processes = [self.env.process(
            self._run_replica(server_A, "replica_A", self.replica_A)
        )]
        if server_B is not None:
            self.replica_B["server_id"] = server_B.server_id
            processes.append(self.env.process(
                self._run_replica(server_B, "replica_B", self.replica_B)
            ))
        # The losing replica continues after first-replica task completion.
        yield self.env.all_of(processes)

    def _run_replica(self, server, replica_name, replica_info):
        transmission_rate = self.get_transmission_rate(server)
        transmission_time = self.task_size / transmission_rate  # MB / (MB/s) = s
        yield self.env.timeout(transmission_time)

        replica_info["queue_enter_time"] = float(self.env.now)
        # CPU cycles / (cycles/s) = seconds.
        service_time = self.computation_demand / server.processing_frequency
        self.env_state.register_waiting_replica(
            server.server_id, self, replica_name, service_time
        )
        with server.queue.request(priority=1) as request:
            yield request
            replica_info["cpu_start_time"] = float(self.env.now)
            self.env_state.start_replica_execution(
                server.server_id,
                self,
                replica_name,
                service_time,
                replica_info["cpu_start_time"],
            )
            yield self.env.timeout(service_time)
            replica_info["finish_time"] = float(self.env.now)
            self.env_state.complete_replica_execution(server.server_id)

        self._signal_task_completion()

    def _signal_task_completion(self):
        if self.task_completion_event.triggered:
            return
        self.task_completion_time = float(self.env.now)
        self.task_completion_event.succeed(self.task_completion_time)
