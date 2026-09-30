"""Runtime task initialized from a run-level task profile."""

class Task:
    def __init__(self, env, state, task_id, task_profile):
        self.env = env
        self.env_state = state
        self.id = task_id
        self.task_size = task_profile["Task_Size"]
        self.computation_demand = task_profile["Computation_Demand"]
        self.reliability_requirement = task_profile["Reliability_Requirement"]
        self.replica_A_reliability = None
        self.replica_B_reliability = None
        self.pair_reliability = None

        self.arrival_time = float(self.env.now)
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
        return server.transmission_rate

    def execute_task(self, server_A, server_B):
        self.replica_A["server_id"] = server_A.server_id
        self.replica_B["server_id"] = server_B.server_id

        process_A = self.env.process(
            self._run_replica(server_A, "replica_A", self.replica_A)
        )
        process_B = self.env.process(
            self._run_replica(server_B, "replica_B", self.replica_B)
        )
        yield self.env.all_of([process_A, process_B])

    def _run_replica(self, server, replica_name, replica_info):
        transmission_rate = self.get_transmission_rate(server)
        transmission_time = self.task_size / transmission_rate
        yield self.env.timeout(transmission_time)

        replica_info["queue_enter_time"] = float(self.env.now)
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
