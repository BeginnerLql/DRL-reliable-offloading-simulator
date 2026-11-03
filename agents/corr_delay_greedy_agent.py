"""Shared delay ranking for Corr-delayGreedy and Indep-delayGreedy."""


class CorrDelayGreedyAgent:
    def select_action(self, task, env_state, action_pairs, *, action_mask):
        """Use current CPU backlog only; uploads in flight are not observed.

        The caller supplies the chosen reliability model's safe set, or maximum-reliability
        pairs when no pair satisfies the requirement. Upload overlaps existing
        CPU work. Ties prefer earlier second completion, then server IDs.
        """
        completion_times = {}
        for server_id in env_state.servers:
            server = env_state.get_server_by_id(server_id)
            running, waiting = env_state.get_server_backlog_components(
                server_id, task.env.now,
            )
            upload = task.task_size / server.transmission_rate
            execution = task.computation_demand / server.processing_frequency
            completion_times[server_id] = max(upload, running + waiting) + execution

        def score(index):
            server_a, server_b = action_pairs[index]
            time_a, time_b = completion_times[server_a], completion_times[server_b]
            return min(time_a, time_b), max(time_a, time_b), server_a, server_b

        return min(
            (index for index, allowed in enumerate(action_mask) if allowed),
            key=score,
        )
