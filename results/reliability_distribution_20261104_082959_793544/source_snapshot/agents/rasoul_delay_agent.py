"""RASOUL-inspired delay LCB using completed observations, without queue access."""

import math


class RasoulDelayAgent:
    def __init__(self, server_ids, exploration_c=0.5):
        self.server_ids = tuple(sorted(server_ids))
        self.exploration_c = float(exploration_c)
        if not math.isfinite(self.exploration_c) or self.exploration_c < 0:
            raise ValueError("RASOUL_DELAY_EXPLORATION_C must be finite and nonnegative")
        self.reset()

    def reset(self):
        self.decision_count = 0
        self.completed_counts = dict.fromkeys(self.server_ids, 0)
        self.discrepancy_sums = dict.fromkeys(self.server_ids, 0.0)

    def predict_delay(self, server_id, base_delay):
        count = self.completed_counts[server_id]
        if count == 0:
            return -math.inf  # Unobserved safe servers have exploration priority.
        return (
            base_delay + self.discrepancy_sums[server_id] / count
            - self.exploration_c * math.sqrt(2 * math.log(max(1, self.decision_count)) / count)
        )

    def select_action(self, base_delays, allowed_servers, *, fallback=False):
        # Includes this decision and Fallback decisions, independently of feedback.
        self.decision_count += 1
        if fallback:
            return min(allowed_servers, key=lambda server: (base_delays[server], server))
        return min(allowed_servers, key=lambda server: (
            self.predict_delay(server, base_delays[server]), base_delays[server], server,
        ))

    def observe(self, server_id, actual_delay, base_delay):
        self.discrepancy_sums[server_id] += actual_delay - base_delay
        self.completed_counts[server_id] += 1
