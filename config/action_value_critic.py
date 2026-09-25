"""Defaults for the independent action-conditioned Q side learner.

These settings are separate from formal PPO configuration so enabling the
auxiliary critic cannot alter legacy experiments.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionValueCriticConfig:
    learning_rate: float = 1e-3
    target_tau: float = 0.005
    target_update_interval: int = 1
    updates_per_rollout: int = 4
    max_grad_norm: float = 1.0
    initialization_seed: int = 845_219
    visitation_epsilon: float = 1e-8

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError("target_tau must lie in (0, 1]")
        if self.target_update_interval < 1:
            raise ValueError("target_update_interval must be at least 1")
        if self.updates_per_rollout < 1:
            raise ValueError("updates_per_rollout must be at least 1")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
