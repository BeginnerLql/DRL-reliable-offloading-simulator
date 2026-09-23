"""Independent masked Pair PPO experiment/deployment defaults.

The existing PPO_ACTOR_MODE and legacy PPO settings remain unchanged.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MaskedPairPPOConfig:
    agent_name: str = 'reliability_masked_pair_ppo'
    deployment_mode: str = 'stochastic'
    fallback: str = 'max_reliability_actor_weighted_ties'
    smoke_train_episodes: int = 100
    smoke_eval_episodes: int = 20
    tasks_per_episode: int = 200


MASKED_PAIR_PPO = MaskedPairPPOConfig()
