from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from agents.ppo_agent import PPOContextPairScoringPolicyNetwork
from tools.run_context_masked_pair_ppo import (
    checkpoint_environment, validate_checkpoint_environment, load_verified_actor,
)


def test_environment_roundtrip_and_legacy_rejection():
    current = checkpoint_environment()
    validate_checkpoint_environment({'checkpoint_environment': deepcopy(current)}, current)
    with pytest.raises(ValueError, match='lacks an environment snapshot'):
        validate_checkpoint_environment({}, current)


@pytest.mark.parametrize('section,key,value', [
    ('data_sha256', 'server_info.xlsx', 'changed'),
    ('data_sha256', 'task_parameters.xlsx', 'changed'),
    ('configuration', 'SPATIAL_CORRELATION_LENGTH_KM', 999),
    ('configuration', 'BACKLOG_TIME_SCALE_SEC', 999),
    ('configuration', 'gamma_ppo', 0.1),
])
def test_environment_drift_rejected(section, key, value):
    current = checkpoint_environment()
    saved = deepcopy(current)
    saved[section][key] = value
    with pytest.raises(ValueError, match=section):
        validate_checkpoint_environment({'checkpoint_environment': saved}, current)


def make_agent():
    def network():
        return PPOContextPairScoringPolicyNetwork(15, 3, [8], 3, [.1, .2, .3])
    return SimpleNamespace(policy_net=network(), policy_old=network())


def test_verified_actor_restores_both_policies():
    source, target = make_agent(), make_agent()
    load_verified_actor(target, source.policy_net.state_dict())
    for network in (target.policy_net, target.policy_old):
        for key, value in source.policy_net.state_dict().items():
            assert torch.equal(value, network.state_dict()[key])


@pytest.mark.parametrize('key', ['pair_indices', 'pair_correlations'])
def test_static_buffers_checked_before_loading(key):
    agent = make_agent()
    before = deepcopy(agent.policy_net.state_dict())
    state = deepcopy(before)
    state[key].view(-1)[0] += 1
    with pytest.raises(ValueError, match=key):
        load_verified_actor(agent, state)
    for name, value in before.items():
        assert torch.equal(value, agent.policy_net.state_dict()[name])
