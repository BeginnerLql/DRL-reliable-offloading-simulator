import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent, masked_logits
from agents.policy_centered_action_advantage import (
    PolicyCenteredActionAdvantage,
    policy_centered_advantage,
)
from config.params import params


def _rho():
    return np.linspace(0.01, 0.95, 28, dtype=float)


def _side(seed=17, **kwargs):
    return PolicyCenteredActionAdvantage(
        params.num_states, params.num_actions, params.hidden_layers_ppo,
        params.serverNo, _rho(), activation=params.af_ppo,
        initialization_seed=seed, **kwargs,
    )


def test_policy_centering_excludes_unsafe_actions_and_centers_each_batch_row():
    scores = torch.tensor([[1.0, 2.0, 100.0], [-9.0, 4.0, 2.0]])
    probabilities = torch.tensor([[0.25, 0.75, 0.0], [0.0, 0.4, 0.6]])
    masks = torch.tensor([[True, True, False], [False, True, True]])
    centered = policy_centered_advantage(scores, probabilities, masks)
    expected = torch.tensor([[-0.75, 0.25, 0.0], [0.0, 1.2, -0.8]])
    assert torch.allclose(centered, expected, atol=1e-6)
    residual = (centered * probabilities).sum(dim=1)
    assert torch.max(torch.abs(residual)).item() < 1e-6


def test_single_effective_action_has_zero_centered_advantage():
    scores = torch.tensor([[1.0, -5.0, 9.0]])
    probabilities = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, False, False]])
    assert torch.equal(policy_centered_advantage(scores, probabilities, mask), torch.zeros_like(scores))


def test_selected_action_loss_rejects_unsafe_actions():
    learner = _side()
    states = torch.rand(2, params.num_states)
    masks = torch.tensor([[True] + [False] * 27, [False, True] + [False] * 26])
    probabilities = masks.float()
    with pytest.raises(ValueError, match="outside its effective mask"):
        learner.loss_for_selected(states, [1, 1], [0.0, 0.0], probabilities, masks)


def test_batch_masks_do_not_leak_across_rows():
    learner = _side()
    states = torch.rand(2, params.num_states)
    masks = torch.zeros((2, 28), dtype=torch.bool)
    masks[0, [0, 1]] = True
    masks[1, [12, 27]] = True
    probs = masks.float() / 2.0
    raw, batch_centered = learner.centered(states, probs, masks)
    first = policy_centered_advantage(raw[:1], probs[:1], masks[:1])
    second = policy_centered_advantage(raw[1:], probs[1:], masks[1:])
    assert torch.equal(batch_centered, torch.cat([first, second], dim=0))
    assert torch.count_nonzero(batch_centered[0, ~masks[0]]) == 0
    assert torch.count_nonzero(batch_centered[1, ~masks[1]]) == 0


def test_pair_mapping_permutation_and_pair_feature_gradient():
    learner = _side()
    states = torch.rand(8, params.num_states)
    checks = learner.wiring_diagnostics(states)
    assert checks["pair_feature_std_mean"] > 0
    assert checks["permutation_pass"]
    assert checks["pair_mapping_matches_combinations"]
    assert checks["pair_feature_gradient_nonzero"]
    assert checks["pair_feature_gradient_norm"] > 0


def _fill_synthetic_rollout(agent, n=16):
    rng = np.random.default_rng(555)
    states = rng.normal(size=(n, params.num_states)).astype(np.float32)
    next_states = rng.normal(size=(n, params.num_states)).astype(np.float32)
    mask = np.ones(params.num_actions, dtype=bool)
    for i in range(n):
        state_tensor = agent._to_tensor(states[i]).unsqueeze(0)
        with torch.no_grad():
            dist = Categorical(logits=masked_logits(agent.policy_old(state_tensor).squeeze(0), mask))
            action = int(i % params.num_actions)
            old_log_prob = float(dist.log_prob(torch.tensor(action)).item())
            probs = dist.probs.cpu().numpy().astype(np.float32)
        task_id = i + 1
        agent.states.append(states[i].copy())
        agent.actions.append(action)
        agent.rewards.append(float(rng.normal()))
        agent.next_states.append(next_states[i].copy())
        agent.dones.append(i == n - 1)
        agent.old_log_probs.append(old_log_prob)
        agent.delta_times.append(float(0.25 + rng.random()))
        agent.task_ids.append(task_id)
        agent.task_id_to_transition_index[task_id] = i
        agent.effective_masks.append(mask.copy())
        if agent.advantage_target_observer is not None:
            agent.rollout_old_policy_probabilities.append(probs.copy())
    agent._current_episode = 1


def test_observer_captures_exact_actor_used_advantage_and_does_not_change_ppo():
    torch.manual_seed(808)
    baseline = ReliabilityMaskedPairPPOAgent(
        params.num_states, params.num_actions, params.hidden_layers_ppo,
        gamma=params.gamma_ppo, actor_lr=params.actor_lr_ppo, critic_lr=params.critic_lr_ppo,
        clip_eps=params.clip_eps_ppo, k_epochs=params.k_epochs_ppo,
        batch_size=params.batch_size_ppo, entropy_coef=params.entropy_coef_ppo,
        reward_scale=params.reward_scale_ppo, gae_lambda=params.gae_lambda_ppo,
        value_loss_coef=params.value_loss_coef_ppo, max_grad_norm=params.max_grad_norm_ppo,
        activation=params.af_ppo, minibatch_seed=1234, actor_mode="pair_scoring",
        num_servers=params.serverNo, pair_correlations=_rho(),
    )
    torch.manual_seed(808)
    observed = ReliabilityMaskedPairPPOAgent(
        params.num_states, params.num_actions, params.hidden_layers_ppo,
        gamma=params.gamma_ppo, actor_lr=params.actor_lr_ppo, critic_lr=params.critic_lr_ppo,
        clip_eps=params.clip_eps_ppo, k_epochs=params.k_epochs_ppo,
        batch_size=params.batch_size_ppo, entropy_coef=params.entropy_coef_ppo,
        reward_scale=params.reward_scale_ppo, gae_lambda=params.gae_lambda_ppo,
        value_loss_coef=params.value_loss_coef_ppo, max_grad_norm=params.max_grad_norm_ppo,
        activation=params.af_ppo, minibatch_seed=1234, actor_mode="pair_scoring",
        num_servers=params.serverNo, pair_correlations=_rho(),
    )
    captured = []
    observed.advantage_target_observer = captured.append
    _fill_synthetic_rollout(baseline)
    _fill_synthetic_rollout(observed)
    baseline.train_step()
    observed.train_step()

    assert len(captured) == 1
    payload = captured[0]
    raw = payload["raw_gae_advantages"]
    expected = (raw - raw.mean()) / (raw.std(ddof=0) + 1e-8)
    assert np.allclose(payload["actor_used_advantages"], expected, rtol=0, atol=1e-6)
    assert not payload["actor_used_advantages"].flags.writeable
    for left, right in zip(baseline.policy_old.parameters(), observed.policy_old.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(baseline.value_net.parameters(), observed.value_net.parameters()):
        assert torch.equal(left, right)


def test_fixed_target_selected_action_fit_reduces_loss():
    torch.manual_seed(909)
    teacher, student = _side(seed=21), _side(seed=22, epochs_per_rollout=12, batch_size=32)
    rng = np.random.default_rng(901)
    n = 128
    states = torch.tensor(rng.normal(size=(n, params.num_states)).astype(np.float32))
    actions = torch.tensor(rng.integers(0, 28, size=n), dtype=torch.long)
    masks = torch.ones((n, 28), dtype=torch.bool)
    probabilities = torch.full((n, 28), 1.0 / 28)
    with torch.no_grad():
        raw = teacher(states)
        centered = policy_centered_advantage(raw, probabilities, masks)
        targets = centered.gather(1, actions[:, None]).squeeze(1)
    payload = {
        "episode": 1, "task_ids": np.arange(1, n + 1), "states": states.numpy(),
        "actions": actions.numpy(), "effective_masks": masks.numpy(),
        "old_policy_probabilities": probabilities.numpy(),
        "actor_used_advantages": targets.numpy(),
    }
    with torch.no_grad():
        initial = float(student.loss_for_selected(states, actions, targets, probabilities, masks)[0])
    student.train_rollout(payload)
    with torch.no_grad():
        final = float(student.loss_for_selected(states, actions, targets, probabilities, masks)[0])
    assert final < initial


def test_heldout_centered_scoring_is_read_only():
    learner = _side()
    before = {key: value.clone() for key, value in learner.network.state_dict().items()}
    state = torch.rand(1, params.num_states)
    mask = torch.ones(1, 28, dtype=torch.bool)
    probabilities = torch.full((1, 28), 1.0 / 28)
    with torch.no_grad():
        scores = learner.network(state)
        _ = policy_centered_advantage(scores, probabilities, mask)
    assert all(torch.equal(before[key], value) for key, value in learner.network.state_dict().items())
