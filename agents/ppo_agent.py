# PPO_template.py
# Project-specific PPO implementation (on-policy), designed to work with the existing simulation codebase.
# Notes:
# - Rollout buffer is collected within an episode.
# - train_step() is intended to run at the end of the episode.
# - Function signatures keep compatibility with the DQN interface used elsewhere in the project.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import combinations
from torch.distributions import Categorical


class PPOPolicyNetwork(nn.Module):
    # Actor network: maps state -> action logits
    def __init__(self, input_dim, output_dim, hidden_layers, activation="tanh"):
        super(PPOPolicyNetwork, self).__init__()
        layers = []
        prev_dim = input_dim

        # Build MLP body
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))

            # Activation selection
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            prev_dim = h

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        # Returns action logits (Categorical distribution will be formed from logits)
        x = self.hidden_layers(x)
        return self.output_layer(x)  # logits


class PPOPairScoringPolicyNetwork(nn.Module):
    """Shared scorer for unordered server-pair actions.

    The observation remains a block-layout state. Each pair is represented by
    the symmetric feature vector [mean, abs-difference, rho, task] and all
    action scores are produced by one shared scorer network.
    """

    node_feature_dim = 4
    task_feature_dim = 3
    pair_feature_dim = 12

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers,
        num_servers,
        pair_correlations,
        activation='tanh',
    ):
        super().__init__()
        try:
            num_servers = int(num_servers)
        except (TypeError, ValueError) as exc:
            raise ValueError('num_servers must be a positive integer') from exc
        if num_servers < 2:
            raise ValueError('num_servers must be at least 2')
        expected_input_dim = self.node_feature_dim * num_servers + self.task_feature_dim
        if int(input_dim) != expected_input_dim:
            raise ValueError(
                'pair_scoring actor expects num_states == '
                f'4 * num_servers + 3 ({expected_input_dim}), got {input_dim}'
            )
        expected_actions = num_servers * (num_servers - 1) // 2
        if int(output_dim) != expected_actions:
            raise ValueError(
                'pair_scoring actor expects num_actions == '
                f'num_servers * (num_servers - 1) // 2 ({expected_actions}), '
                f'got {output_dim}'
            )
        correlations = np.asarray(pair_correlations, dtype=float)
        if correlations.ndim != 1 or correlations.shape[0] != expected_actions:
            raise ValueError(
                'pair_correlations must have shape '
                f'({expected_actions},), got {correlations.shape}'
            )
        if not np.isfinite(correlations).all():
            raise ValueError('pair_correlations must contain only finite values')
        if (correlations < -1e-10).any() or (correlations > 1.0 + 1e-10).any():
            raise ValueError('pair_correlations must lie in [0, 1]')
        correlations = np.clip(correlations, 0.0, 1.0)

        self.num_servers = num_servers
        self.num_actions = expected_actions
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.register_buffer(
            'pair_indices',
            torch.tensor(list(combinations(range(num_servers), 2)), dtype=torch.long),
        )
        self.register_buffer(
            'pair_correlations',
            torch.tensor(correlations, dtype=torch.float32),
        )

        layers = []
        prev_dim = self.pair_feature_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f'Unsupported activation function: {activation}')
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 1))
        self.scorer = nn.Sequential(*layers)

    @staticmethod
    def compose_pair_features(node_j, node_k, spatial_risk_correlation, task_features):
        """Compose symmetric [mean, absolute difference, rho, task] features."""
        node_j = torch.as_tensor(node_j, dtype=torch.float32)
        node_k = torch.as_tensor(node_k, dtype=torch.float32, device=node_j.device)
        task_features = torch.as_tensor(
            task_features, dtype=torch.float32, device=node_j.device
        )
        rho = torch.as_tensor(
            spatial_risk_correlation, dtype=torch.float32, device=node_j.device
        )
        if node_j.shape != node_k.shape or node_j.shape[-1] != 4:
            raise ValueError('node_j and node_k must have matching shape [..., 4]')
        if task_features.shape[:-1] != node_j.shape[:-1] or task_features.shape[-1] != 3:
            raise ValueError('task_features must have shape [..., 3]')
        if rho.shape == node_j.shape[:-1]:
            rho = rho.unsqueeze(-1)
        if rho.shape != node_j.shape[:-1] + (1,):
            raise ValueError('spatial_risk_correlation must have shape [..., 1]')
        mean_features = (node_j + node_k) / 2.0
        absolute_difference = torch.abs(node_j - node_k)
        return torch.cat((mean_features, absolute_difference, rho, task_features), dim=-1)

    def _batched_state(self, state):
        state = torch.as_tensor(
            state, dtype=torch.float32, device=self.pair_correlations.device
        )
        single = state.ndim == 1
        if single:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[-1] != self.input_dim:
            raise ValueError(
                f'state must have shape [{self.input_dim}] or [B, {self.input_dim}], '
                f'got {tuple(state.shape)}'
            )
        return state, single

    def build_node_features(self, state):
        """Recover [failure, frequency, backlog, uplink] per node."""
        state, single = self._batched_state(state)
        n = self.num_servers
        node_features = torch.stack(
            (
                state[:, 0:n],
                state[:, n : 2 * n],
                state[:, 2 * n : 3 * n],
                state[:, 3 * n : 4 * n],
            ),
            dim=-1,
        )
        return node_features[0] if single else node_features

    def build_pair_features(self, state):
        """Build pair features in the fixed action-index order."""
        state, single = self._batched_state(state)
        n = self.num_servers
        node_features = self.build_node_features(state)
        if node_features.ndim == 2:
            node_features = node_features.unsqueeze(0)
        pair_indices = self.pair_indices.to(device=state.device)
        node_j = node_features[:, pair_indices[:, 0], :]
        node_k = node_features[:, pair_indices[:, 1], :]
        task_features = state[:, 4 * n : 4 * n + 3]
        rho = self.pair_correlations.to(device=state.device).view(1, -1)
        rho = rho.expand(state.shape[0], -1)
        pair_features = self.compose_pair_features(
            node_j,
            node_k,
            rho,
            task_features.unsqueeze(1).expand(-1, self.num_actions, -1),
        )
        return pair_features[0] if single else pair_features

    def forward(self, state):
        single = torch.as_tensor(state).ndim == 1
        pair_features = self.build_pair_features(state)
        if single:
            pair_features = pair_features.unsqueeze(0)
        logits = self.scorer(pair_features.reshape(-1, self.pair_feature_dim))
        logits = logits.reshape(-1, self.num_actions)
        return logits[0] if single else logits


class PPOValueNetwork(nn.Module):
    # Critic network: maps state -> scalar value V(s)
    def __init__(self, input_dim, hidden_layers, activation="tanh"):
        super(PPOValueNetwork, self).__init__()
        layers = []
        prev_dim = input_dim

        # Build MLP body
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))

            # Activation selection
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            prev_dim = h

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x):
        # Squeeze last dimension to return shape [batch] instead of [batch, 1]
        x = self.hidden_layers(x)
        return self.output_layer(x).squeeze(-1)  # V(s)


class PPOAgent:
    """
    Project-specific PPO agent (on-policy).

    Key assumptions in this project:
    - Rollout data is collected only for the current episode.
    - train_step() is called at the end of the episode.
    """

    def __init__(
        self,
        num_states,
        num_actions,
        hidden_layers,
        device="cpu",
        gamma=0.99,
        actor_lr=3e-4,
        critic_lr=1e-3,
        clip_eps=0.2,
        k_epochs=2,
        batch_size=64,
        entropy_coef=0.01,
        reward_scale=1.0,
        gae_lambda=0.95,        # GAE coefficient for ordered event-driven rollout
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        activation="tanh",
        min_rollout=8,          # Minimum number of transitions required before performing an update
        minibatch_seed=2028,
        actor_mode="flat",
        num_servers=None,
        pair_correlations=None,
    ):
        self.device = torch.device(device)
        self.num_actions = int(num_actions)
        self.actor_mode = str(actor_mode).strip().lower()
        if self.actor_mode not in {"flat", "pair_scoring"}:
            raise ValueError(
                "actor_mode must be either 'flat' or 'pair_scoring', "
                f"got {actor_mode!r}"
            )
        if self.actor_mode == "pair_scoring":
            if num_servers is None or pair_correlations is None:
                raise ValueError(
                    "pair_scoring mode requires num_servers and pair_correlations"
                )
            self.num_servers = int(num_servers)
            expected_actions = self.num_servers * (self.num_servers - 1) // 2
            correlations = np.asarray(pair_correlations, dtype=float)
            if correlations.shape != (expected_actions,):
                raise ValueError(
                    "pair_correlations must have shape "
                    f"({expected_actions},), got {correlations.shape}"
                )
            if not np.isfinite(correlations).all() or (
                (correlations < -1e-10).any()
                or (correlations > 1.0 + 1e-10).any()
            ):
                raise ValueError("pair_correlations must be finite and lie in [0, 1]")
            if self.num_actions != expected_actions:
                raise ValueError(
                    "pair_scoring mode requires num_actions == "
                    f"{expected_actions}, got {self.num_actions}"
                )
            self.pair_correlations = np.clip(correlations, 0.0, 1.0)
        else:
            self.num_servers = None
            self.pair_correlations = None

        # PPO / RL hyperparameters
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.min_rollout = int(min_rollout)
        self.minibatch_rng = np.random.default_rng(minibatch_seed)

        # Policy networks:
        # - policy_net: trainable policy
        # - policy_old: frozen snapshot used for sampling and stable old log-prob computation
        actor_class = (
            PPOPairScoringPolicyNetwork
            if self.actor_mode == "pair_scoring"
            else PPOPolicyNetwork
        )
        if self.actor_mode == "pair_scoring":
            actor_kwargs = {
                "num_servers": self.num_servers,
                "pair_correlations": self.pair_correlations,
            }
        else:
            actor_kwargs = {}
        self.policy_net = actor_class(
            num_states, num_actions, hidden_layers, activation=activation, **actor_kwargs
        ).to(self.device)

        self.policy_old = actor_class(
            num_states, num_actions, hidden_layers, activation=activation, **actor_kwargs
        ).to(self.device)
        self.policy_old.load_state_dict(self.policy_net.state_dict())

        # Value network (critic)
        self.value_net = PPOValueNetwork(
            num_states, hidden_layers, activation
        ).to(self.device)

        # Separate optimizers for actor and critic
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=actor_lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=critic_lr)

        # Episode rollout buffer (cleared after train_step)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.old_log_probs = []
        self.delta_times = []
        self.task_ids = []
        self.task_id_to_transition_index = {}
        self.pending_task_rewards = {}

        # Compatibility placeholder:
        # Some parts of the project check agent.replay_buffer length (DQN-style).
        self.replay_buffer = []

    # -----------------------------
    # utilities
    # -----------------------------
    def _to_tensor(self, x):
        # Convert numpy-like input to float32 tensor on target device
        return torch.tensor(np.array(x), dtype=torch.float32, device=self.device)

    # -----------------------------
    # action selection
    # -----------------------------
    def select_action(self, state, epsilon, use_softmax=False, temperature=1.5):
        """
        PPO action selection.
        NOTE: The signature matches other agents in this project for compatibility.
              epsilon/use_softmax/temperature are not used by PPO here.
        """
        state_tensor = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            # Sample from policy_old for more stable on-policy behavior
            logits = self.policy_old(state_tensor)
            if not torch.isfinite(logits).all():
                # Fallback: if logits become invalid, return a random action
                return int(np.random.randint(0, self.num_actions))
            dist = Categorical(logits=logits)
            action = dist.sample()
        return int(action.item())

    # -----------------------------
    # store transition
    # -----------------------------
    def store_transition(self, s, a, r, s_next, delta_t, done=False, task_id=None):
        """Store one arrival-ordered SMDP transition bound to its origin task.

        ``delta_t`` remains the elapsed simulation time to the next arrival
        decision (or terminal drain).  A reward may be unresolved when the
        transition shell is created; :meth:`assign_task_reward` fills it in
        later, or a previously pending reward is consumed immediately.
        """
        if task_id is None:
            raise ValueError("PPO store_transition requires a task_id")
        if task_id in self.task_id_to_transition_index:
            raise RuntimeError(f"Duplicate PPO transition task_id: {task_id}")

        delta_t = float(delta_t)
        if not np.isfinite(delta_t):
            raise ValueError(f"PPO delta_t must be finite, got {delta_t}")
        if delta_t < -1e-8:
            raise ValueError(f"PPO delta_t cannot be negative, got {delta_t}")
        delta_t = max(delta_t, 0.0)

        resolved_reward = None if r is None else float(r)
        if task_id in self.pending_task_rewards:
            pending_reward = self.pending_task_rewards.pop(task_id)
            if resolved_reward is not None and not np.isclose(
                resolved_reward, pending_reward, rtol=0.0, atol=1e-12
            ):
                raise RuntimeError(
                    f"Conflicting PPO rewards for task_id {task_id}: "
                    f"pending={pending_reward}, direct={resolved_reward}"
                )
            resolved_reward = pending_reward

        transition_index = len(self.states)
        self.states.append(np.array(s, copy=True))
        self.actions.append(int(a))
        self.rewards.append(resolved_reward)
        self.next_states.append(np.array(s_next, copy=True))
        self.dones.append(bool(done))
        self.delta_times.append(delta_t)
        self.task_ids.append(task_id)
        self.task_id_to_transition_index[task_id] = transition_index

        # Store old log-prob using policy_old (standard PPO approach).
        with torch.no_grad():
            s_tensor = self._to_tensor(s).unsqueeze(0)
            logits = self.policy_old(s_tensor)
            if not torch.isfinite(logits).all():
                self.old_log_probs.append(0.0)
            else:
                dist = Categorical(logits=logits)
                a_tensor = torch.tensor(int(a), dtype=torch.int64, device=self.device)
                log_prob = dist.log_prob(a_tensor).item()
                if not np.isfinite(log_prob):
                    log_prob = 0.0
                self.old_log_probs.append(float(log_prob))

    def assign_task_reward(self, task_id, reward):
        """Assign one resolved task reward to its origin transition.

        Resolution may happen before or after the arrival transition shell is
        created.  Every task may be assigned at most once.
        """
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError(f"PPO task reward must be finite, got {reward}")
        if task_id in self.task_id_to_transition_index:
            index = self.task_id_to_transition_index[task_id]
            if self.rewards[index] is not None:
                raise RuntimeError(f"Duplicate PPO reward assignment for task_id {task_id}")
            self.rewards[index] = reward
            return
        if task_id in self.pending_task_rewards:
            raise RuntimeError(f"Duplicate PPO reward assignment for task_id {task_id}")
        self.pending_task_rewards[task_id] = reward

    # -----------------------------
    # training: end of episode
    # -----------------------------
    def _shuffled_indices(self, n):
        """Return a minibatch permutation without consuming global NumPy RNG."""
        if n < 0:
            raise ValueError("n must be non-negative")
        indices = np.arange(n)
        self.minibatch_rng.shuffle(indices)
        return indices

    def train_step(self):
        """Optimize the arrival-ordered event-driven PPO rollout."""
        N = len(self.states)
        if N == 0:
            if self.pending_task_rewards:
                self.clear_rollout()
                raise RuntimeError("PPO rollout has pending rewards without transitions")
            return

        if not (
            len(self.actions) == len(self.rewards) == len(self.next_states)
            == len(self.dones) == len(self.old_log_probs) == len(self.delta_times)
            == len(self.task_ids) == N
        ):
            self.clear_rollout()
            raise RuntimeError("PPO rollout buffers have inconsistent lengths")

        if (
            len(self.task_id_to_transition_index) != N
            or len(set(self.task_ids)) != N
            or any(
                self.task_id_to_transition_index.get(task_id) != index
                for index, task_id in enumerate(self.task_ids)
            )
            or self.pending_task_rewards
            or any(reward is None for reward in self.rewards)
        ):
            self.clear_rollout()
            raise RuntimeError(
                "PPO rollout task/reward bookkeeping is unresolved or inconsistent"
            )
        try:
            strictly_increasing = all(
                self.task_ids[index] < self.task_ids[index + 1]
                for index in range(N - 1)
            )
        except TypeError as exc:
            self.clear_rollout()
            raise RuntimeError("PPO task_ids must be orderable") from exc
        if not strictly_increasing:
            self.clear_rollout()
            raise RuntimeError("PPO task_ids must be strictly increasing")

        # A normal episode must explicitly provide a terminal transition.
        if not any(self.dones):
            self.clear_rollout()
            raise RuntimeError("PPO rollout is missing its terminal transition")

        if N < self.min_rollout:
            self.clear_rollout()
            return

        states = self._to_tensor(self.states)
        next_states = self._to_tensor(self.next_states)
        actions = torch.tensor(self.actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(
            self.rewards, dtype=torch.float32, device=self.device
        ) * float(self.reward_scale)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        delta_times = torch.tensor(
            self.delta_times, dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.tensor(
            self.old_log_probs, dtype=torch.float32, device=self.device
        )

        if (
            not torch.isfinite(states).all()
            or not torch.isfinite(next_states).all()
            or not torch.isfinite(rewards).all()
            or not torch.isfinite(delta_times).all()
            or torch.any(delta_times < 0.0)
        ):
            self.clear_rollout()
            return

        values = self.value_net(states)

        with torch.no_grad():
            next_values = self.value_net(next_states)
            # gamma_ppo is interpreted as a per-second discount base.
            gamma_k = torch.pow(
                torch.full_like(delta_times, float(self.gamma)), delta_times
            )
            nonterminal = 1.0 - dones

            # Ordered, variable-discount GAE. The reverse pass must happen
            # before any minibatch shuffling.
            deltas = (
                rewards
                + gamma_k * next_values * nonterminal
                - values.detach()
            )
            advantages = torch.zeros_like(rewards)
            gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            for k in reversed(range(N)):
                gae = (
                    deltas[k]
                    + gamma_k[k] * self.gae_lambda * nonterminal[k] * gae
                )
                advantages[k] = gae

            returns = advantages + values.detach()

            if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
                self.clear_rollout()
                return

            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            if (not torch.isfinite(adv_std)) or adv_std.item() < 1e-8:
                advantages = advantages - adv_mean
            else:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            if not torch.isfinite(advantages).all():
                self.clear_rollout()
                return

        batch_size = min(int(self.batch_size), N)

        for _ in range(int(self.k_epochs)):
            # Minibatch shuffling is safe only after ordered GAE is complete.
            idx = self._shuffled_indices(N)

            for start in range(0, N, batch_size):
                end = start + batch_size
                batch_idx = idx[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx].detach()
                b_old_log_probs = old_log_probs[batch_idx]

                logits = self.policy_net(b_states)
                if not torch.isfinite(logits).all():
                    continue

                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                log_ratio = torch.clamp(log_probs - b_old_log_probs, -20.0, 20.0)
                ratios = torch.exp(log_ratio)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(
                    ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * b_advantages
                policy_loss = (
                    -torch.min(surr1, surr2).mean()
                    - self.entropy_coef * entropy
                )

                new_values = self.value_net(b_states)
                value_loss = self.value_loss_coef * nn.MSELoss()(new_values, b_returns)
                total_loss = policy_loss + value_loss

                if not torch.isfinite(total_loss):
                    continue

                self.optimizer_policy.zero_grad(set_to_none=True)
                self.optimizer_value.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer_policy.step()
                self.optimizer_value.step()

        self.policy_old.load_state_dict(self.policy_net.state_dict())
        self.clear_rollout()

    # -----------------------------
    # saving / loading
    # -----------------------------
    def save_model(self, path):
        # Save actor/critic parameters and optimizer states
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "policy_old": self.policy_old.state_dict(),
                "value_net": self.value_net.state_dict(),
                "opt_policy": self.optimizer_policy.state_dict(),
                "opt_value": self.optimizer_value.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        # Load actor/critic parameters and (optionally) optimizer states
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        if "policy_old" in checkpoint:
            self.policy_old.load_state_dict(checkpoint["policy_old"])
        else:
            self.policy_old.load_state_dict(checkpoint["policy_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        if "opt_policy" in checkpoint:
            self.optimizer_policy.load_state_dict(checkpoint["opt_policy"])
        if "opt_value" in checkpoint:
            self.optimizer_value.load_state_dict(checkpoint["opt_value"])

    # -----------------------------
    # rollout management
    # -----------------------------
    def clear_rollout(self):
        # Clear per-episode rollout buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.old_log_probs.clear()
        self.delta_times.clear()
        self.task_ids.clear()
        self.task_id_to_transition_index.clear()
        self.pending_task_rewards.clear()
