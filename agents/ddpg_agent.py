import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_torch_tensor(x, dtype=torch.float32, device="cpu"):
    """
    Convert NumPy arrays, lists, or PyTorch tensors to the requested device and dtype.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    x = np.asarray(x, dtype=np.float32)
    return torch.tensor(x, dtype=dtype, device=device)


class _BaseNet(nn.Module):
    @property
    def variables(self):
        return list(self.parameters())


class _ActorNet(_BaseNet):
    def __init__(self, num_states, num_actions, activation_fn: str):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.activation_fn = str(activation_fn).lower()

        self.ln_in = nn.LayerNorm(num_states)
        self.fc1 = nn.Linear(num_states, 300)
        self.fc2 = nn.Linear(300, 200)
        self.out = nn.Linear(200, num_actions)

        nn.init.uniform_(self.out.weight, a=-0.003, b=0.003)
        nn.init.uniform_(self.out.bias, a=-0.003, b=0.003)

    def forward(self, x):
        x = self.ln_in(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        if self.activation_fn == "tanh":
            x = torch.tanh(x)
        elif self.activation_fn == "softmax":
            x = F.softmax(x, dim=-1)
        elif self.activation_fn == "sigmoid":
            x = torch.sigmoid(x)
        else:
            # Unknown activation: leave linear
            pass

        return x


class _CriticNet(_BaseNet):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.s_fc1 = nn.Linear(num_states, 300)
        self.s_fc2 = nn.Linear(300, 200)

        self.a_fc1 = nn.Linear(num_actions, 200)

        self.ln_concat = nn.LayerNorm(400)
        self.c_fc1 = nn.Linear(400, 200)
        self.out = nn.Linear(200, 1)

    def forward(self, state, action):
        s = F.relu(self.s_fc1(state))
        s = F.relu(self.s_fc2(s))

        a = F.relu(self.a_fc1(action))

        x = torch.cat([s, a], dim=-1)
        x = self.ln_concat(x)
        x = F.relu(self.c_fc1(x))
        return self.out(x)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)


class Buffer:
    def __init__(self, ddpgObj, buffer_capacity=100000, batch_size=64):
        self.ddpgObj = ddpgObj
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.ddpgObj.num_states), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, self.ddpgObj.num_actions), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.ddpgObj.num_states), dtype=np.float32)

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with torch.no_grad():
            target_actions = self.ddpgObj.target_actor(next_state_batch)
            y = reward_batch + self.ddpgObj.gamma * self.ddpgObj.target_critic(next_state_batch, target_actions)

        critic_value = self.ddpgObj.critic_model(state_batch, action_batch)
        critic_loss = torch.mean((y - critic_value) ** 2)

        self.ddpgObj.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.ddpgObj.critic_optimizer.step()

        actions = self.ddpgObj.actor_model(state_batch)
        actor_value = self.ddpgObj.critic_model(state_batch, actions)
        actor_loss = -torch.mean(actor_value)

        self.ddpgObj.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.ddpgObj.actor_optimizer.step()

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        if record_range < self.batch_size:
            return

        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = _to_torch_tensor(self.state_buffer[batch_indices], device=self.ddpgObj.device)
        action_batch = _to_torch_tensor(self.action_buffer[batch_indices], device=self.ddpgObj.device)
        reward_batch = _to_torch_tensor(self.reward_buffer[batch_indices], device=self.ddpgObj.device)
        next_state_batch = _to_torch_tensor(self.next_state_buffer[batch_indices], device=self.ddpgObj.device)

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


class ddpgModel:
    def __init__(self, num_states, num_actions, std_dev, critic_lr, actor_lr, gamma, tau, activationFunction,
                 buffer_capacity=100000, batch_size=64):
        self.activationFunction = activationFunction
        self.num_states = num_states
        self.num_actions = num_actions
        self.std_dev = std_dev
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.tau = tau

        self.device = "cpu"

        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1), theta=0.2)

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        self.target_actor.load_state_dict(self.actor_model.state_dict())
        self.target_critic.load_state_dict(self.critic_model.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)

        self.buffer = Buffer(self, buffer_capacity=buffer_capacity, batch_size=batch_size)

    def update_target(self, target_weights, weights):
        with torch.no_grad():
            for (a, b) in zip(target_weights, weights):
                a.copy_(b * self.tau + a * (1.0 - self.tau))

    def get_actor(self):
        return _ActorNet(self.num_states, self.num_actions, self.activationFunction).to(self.device)

    def get_critic(self):
        return _CriticNet(self.num_states, self.num_actions).to(self.device)

    def addNoise(self, sampled_actions, thisEpNo, totalEpNo):
        noise = self.ou_noise()
        noisy_actions = []
        for sa in sampled_actions:
            sa = sa + noise[0]
            sa = np.clip(sa, -1, 1)
            noisy_actions.append(sa)
        return noisy_actions

    def policy(self, state):
        s = _to_torch_tensor(state, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)

        with torch.no_grad():
            a = self.actor_model(s)
            a = a.squeeze(0)

        return a.detach().cpu()
