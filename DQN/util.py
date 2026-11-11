"""Replay buffer, policies, and update rules shared by the DQN variants."""

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# One transition: (state, action, reward, next_state, terminated)
Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


def epsilon_greedy_policy(
    env: gym.Env, q_net: nn.Module, state: np.ndarray, epsilon: float
) -> int:
    """Pick a random action with probability `epsilon`, else the greedy action."""
    if np.random.random() < epsilon:
        return env.action_space.sample()
    state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(state_tensor)
    return q_values.argmax(dim=1).item()


class ReplayBuffer:
    """Fixed-size circular buffer of transitions with uniform sampling."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> tuple[tuple, ...]:
        batch = random.sample(self.buffer, batch_size)
        return tuple(zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


def update_network(target_network: nn.Module, online_network: nn.Module) -> None:
    """Copy weights from the online network to the target network."""
    target_network.load_state_dict(online_network.state_dict())


def _batch_to_tensors(
    batch: tuple[tuple, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states, actions, rewards, next_states, terminated = batch
    return (
        torch.as_tensor(np.asarray(states), dtype=torch.float32),
        torch.as_tensor(np.asarray(actions), dtype=torch.int64),
        torch.as_tensor(np.asarray(rewards), dtype=torch.float32),
        torch.as_tensor(np.asarray(next_states), dtype=torch.float32),
        torch.as_tensor(np.asarray(terminated), dtype=torch.float32),
    )


def _gradient_step(
    optimizer: torch.optim.Optimizer,
    network: nn.Module,
    loss: torch.Tensor,
    max_grad_norm: float,
) -> float:
    optimizer.zero_grad()
    loss.backward()
    # Clipping keeps occasional large TD errors from destabilizing training.
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()


def dqn_update(
    target_network: nn.Module,
    online_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    max_grad_norm: float = 10.0,
) -> float:
    """One standard DQN update: the target network both selects and evaluates
    the next action, which can overestimate Q-values."""
    states, actions, rewards, next_states, terminated = _batch_to_tensors(
        replay_buffer.sample(batch_size)
    )

    q_values = online_network(states)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        max_next_q = target_network(next_states).max(dim=1).values
        targets = rewards + gamma * (1 - terminated) * max_next_q

    loss = F.mse_loss(q_sa, targets)
    return _gradient_step(optimizer, online_network, loss, max_grad_norm)


def double_dqn_update(
    target_network: nn.Module,
    online_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    max_grad_norm: float = 10.0,
) -> float:
    """One Double DQN update: the online network selects the next action and
    the target network evaluates it, reducing overestimation bias."""
    states, actions, rewards, next_states, terminated = _batch_to_tensors(
        replay_buffer.sample(batch_size)
    )

    q_values = online_network(states)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = online_network(next_states).argmax(dim=1)
        next_q = target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + gamma * (1 - terminated) * next_q

    loss = F.mse_loss(q_sa, targets)
    return _gradient_step(optimizer, online_network, loss, max_grad_norm)


def linear_epsilon_schedule(
    step: int, epsilon_start: float = 1.0, epsilon_end: float = 0.05, decay_steps: int = 20000
) -> float:
    """Linearly anneal epsilon from `epsilon_start` to `epsilon_end`."""
    fraction = min(step / decay_steps, 1)
    return epsilon_start + fraction * (epsilon_end - epsilon_start)


def evaluate_dqn(env: gym.Env, q_net: nn.Module, num_episodes: int = 20) -> float:
    """Mean return of the greedy policy over `num_episodes` episodes.

    Temporarily switches the network to eval mode and restores the previous
    mode afterwards.
    """
    was_training = q_net.training
    q_net.eval()
    returns = []
    for _ in range(num_episodes):
        state, _info = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = epsilon_greedy_policy(env, q_net, state, epsilon=0.0)
            state, reward, terminated, truncated, _info = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
    if was_training:
        q_net.train()
    return float(np.mean(returns))

