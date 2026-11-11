import gymnasium as gym
import torch
from torch import optim
from util import (
    collect_episode_mc,
    collect_episode_td,
    discounted_reward,
    evaluate_value_network_mc,
    evaluate_value_network_td,
    simple_acrobot_policy,
)
from value_prediction import ValueNetwork

gamma = 0.9
learning_rate = 0.001
num_episodes = 1000

env = gym.make("Acrobot-v1")

value_network_mc = ValueNetwork(observation_dim=env.observation_space.shape[0])
optimizer_mc = optim.Adam(value_network_mc.parameters(), lr=learning_rate)

value_network_td = ValueNetwork(observation_dim=env.observation_space.shape[0])
optimizer_td = optim.Adam(value_network_td.parameters(), lr=learning_rate)
# ====================================
# Monte Carlo
# ====================================
for episode in range(num_episodes):
    states, rewards = collect_episode_mc(env, simple_acrobot_policy)
    returns = discounted_reward(rewards, gamma)
    states_tensor = torch.tensor(states)
    returns_tensor = torch.tensor(returns)

    predictions = value_network_mc(states_tensor)
    loss = 0.5 * (returns_tensor - predictions).pow(2).mean()

    optimizer_mc.zero_grad()
    loss.backward()
    optimizer_mc.step()

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}:Length = {len(rewards)}Loss = {loss.item():.3f}")


evaluation_mc = evaluate_value_network_mc(
    env, simple_acrobot_policy, value_network_mc, gamma, num_episodes
)
print("Evaluation of Monte Carlo: ", evaluation_mc)

# ====================================
# TD
# ====================================
for episode in range(num_episodes):
    states, targets = collect_episode_td(
        env, simple_acrobot_policy, value_network_td, gamma
    )
    states_tensor = torch.tensor(states)
    targets_tensor = torch.tensor(targets)

    predictions = value_network_td(states_tensor)
    loss = 0.5 * (targets_tensor - predictions).pow(2).mean()

    optimizer_td.zero_grad()
    loss.backward()
    optimizer_td.step()

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}:Length = {len(targets)}Loss = {loss.item():.3f}")

evaluation_td = evaluate_value_network_td(
    env, simple_acrobot_policy, value_network_td, gamma, num_episodes
)
print("Evaluation of Temporal Difference ", evaluation_td)

env.close()
