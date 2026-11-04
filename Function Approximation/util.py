import torch


def simple_acrobot_policy(observation):
    """
    A simple fixed rule policy for Acrobot-v1.

    Observation array components:
    obs[0] = cos(q1)
    obs[1] = sin(q1)
    obs[2] = cos(q2)
    obs[3] = sin(q2)
    obs[4] = angular velocity of joint 1 (q1)
    obs[5] = angular velocity of joint 2 (q2)
    action[0] = apply -1 torque to the actuated joint
    action[1] = apply 0 torque to the actuated joint
    action[2] = apply 1 torque to the actuated joint
    """
    _cos_q1, _sin_q1, _cos_q2, _sin_q2, q1_dot, q2_dot = observation

    # Rule 1: Pump energy based on velocity direction of joint 1
    # Swing link 2 in the opposite direction of joint 1 movement
    if q1_dot > 0.1:
        return 0  # Apply negative torque (-1)
    elif q1_dot < -0.1:
        return 2  # Apply positive torque (+1)
    else:
        # If joint 1 velocity is low, add a kick to get started
        return 2 if q2_dot >= 0 else 0


def collect_episode_mc(env, policy):
    """
    Run one peisode and stores the encountered states and rewards
    """
    states = []
    rewards = []

    state, _info = env.reset()
    done = False

    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, _info = env.step(action)
        states.append(state)
        rewards.append(reward)
        if terminated or truncated:
            done = True
        state = next_state
    return states, rewards


def collect_episode_td(env, policy, value_network, gamma):
    states = []
    targets = []
    state, _info = env.reset()
    done = False

    value_network.eval()
    with torch.no_grad():
        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _info = env.step(action)

            if terminated:
                target = reward
            else:
                next_state_tensor = torch.tensor(
                    next_state, dtype=torch.float32
                ).unsqueeze(0)
                target = reward + value_network(next_state_tensor).item() * gamma
            states.append(state)
            targets.append(target)
            if terminated or truncated:
                done = True
            state = next_state
    value_network.train()
    return states, targets


def discounted_reward(rewards, gamma):
    """
    A function to compute monte carlo returns
    """
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def evaluate_value_network_mc(env, policy, value_network, gamma, num_episodes):
    sum_squared_error_list = []

    value_network.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            states, rewards = collect_episode_mc(env, policy)
            returns = discounted_reward(rewards, gamma)

            states_tensors = torch.tensor(states)
            returns_tensors = torch.tensor(returns)
            predictions = value_network(states_tensors)
            loss = 0.5 * (returns_tensors - predictions).pow(2)
            sum_squared_error_list.extend(loss.tolist())
    value_network.train()
    return sum(sum_squared_error_list) / len(sum_squared_error_list)


def evaluate_value_network_td(env, policy, value_network, gamma, num_episodes):
    sum_squared_error_list = []

    value_network.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            states, targets = collect_episode_td(env, policy, value_network, gamma)

            states_tensors = torch.tensor(states)
            targets_tensor = torch.tensor(targets)
            predictions = value_network(states_tensors)
            loss = 0.5 * (targets_tensor - predictions).pow(2)
            sum_squared_error_list.extend(loss.tolist())
    value_network.train()
    return sum(sum_squared_error_list) / len(sum_squared_error_list)
