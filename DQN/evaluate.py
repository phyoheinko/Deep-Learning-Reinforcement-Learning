"""Evaluate a trained DQN checkpoint on Acrobot."""

import argparse
import sys

import gymnasium as gym
import numpy as np
import torch
from util import epsilon_greedy_policy

from dqn import DQN


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN on Acrobot.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="standard_dqn_acrobot.pth",
        help="Path to the saved model weights.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment so you can watch the agent.",
    )
    args = parser.parse_args()

    env = gym.make("Acrobot-v1", render_mode="human" if args.render else None)

    # Infer dimensions from the env instead of hardcoding them.
    eval_net = DQN(obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n)
    try:
        state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        print(f"Error: model '{args.model_path}' not found. Train one first with train.py.")
        env.close()
        sys.exit(1)
    eval_net.load_state_dict(state_dict)
    eval_net.eval()

    print(f"\n--- Evaluating {args.model_path} over {args.episodes} episodes ---")
    returns = []
    for episode in range(args.episodes):
        state, _info = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = epsilon_greedy_policy(env, eval_net, state, epsilon=0.0)
            state, reward, terminated, truncated, _info = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
        print(f"Episode {episode + 1}: return = {episode_return:.0f}")

    env.close()
    print(f"\nMean return: {np.mean(returns):.1f} ± {np.std(returns):.1f}")


if __name__ == "__main__":
    main()
