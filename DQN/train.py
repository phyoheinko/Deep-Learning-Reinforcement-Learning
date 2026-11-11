"""Train DQN variants (standard / double) on a Gymnasium environment.

Reproducibility checklist implemented here:
  * every RNG is seeded (random, numpy, torch, env, action space)
  * hyperparameters live in a single TrainConfig dataclass
  * per-episode metrics are written to <model>_metrics.csv
  * a separate eval env periodically measures the greedy policy and the
    best checkpoint is saved (not just the final one)
  * training early-stops once the running mean return solves the env
"""

import argparse
import csv
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import optim
from util import (
    ReplayBuffer,
    double_dqn_update,
    dqn_update,
    epsilon_greedy_policy,
    evaluate_dqn,
    linear_epsilon_schedule,
    update_network,
)

from dqn import DQN

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """All hyperparameters and experiment settings in one place."""

    env_name: str = "Acrobot-v1"
    num_episodes: int = 30_000
    lr: float = 1e-3
    gamma: float = 0.99
    min_replay: int = 1_000
    batch_size: int = 64
    buffer_capacity: int = 50_000
    target_update_freq: int = 100
    max_grad_norm: float = 10.0
    eval_every: int = 500
    eval_episodes: int = 20
    # Acrobot-v1 is considered solved at a mean return of -100 over 100 episodes.
    solved_threshold: float = -100.0
    seed: int = 0


def set_seed(seed: int) -> None:
    """Seed every RNG used during training for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_dqn(
    update_func: Callable[..., float],
    model_name: str,
    config: TrainConfig | None = None,
) -> tuple[DQN, list[float], list[float]]:
    """Train a DQN agent and return (network, episode_returns, losses).

    Saves the best checkpoint (by periodic greedy evaluation) to `model_name`
    and per-episode metrics to a CSV next to it.
    """
    config = config or TrainConfig()
    set_seed(config.seed)

    env = gym.make(config.env_name)
    env.action_space.seed(config.seed)
    # Separate env for evaluation so eval episodes never touch training state.
    eval_env = gym.make(config.env_name)
    eval_env.action_space.seed(config.seed + 1)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    online_network = DQN(obs_dim=obs_dim, n_actions=n_actions)
    target_network = DQN(obs_dim=obs_dim, n_actions=n_actions)
    update_network(target_network, online_network)

    replay = ReplayBuffer(capacity=config.buffer_capacity)
    optimizer = optim.Adam(online_network.parameters(), lr=config.lr)

    logger.info(
        "Starting training with %s (seed=%d, saving best model to %s)",
        update_func.__name__, config.seed, model_name,
    )

    episode_returns: list[float] = []
    losses: list[float] = []
    metrics_rows: list[dict] = []
    best_eval_return = -np.inf
    global_step = 0
    epsilon = 1.0

    for episode in range(config.num_episodes):
        # Seed only the first reset; reseeding every episode would make every
        # episode start from the identical state. Afterwards the env's RNG
        # continues from its seeded state on its own.
        state, _info = env.reset(seed=config.seed if episode == 0 else None)

        done = False
        episode_return = 0.0
        episode_losses: list[float] = []

        while not done:
            # Epsilon decays only once learning starts, so the warmup steps
            # spent filling the replay buffer don't waste the decay budget.
            epsilon = linear_epsilon_schedule(max(0, global_step - config.min_replay))
            action = epsilon_greedy_policy(env, online_network, state, epsilon)
            next_state, reward, terminated, truncated, _info = env.step(action)

            replay.add(state, action, reward, next_state, terminated)

            if len(replay) >= max(config.min_replay, config.batch_size):
                loss = update_func(
                    target_network, online_network, optimizer, replay,
                    config.batch_size, config.gamma, config.max_grad_norm,
                )
                losses.append(loss)
                episode_losses.append(loss)

            state = next_state
            episode_return += reward
            done = terminated or truncated

            if global_step > 0 and global_step % config.target_update_freq == 0:
                update_network(target_network, online_network)
            global_step += 1

        episode_returns.append(episode_return)
        mean_loss = float(np.mean(episode_losses)) if episode_losses else float("nan")
        running_mean = float(np.mean(episode_returns[-100:]))

        eval_return: float | str = ""
        if (episode + 1) % config.eval_every == 0:
            eval_return = evaluate_dqn(eval_env, online_network, config.eval_episodes)
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                torch.save(online_network.state_dict(), model_name)
                logger.info(
                    "Episode %d | new best model saved (eval return %.1f)",
                    episode + 1, best_eval_return,
                )

        metrics_rows.append({
            "episode": episode + 1,
            "train_return": episode_return,
            "loss": mean_loss,
            "epsilon": epsilon,
            "eval_return": eval_return,
        })

        if (episode + 1) % 10 == 0:
            logger.info(
                "Episode %d | return %6.0f | running mean (100) %7.1f | loss %.4f | eps %.2f",
                episode + 1, episode_return, running_mean, mean_loss, epsilon,
            )

        if len(episode_returns) >= 100 and running_mean >= config.solved_threshold:
            logger.info(
                "Environment solved at episode %d (running mean %.1f) — stopping early.",
                episode + 1, running_mean,
            )
            break

    env.close()
    eval_env.close()

    metrics_path = Path(model_name).with_name(f"{Path(model_name).stem}_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "train_return", "loss", "epsilon", "eval_return"]
        )
        writer.writeheader()
        writer.writerows(metrics_rows)
    logger.info("Metrics saved to %s", metrics_path)

    final_eval = evaluate_dqn(eval_env, online_network, config.eval_episodes)
    logger.info(
        "Training done. Final eval return %.1f | best eval return during training %.1f",
        final_eval, best_eval_return,
    )

    # Guarantee a checkpoint exists even for runs shorter than eval_every.
    if not Path(model_name).exists():
        torch.save(online_network.state_dict(), model_name)
        logger.info("No eval checkpoint was produced; saved final weights to %s", model_name)

    return online_network, episode_returns, losses


# --- ALGORITHM REGISTRY ---
ALGO_CONFIGS = {
    "double": {
        "func": double_dqn_update,
        "default_model": "double_dqn_acrobot.pth",
    },
    "standard": {
        "func": dqn_update,
        "default_model": "standard_dqn_acrobot.pth",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN variants on Acrobot.")
    parser.add_argument(
        "--algo",
        type=str,
        default="double",
        choices=list(ALGO_CONFIGS.keys()),
        help="Choose the training algorithm variant.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Custom filename to save model weights.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the number of training episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed (run several seeds for sound comparisons).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = TrainConfig()
    if args.episodes is not None:
        config.num_episodes = args.episodes
    if args.seed is not None:
        config.seed = args.seed

    algo = ALGO_CONFIGS[args.algo]
    model_name = args.model_name or algo["default_model"]

    train_dqn(update_func=algo["func"], model_name=model_name, config=config)


if __name__ == "__main__":
    main()