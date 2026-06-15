from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from tqdm import trange

from src.rl.envs import action_dim, make_env, obs_dim
from src.rl.nstep_buffer import NStepBuffer
from src.rl.replay_buffer import ReplayBuffer
from src.rl.sac import SACAgent
from src.rl.utils import ensure_dir, get_device, load_config, set_seed


def evaluate(agent: SACAgent, env_id: str, seed: int, episodes: int) -> float:
    env = make_env(env_id, seed + 10_000)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device()

    env = make_env(cfg["env_id"], int(cfg["seed"]))
    o_dim = obs_dim(env)
    a_dim = action_dim(env)
    act_limit = float(env.action_space.high[0])

    agent = SACAgent(o_dim, a_dim, act_limit, cfg, device)
    replay = ReplayBuffer(o_dim, a_dim, int(cfg["replay_size"]))
    nstep = NStepBuffer(int(cfg["n_step"]), float(cfg["gamma"]))

    log_dir = ensure_dir(cfg["log_dir"])
    log_path = Path(log_dir) / f"{cfg['run_name']}.csv"

    obs, _ = env.reset()
    last_stats = None
    start_time = time.time()

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "eval_return", "actor_loss", "critic_loss", "alpha", "elapsed_sec"],
        )
        writer.writeheader()

        for step in trange(1, int(cfg["total_steps"]) + 1):
            if step <= int(cfg["start_steps"]):
                action = env.action_space.sample()
            else:
                action = agent.act(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            built = nstep.append(obs, action, reward, next_obs, done)
            if built is not None:
                replay.store(*built)
                nstep.pop_left()

            obs = next_obs

            if done:
                for item in nstep.flush():
                    replay.store(*item)
                obs, _ = env.reset()

            if replay.size >= int(cfg["update_after"]) and step % int(cfg["update_every"]) == 0:
                batch = replay.sample_batch(int(cfg["batch_size"]))
                last_stats = agent.update(batch)

            if step % int(cfg["eval_every"]) == 0:
                eval_return = evaluate(agent, cfg["env_id"], int(cfg["seed"]), int(cfg["eval_episodes"]))
                writer.writerow(
                    {
                        "step": step,
                        "eval_return": eval_return,
                        "actor_loss": None if last_stats is None else last_stats.actor_loss,
                        "critic_loss": None if last_stats is None else last_stats.critic_loss,
                        "alpha": None if last_stats is None else last_stats.alpha,
                        "elapsed_sec": round(time.time() - start_time, 3),
                    }
                )
                f.flush()

    env.close()
    print(f"Saved log to {log_path}")


if __name__ == "__main__":
    main()
