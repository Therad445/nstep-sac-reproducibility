import gymnasium as gym
import numpy as np


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def action_dim(env) -> int:
    return int(np.prod(env.action_space.shape))


def obs_dim(env) -> int:
    return int(np.prod(env.observation_space.shape))
