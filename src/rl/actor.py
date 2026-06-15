from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes, act_limit: float):
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes], nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic: bool = False, with_logprob: bool = True):
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        pi_action = mu if deterministic else dist.rsample()

        logp_pi = None
        if with_logprob:
            logp_pi = dist.log_prob(pi_action).sum(axis=-1)
            correction = 2 * (
                torch.log(torch.tensor(2.0, device=obs.device))
                - pi_action
                - torch.nn.functional.softplus(-2 * pi_action)
            )
            logp_pi -= correction.sum(axis=-1)

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi
