from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.rl.actor import SquashedGaussianActor
from src.rl.critic import QFunction


@dataclass
class SACUpdateStats:
    actor_loss: float
    critic_loss: float
    alpha: float


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, cfg: dict, device: torch.device):
        self.device = device
        hidden_sizes = cfg["hidden_sizes"]

        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes, act_limit).to(device)
        self.q1 = QFunction(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2 = QFunction(obs_dim, act_dim, hidden_sizes).to(device)
        self.q1_target = QFunction(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2_target = QFunction(obs_dim, act_dim, hidden_sizes).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg["critic_lr"])

        self.auto_alpha = bool(cfg.get("auto_alpha", True))
        init_alpha = float(cfg.get("alpha", 0.2))
        if self.auto_alpha:
            self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["alpha_lr"])
            self.target_entropy = -float(act_dim)
        else:
            self.log_alpha = torch.tensor(np.log(init_alpha), device=device)
            self.alpha_opt = None
            self.target_entropy = None

        self.tau = float(cfg["tau"])

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        return action.cpu().numpy()[0]

    def update(self, batch: dict[str, np.ndarray]) -> SACUpdateStats:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor(batch["discounts"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_logp = self.actor(next_obs)
            q1_next = self.q1_target(next_obs, next_actions)
            q2_next = self.q2_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_logp
            target_q = rewards + (1.0 - dones) * discounts * q_next

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        critic_loss = (q1 - target_q).pow(2).mean() + (q2 - target_q).pow(2).mean()

        self.q_opt.zero_grad()
        critic_loss.backward()
        self.q_opt.step()

        for p in list(self.q1.parameters()) + list(self.q2.parameters()):
            p.requires_grad = False

        pi, logp_pi = self.actor(obs)
        q_pi = torch.min(self.q1(obs, pi), self.q2(obs, pi))
        actor_loss = (self.alpha.detach() * logp_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        for p in list(self.q1.parameters()) + list(self.q2.parameters()):
            p.requires_grad = True

        self._update_targets()
        return SACUpdateStats(float(actor_loss.item()), float(critic_loss.item()), float(self.alpha.item()))

    def _update_targets(self):
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)
            for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)
