from __future__ import annotations

from collections import deque


class NStepBuffer:
    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def append(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
        if len(self.buffer) < self.n_step and not done:
            return None
        return self._build_transition()

    def flush(self):
        items = []
        while self.buffer:
            items.append(self._build_transition())
            self.buffer.popleft()
        return items

    def _build_transition(self):
        reward_sum = 0.0
        discount = 1.0
        next_obs = self.buffer[-1][3]
        done = self.buffer[-1][4]

        for i, (_, _, reward, candidate_next_obs, candidate_done) in enumerate(self.buffer):
            reward_sum += (self.gamma ** i) * float(reward)
            next_obs = candidate_next_obs
            if candidate_done:
                done = True
                discount = self.gamma ** (i + 1)
                break
        else:
            discount = self.gamma ** len(self.buffer)

        obs, action, _, _, _ = self.buffer[0]
        return obs, action, reward_sum, next_obs, float(done), discount

    def pop_left(self):
        if self.buffer:
            self.buffer.popleft()
