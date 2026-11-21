"""Tabular SARSA and Q-learning with an identical epsilon-greedy behaviour policy.

The *only* difference between the two agents is the update target, which makes
the comparison a controlled ablation of on-policy vs off-policy bootstrapping.
"""
from __future__ import annotations

import numpy as np


class TabularAgent:
    name = "tabular"

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=10_000, seed=0):
        self.n_states, self.n_actions = n_states, n_actions
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.eps_start, self.eps_end = float(eps_start), float(eps_end)
        self.eps_decay_steps = max(1, int(eps_decay_steps))
        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.rng = np.random.default_rng(seed)

    def epsilon(self, step: int) -> float:
        frac = min(1.0, step / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def greedy(self, state: int) -> int:
        q = self.Q[state]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def act(self, state: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(state)

    def update(self, s, a, r, s2, terminated, a2=None):
        raise NotImplementedError


class QLearning(TabularAgent):
    name = "q_learning"

    def update(self, s, a, r, s2, terminated, a2=None):
        target = r if terminated else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


class SARSA(TabularAgent):
    name = "sarsa"

    def update(self, s, a, r, s2, terminated, a2=None):
        if terminated:
            target = r
        else:
            if a2 is None:                      # defensive; normally supplied
                a2 = self.act(s2, self.epsilon(0))
            target = r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


AGENTS = {"sarsa": SARSA, "q_learning": QLearning}
