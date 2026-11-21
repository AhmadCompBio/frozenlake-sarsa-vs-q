"""Parameterised stochastic FrozenLake (Gymnasium-compatible) + exact DP solvers.

P(intended) = 1 - slip,  P(each perpendicular) = slip / 2.

    slip = 0.0  -> deterministic FrozenLake
    slip = 2/3  -> Gymnasium "FrozenLake-v1" with is_slippery=True
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from gymnasium import spaces

MAP_4x4: Tuple[str, ...] = ("SFFF", "FHFH", "FFFH", "HFFG")
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
_DELTA = {LEFT: (0, -1), DOWN: (1, 0), RIGHT: (0, 1), UP: (-1, 0)}


class SlipFrozenLake:
    """4x4 FrozenLake with a single scalar slip probability."""

    metadata = {"render_modes": []}

    def __init__(self, slip: float = 2.0 / 3.0,
                 desc: Sequence[str] = MAP_4x4,
                 max_steps: int = 200):
        if not 0.0 <= slip <= 1.0:
            raise ValueError("slip must lie in [0, 1]")
        self.slip = float(slip)
        self.desc = tuple(desc)
        self.nrow, self.ncol = len(self.desc), len(self.desc[0])
        self.n_states = self.nrow * self.ncol
        self.n_actions = 4
        self.max_steps = int(max_steps)
        self.start_state = 0
        self.goal_states = {s for s in range(self.n_states) if self.letter(s) == "G"}
        self.hole_states = {s for s in range(self.n_states) if self.letter(s) == "H"}
        self.terminal_states = self.goal_states | self.hole_states
        self.P = self._build_transitions()
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)
        self._rng = np.random.default_rng()
        self._s = self.start_state
        self._t = 0

    # ---------- helpers ----------
    def letter(self, s: int) -> str:
        return self.desc[s // self.ncol][s % self.ncol]

    def _build_transitions(self) -> List[List[List[Tuple[float, int, float, bool]]]]:
        P = [[[] for _ in range(self.n_actions)] for _ in range(self.n_states)]
        for s in range(self.n_states):
            if self.letter(s) in "GH":
                for a in range(self.n_actions):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue
            row, col = divmod(s, self.ncol)
            for a in range(self.n_actions):
                outcomes = {
                    a: 1.0 - self.slip,
                    (a - 1) % 4: self.slip / 2.0,
                    (a + 1) % 4: self.slip / 2.0,
                }
                for direction, prob in outcomes.items():
                    if prob <= 0.0:
                        continue
                    dr, dc = _DELTA[direction]
                    nr = min(max(row + dr, 0), self.nrow - 1)
                    nc = min(max(col + dc, 0), self.ncol - 1)
                    s2 = nr * self.ncol + nc
                    reward = 1.0 if s2 in self.goal_states else 0.0
                    P[s][a].append((prob, s2, reward, s2 in self.terminal_states))
        return P

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._s, self._t = self.start_state, 0
        return self._s, {}

    def step(self, action):
        if not 0 <= int(action) < self.n_actions:
            raise ValueError(f"invalid action {action}")
        outcomes = self.P[self._s][int(action)]
        probs = np.array([o[0] for o in outcomes], dtype=float)
        idx = int(self._rng.choice(len(outcomes), p=probs / probs.sum()))
        _, s2, reward, terminated = outcomes[idx]
        self._s = s2
        self._t += 1
        truncated = (not terminated) and self._t >= self.max_steps
        return s2, float(reward), bool(terminated), bool(truncated), {}


# ------------------------- exact reference values -------------------------

def value_iteration(P, n_states, n_actions, gamma=1.0, tol=1e-13, max_iter=200_000):
    """V*(s) = max_a sum_s' p (r + gamma V*(s')); terminal states stay at 0.

    With gamma = 1 this is the maximum probability of reaching the goal.
    """
    V = np.zeros(n_states)
    for it in range(max_iter):
        delta = 0.0
        for s in range(n_states):
            q = [
                sum(p * (r + (0.0 if term else gamma * V[s2])) for p, s2, r, term in P[s][a])
                for a in range(n_actions)
            ]
            best = max(q)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < tol:
            return V, it + 1
    return V, max_iter


def greedy_policy(P, V, n_states, n_actions, gamma=1.0):
    pi = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q = [
            sum(p * (r + (0.0 if term else gamma * V[s2])) for p, s2, r, term in P[s][a])
            for a in range(n_actions)
        ]
        pi[s] = int(np.argmax(q))
    return pi


def uniform_random_value(P, n_states, n_actions):
    """Exact success probability of the uniform random policy (linear solve)."""
    A = np.eye(n_states)
    b = np.zeros(n_states)
    for s in range(n_states):
        for a in range(n_actions):
            for p, s2, r, term in P[s][a]:
                w = p / n_actions
                b[s] += w * r
                if not term:
                    A[s, s2] -= w
    return np.linalg.solve(A, b)
