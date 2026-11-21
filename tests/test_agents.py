import numpy as np

from src.agents import QLearning, SARSA


def test_q_learning_update_formula():
    """Q(s,a) <- (1-a)Q(s,a) + a[r + gamma * max_a' Q(s',a')]."""
    ag = QLearning(2, 2, alpha=0.5, gamma=0.9)
    ag.Q[:] = 0.0
    ag.Q[1] = [1.0, 2.0]
    ag.update(0, 0, 1.0, 1, terminated=False)
    assert np.isclose(ag.Q[0, 0], 0.5 * (1.0 + 0.9 * 2.0))     # 1.4


def test_sarsa_update_formula():
    """Q(s,a) <- (1-a)Q(s,a) + a[r + gamma * Q(s', a')] using the actual next action."""
    ag = SARSA(2, 2, alpha=0.5, gamma=0.9)
    ag.Q[:] = 0.0
    ag.Q[1] = [1.0, 2.0]
    ag.update(0, 0, 1.0, 1, terminated=False, a2=0)
    assert np.isclose(ag.Q[0, 0], 0.5 * (1.0 + 0.9 * 1.0))     # 0.95


def test_terminal_update_has_no_bootstrap():
    """Terminal update must be Q(s,a) <- (1-a)Q(s,a) + a*r with no V(s') term."""
    for cls in (QLearning, SARSA):
        ag = cls(2, 2, alpha=0.5, gamma=0.9)
        ag.Q[:] = 5.0
        ag.update(0, 0, 1.0, 1, terminated=True, a2=1)
        assert np.isclose(ag.Q[0, 0], 0.5 * 1.0 + 0.5 * 5.0)


def test_q_learning_solves_deterministic_frozenlake():
    """End-to-end: with a sufficient budget, Q-learning must reach 100% greedy success."""
    from src.fl_env import SlipFrozenLake
    from src.train import evaluate_greedy
    env = SlipFrozenLake(slip=0.0)
    ag = QLearning(16, 4, alpha=0.2, gamma=0.99,
                   eps_start=1.0, eps_end=0.05, eps_decay_steps=10_000, seed=0)
    s, _ = env.reset(seed=0)
    for step in range(30_000):
        a = ag.act(s, ag.epsilon(step))
        s2, r, term, trunc, _ = env.step(a)
        a2 = ag.act(s2, ag.epsilon(step)) if not term else None
        ag.update(s, a, r, s2, term, a2)
        if term or trunc:
            s, _ = env.reset()
        else:
            s = s2
    assert evaluate_greedy(ag, SlipFrozenLake(slip=0.0), 50, seed=999) == 1.0
