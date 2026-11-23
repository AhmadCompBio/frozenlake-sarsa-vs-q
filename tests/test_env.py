import numpy as np
import pytest

from src.fl_env import SlipFrozenLake, uniform_random_value, value_iteration


def _pairs(entries):
    return sorted([(float(p), int(s), float(r), bool(t)) for p, s, r, t in entries],
                  key=lambda e: (e[1], e[2], e[3]))


@pytest.mark.parametrize("slip,is_slippery", [(2 / 3, True), (0.0, False)])
def test_matches_gymnasium(slip, is_slippery):
    """Bit-identical transitions to Gymnasium's FrozenLake-v1 at both endpoints."""
    gym = pytest.importorskip("gymnasium")
    ours = SlipFrozenLake(slip=slip)
    ref = gym.make("FrozenLake-v1", map_name="4x4",
                   is_slippery=is_slippery).unwrapped
    for s in range(16):
        for a in range(4):
            o, r = _pairs(ours.P[s][a]), _pairs(ref.P[s][a])
            assert len(o) == len(r)
            for (p1, s1, r1, t1), (p2, s2, r2, t2) in zip(o, r):
                assert (s1, r1, t1) == (s2, r2, t2)
                assert abs(p1 - p2) < 1e-12


def test_rows_are_distributions():
    """Every (s, a) row must sum to 1 — catches off-by-one in slip arithmetic."""
    for slip in (0.0, 0.1, 0.2, 2 / 3):
        env = SlipFrozenLake(slip=slip)
        for s in range(16):
            for a in range(4):
                assert abs(sum(o[0] for o in env.P[s][a]) - 1.0) < 1e-12


def test_terminal_states_are_absorbing():
    """Goal and hole states must self-loop with termination and zero reward."""
    env = SlipFrozenLake(slip=2 / 3)
    for s in env.terminal_states:
        for a in range(4):
            p, s2, r, term = env.P[s][a][0]
            assert (p, s2, r, term) == (1.0, s, 0.0, True)


def test_dp_reference_values():
    """Value iteration must give V*(s0) = 1.0 (deterministic) and 14/17 ≈ 0.8235 (slippery)."""
    det = SlipFrozenLake(slip=0.0)
    V, _ = value_iteration(det.P, 16, 4, gamma=1.0)
    assert V[0] == pytest.approx(1.0, abs=1e-9)
    slip = SlipFrozenLake(slip=2 / 3)
    V, _ = value_iteration(slip.P, 16, 4, gamma=1.0)
    # Gymnasium FrozenLake-v1 slippery: V*(s0) = 14/17 ≈ 0.8235
    # (the commonly-cited 0.74 is the *trained-agent* success rate, not the DP optimum)
    assert V[0] == pytest.approx(14 / 17, abs=1e-6)
    Vr = uniform_random_value(slip.P, 16, 4)
    assert 0.0 <= Vr[0] < V[0]


def test_truncation_is_not_termination():
    """Hitting max_steps sets truncated=True, not terminated — bootstrap must not zero."""
    env = SlipFrozenLake(slip=0.0, max_steps=3)
    env.reset(seed=0)
    env.step(3)                                   # UP into wall -> stay at (0,0)
    env.step(0)                                   # LEFT into wall -> stay at (0,0)
    _, _, term, trunc, _ = env.step(0)            # LEFT again -> t=3 >= max_steps
    assert (not term) and trunc
