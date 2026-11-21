import numpy as np
import pytest

from src.metrics import (auc, bootstrap_ci, cliffs_delta, cliffs_magnitude,
                         steps_to_threshold)
from src.stats import holm


def test_auc_is_mean_of_curve():
    assert auc([0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert auc([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert np.isnan(auc([]))


def test_steps_to_threshold_first_crossing():
    steps = [0, 100, 200, 300, 400]
    curve = [0.1, 0.2, 0.9, 0.95, 1.0]
    assert steps_to_threshold(steps, curve, 0.9) == 200
    # Threshold above the curve maximum -> never reached -> NaN
    assert np.isnan(steps_to_threshold(steps, curve, 1.01))


def test_bootstrap_ci_brackets_constant_mean():
    x = np.full(10, 0.42)
    lo, hi = bootstrap_ci(x, n_boot=2000, seed=0)
    assert lo <= 0.42 <= hi
    assert np.isclose(lo, 0.42) and np.isclose(hi, 0.42)


def test_cliffs_delta_extremes():
    # x = [3,4,5], y = [1,2,3]: 8 strict wins, 1 tie, 0 losses -> (8-0)/9
    assert cliffs_delta([3, 4, 5], [1, 2, 3]) == pytest.approx(1.0 - 1 / 9)
    assert cliffs_delta([1, 1, 1], [1, 1, 1]) == pytest.approx(0.0)
    assert cliffs_magnitude(0.0) == "negligible"
    assert cliffs_magnitude(0.6) == "large"


def test_holm_matches_hand_computation():
    p = np.array([0.001, 0.02, 0.04])
    adj = holm(p)
    # Holm step-down: 0.001*3 = 0.003; 0.02*2 = 0.04; 0.04*1 = 0.04
    assert adj[0] == pytest.approx(0.003)
    assert adj[1] == pytest.approx(0.04)
    assert adj[2] == pytest.approx(0.04)
