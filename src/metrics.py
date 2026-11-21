from __future__ import annotations

import numpy as np


def auc(curve) -> float:
    """Equally spaced evaluations -> mean success == normalised area under curve."""
    curve = np.asarray(curve, dtype=float)
    return float(np.mean(curve)) if curve.size else float("nan")


def steps_to_threshold(steps, curve, threshold: float) -> float:
    """First evaluation step at which success >= threshold; NaN if never reached."""
    steps = np.asarray(steps, dtype=float)
    curve = np.asarray(curve, dtype=float)
    idx = np.flatnonzero(curve >= threshold)
    return float(steps[idx[0]]) if idx.size else float("nan")


def bootstrap_ci(x, n_boot: int = 10_000, ci: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI of the mean (Efron & Tibshirani, 1993)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def cliffs_delta(x, y) -> float:
    """P(X>Y) - P(X<Y) in [-1, 1]; rank-based, distribution-free."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    gt = float(np.sum(x[:, None] > y[None, :]))
    lt = float(np.sum(x[:, None] < y[None, :]))
    return (gt - lt) / (x.size * y.size)


def cliffs_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"
