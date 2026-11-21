from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind


def mannwhitney(x, y) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    t, p_t = ttest_ind(x, y, equal_var=False)     # Welch, reported as secondary
    return dict(U=float(u), p=float(p), welch_t=float(t), welch_p=float(p_t),
                n_x=int(x.size), n_y=int(y.size))


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment; returns adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj
