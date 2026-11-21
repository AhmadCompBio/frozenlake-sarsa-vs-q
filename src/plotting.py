from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import bootstrap_ci

COLORS = {"sarsa": "#1b6ca8", "q_learning": "#d1495b"}
LABELS = {"sarsa": "SARSA", "q_learning": "Q-learning"}
ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}
SLIP_KEY = lambda s: round(float(s), 6)


def slip_label(slip: float) -> str:
    return "2/3" if abs(slip - 2 / 3) < 1e-6 else f"{slip:.2f}"


def _group_ci(sub: pd.DataFrame, algo: str):
    g = sub[sub["algo"] == algo].groupby("eval_step")["success"]
    x = g.mean().index.values.astype(float)
    mean = g.mean().values
    ci = np.array([bootstrap_ci(v.values, seed=0) for _, v in g])
    return x, mean, ci[:, 0], ci[:, 1]


def plot_learning_curves(curves, refs, slips, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True, sharey=True)
    for ax, slip in zip(axes.ravel(), slips):
        sub = curves[np.isclose(curves["slip"], slip)]
        for algo in ("sarsa", "q_learning"):
            x, m, lo, hi = _group_ci(sub, algo)
            ax.fill_between(x, lo, hi, color=COLORS[algo], alpha=0.25, linewidth=0)
            ax.plot(x, m, color=COLORS[algo], linewidth=1.2, label=LABELS[algo])
        ref = refs[SLIP_KEY(slip)]
        ax.axhline(ref["optimal"], color="k", ls="--", lw=0.8, label="DP optimum")
        ax.axhline(ref["random"], color="grey", ls=":", lw=0.8, label="uniform random")
        ax.set_title(f"slip = {slip_label(slip)}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("environment steps", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("evaluation success rate", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_final_distribution(summary, slips, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    positions, labels = [], []
    for i, slip in enumerate(slips):
        d = summary[np.isclose(summary["slip"], slip)]
        for j, algo in enumerate(("sarsa", "q_learning")):
            vals = d[d["algo"] == algo]["final_success"].values
            pos = i * 3 + j
            positions.append(pos)
            ax.scatter(np.full(vals.size, pos) + np.linspace(-0.12, 0.12, vals.size),
                       vals, s=12, color=COLORS[algo], alpha=0.8, zorder=3)
            ax.hlines(np.mean(vals), pos - 0.25, pos + 0.25, color="k", lw=1.2, zorder=4)
        labels.append(f"slip = {slip_label(slip)}")
    ax.set_xticks([i * 3 + 0.5 for i in range(len(slips))])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("final success rate", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[a], label=LABELS[a])
               for a in ("sarsa", "q_learning")]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(summary, refs, slips, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, metric, ylabel in ((axes[0], "final_success", "final success rate"),
                               (axes[1], "auc", "area under learning curve")):
        for algo in ("sarsa", "q_learning"):
            means, los, his = [], [], []
            for slip in slips:
                vals = summary[(np.isclose(summary["slip"], slip)) &
                               (summary["algo"] == algo)][metric].values
                lo, hi = bootstrap_ci(vals, seed=0)
                means.append(np.mean(vals)); los.append(lo); his.append(hi)
            x = np.arange(len(slips))
            ax.errorbar(x, means, yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                        color=COLORS[algo], marker="o", ms=3, lw=1.1,
                        capsize=2, label=LABELS[algo])
        ax.set_xticks(np.arange(len(slips)))
        ax.set_xticklabels([slip_label(s) for s in slips], fontsize=7)
        ax.set_xlabel("slip probability", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(lw=0.3, alpha=0.5)
    if axes[0] is not None:
        axes[0].plot(np.arange(len(slips)), [refs[SLIP_KEY(s)]["optimal"] for s in slips],
                     "k--", lw=0.8, label="DP optimum")
        axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_policy_value(value_maps, policy_maps, titles, out_path, nrow=4, ncol=4):
    n = len(value_maps)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.5))
    vmax = max(1e-9, max(float(np.max(v)) for v in value_maps))
    im = None
    for ax, V, pi, title in zip(np.atleast_1d(axes), value_maps, policy_maps, titles):
        im = ax.imshow(V.reshape(nrow, ncol), cmap="viridis", vmin=0, vmax=vmax)
        for s in range(nrow * ncol):
            r, c = divmod(s, ncol)
            ax.text(c, r, f"{V[s]:.2f}\n{ARROWS[int(pi[s])]}", ha="center", va="center",
                    fontsize=5.5, color="w")
        ax.set_title(title, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), shrink=0.85, label="V(s)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
