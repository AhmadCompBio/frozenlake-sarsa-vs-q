"""Regenerate ONE figure from committed raw results. Never retrains.

Usage:
    python -m scripts.make_figures --only learning_curves
    python -m scripts.make_figures --only final_distribution
    python -m scripts.make_figures --only sensitivity
    python -m scripts.make_figures --only policy_value

Requires:
    raw/curves_long.csv, raw/summary_per_seed.csv
    raw/Q_slip*_*.npy (only for policy_value)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import plotting
from src.fl_env import (SlipFrozenLake, greedy_policy, uniform_random_value,
                        value_iteration)

ONLY = ("learning_curves", "final_distribution", "sensitivity", "policy_value")


def _build_refs(cfg) -> dict:
    refs = {}
    for slip in cfg["env"]["slips"]:
        key = round(float(slip), 6)
        env = SlipFrozenLake(slip=slip, desc=cfg["env"]["desc"],
                             max_steps=cfg["env"]["max_steps"])
        V, _ = value_iteration(env.P, env.n_states, env.n_actions, gamma=1.0)
        Vr = uniform_random_value(env.P, env.n_states, env.n_actions)
        refs[key] = dict(optimal=float(V[0]), random=float(Vr[0]),
                         V=V, env=env)
    return refs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate one figure from raw results.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", required=True, choices=ONLY)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    slips = [round(float(s), 6) for s in cfg["env"]["slips"]]
    refs = _build_refs(cfg)

    if args.only in ("learning_curves", "sensitivity"):
        curves = pd.read_csv(out / "raw" / "curves_long.csv")
        summary = pd.read_csv(out / "raw" / "summary_per_seed.csv")
        if args.only == "learning_curves":
            plotting.plot_learning_curves(
                curves, refs, slips, out / "figures" / "fig_learning_curves.pdf")
        else:
            plotting.plot_sensitivity(
                summary, refs, slips, out / "figures" / "fig_sensitivity.pdf")

    elif args.only == "final_distribution":
        summary = pd.read_csv(out / "raw" / "summary_per_seed.csv")
        plotting.plot_final_distribution(
            summary, slips, out / "figures" / "fig_final_distribution.pdf")

    elif args.only == "policy_value":
        summary = pd.read_csv(out / "raw" / "summary_per_seed.csv")
        slip = slips[-1]
        d = summary[np.isclose(summary["slip"], slip)]
        ref = refs[slip]
        pi_dp = greedy_policy(ref["env"].P, ref["V"],
                              ref["env"].n_states, ref["env"].n_actions)
        V_maps, pi_maps = [ref["V"]], [pi_dp]
        titles = ["DP optimum"]
        for algo, name in (("sarsa", "SARSA"), ("q_learning", "Q-learning")):
            s = d[d["algo"] == algo].sort_values("final_success").reset_index(drop=True)
            median_seed = int(s.iloc[len(s) // 2]["seed"])
            Q = np.load(out / "raw" / f"Q_slip{slip}_{algo}_seed{median_seed}.npy")
            V_maps.append(Q.max(axis=1))
            pi_maps.append(Q.argmax(axis=1))
            titles.append(f"{name} (median seed {median_seed})")
        plotting.plot_policy_value(V_maps, pi_maps, titles,
                                   out / "figures" / "fig_policy_value.pdf")

    print(f"figure written: {args.only} -> {out/'figures'}")
    return 0


if __name__ == "__main__":
    main()
