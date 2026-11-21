from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.fl_env import (SlipFrozenLake, greedy_policy, uniform_random_value,
                        value_iteration)
from src.metrics import auc, bootstrap_ci, cliffs_delta, cliffs_magnitude, steps_to_threshold
from src.stats import holm, mannwhitney
from src.train import run_training

SLIP_KEY = lambda s: round(float(s), 6)


def reference_values(cfg) -> dict:
    """Exact DP optimum and exact uniform-random success probability per slip."""
    refs = {}
    for slip in cfg["env"]["slips"]:
        env = SlipFrozenLake(slip=slip, desc=cfg["env"]["desc"],
                             max_steps=cfg["env"]["max_steps"])
        V, iters = value_iteration(env.P, env.n_states, env.n_actions, gamma=1.0)
        Vr = uniform_random_value(env.P, env.n_states, env.n_actions)
        refs[SLIP_KEY(slip)] = dict(V=V, optimal=float(V[0]), random=float(Vr[0]),
                                    iterations=int(iters), env=env)
    return refs


def run_tuning(cfg, out_dir: Path) -> dict:
    rows = []
    for slip in cfg["env"]["slips"]:
        for algo in ("sarsa", "q_learning"):
            for alpha in cfg["agent"]["alphas"]:
                for seed in cfg["tuning"]["seeds"]:
                    res = run_training(slip=slip, algo=algo, alpha=alpha,
                                       seed=seed, cfg=cfg, stage="tuning")
                    rows.append(dict(slip=SLIP_KEY(slip), algo=algo, alpha=alpha,
                                     seed=seed, auc=auc(res["eval_success"]),
                                     final=res["final_success"]))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "raw" / "tuning_runs.csv", index=False)

    best = (df.groupby(["slip", "algo", "alpha"])["auc"].mean().reset_index()
              .sort_values(["slip", "algo", "auc"], ascending=[True, True, False])
              .groupby(["slip", "algo"]).head(1))
    best.to_csv(out_dir / "raw" / "tuning_selected.csv", index=False)
    return {(r.slip, r.algo): float(r.alpha) for r in best.itertuples()}


def run_main(cfg, selected: dict, refs: dict, out_dir: Path):
    curves, summaries = [], []
    thr_frac = float(cfg["evaluation"]["threshold_frac"])

    for slip in cfg["env"]["slips"]:
        key = SLIP_KEY(slip)
        ref = refs[key]
        threshold = thr_frac * ref["optimal"]

        for algo in ("sarsa", "q_learning"):
            alpha = selected[(key, algo)]
            for seed in cfg["evaluation"]["seeds"]:
                res = run_training(slip=slip, algo=algo, alpha=alpha,
                                   seed=seed, cfg=cfg, stage="train")
                ev = res["episodes"]
                tail = ev.tail(max(1, int(0.2 * len(ev))))

                for st, sc in zip(res["eval_steps"], res["eval_success"]):
                    curves.append(dict(slip=key, algo=algo, seed=seed,
                                       eval_step=st, success=sc))
                summaries.append(dict(
                    slip=key, algo=algo, alpha=alpha, seed=seed,
                    final_success=res["final_success"],
                    auc=auc(res["eval_success"]),
                    steps_to_threshold=steps_to_threshold(
                        res["eval_steps"], res["eval_success"], threshold),
                    late_std=float(np.std(res["eval_success"][-5:])),
                    optimality_gap=ref["optimal"] - res["final_success"],
                    train_success=float(tail["success"].mean()),
                    mean_episode_length=float(tail["length"].mean()),
                ))
                np.save(out_dir / "raw" / f"Q_slip{key}_{algo}_seed{seed}.npy", res["Q"])
                res["episodes"].to_csv(
                    out_dir / "raw" / f"episodes_slip{key}_{algo}_seed{seed}.csv", index=False)

    curves_df, summary_df = pd.DataFrame(curves), pd.DataFrame(summaries)
    curves_df.to_csv(out_dir / "raw" / "curves_long.csv", index=False)
    summary_df.to_csv(out_dir / "raw" / "summary_per_seed.csv", index=False)

    with open(out_dir / "raw" / "reference_values.json", "w") as fh:
        json.dump({str(k): dict(optimal=v["optimal"], random=v["random"],
                                iterations=v["iterations"]) for k, v in refs.items()},
                  fh, indent=2)
    return curves_df, summary_df


def statistical_tests(summary_df: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for slip in cfg["env"]["slips"]:
        key = SLIP_KEY(slip)
        d = summary_df[np.isclose(summary_df["slip"], key)]
        for metric in ("final_success", "auc"):
            x = d[d["algo"] == "sarsa"][metric].values
            y = d[d["algo"] == "q_learning"][metric].values
            test = mannwhitney(x, y)
            delta = cliffs_delta(x, y)
            rows.append(dict(slip=key, metric=metric,
                             sarsa_mean=float(np.mean(x)), qlearn_mean=float(np.mean(y)),
                             U=test["U"], p=test["p"], welch_p=test["welch_p"],
                             cliffs_delta=delta, magnitude=cliffs_magnitude(delta)))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for metric in out["metric"].unique():
        mask = out["metric"] == metric
        out.loc[mask, "p_holm"] = holm(out.loc[mask, "p"].values)
    out.to_csv(Path(cfg["_out"]) / "raw" / "statistics.csv", index=False)
    return out
