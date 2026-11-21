from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import bootstrap_ci
from src.plotting import slip_label


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def mdp_table(cfg, refs) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Environment and MDP definition. Values are exact (value iteration, $\gamma=1$).}",
        r"\label{tab:mdp}",
        r"\begin{tabular}{ll}", r"\toprule",
        r"Property & Value \\", r"\midrule",
        r"Map & 4$\times$4 (SFFF/FHFH/FFFH/HFFG) \\",
        r"States / actions & 16 / 4 (left, down, right, up) \\",
        r"Transitions & $(1-p)$ intended, $p/2$ each perpendicular \\",
        r"Reward & $+1$ on entering the goal, $0$ otherwise \\",
        r"Discount $\gamma$ & 0.99 (agents), 1.0 (DP reference) \\",
        r"Termination & goal or hole; truncation at 200 steps \\",
        r"Observation & discrete state index \\",
        r"\midrule",
        r"Slip $p$ & $V^\star(s_0)$ / uniform-random $V^\pi(s_0)$ \\",
    ]
    for slip in cfg["env"]["slips"]:
        r = refs[round(float(slip), 6)]
        lines.append(rf"{slip_label(slip)} & {r['optimal']:.3f} / {r['random']:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def hyperparameter_table(cfg, selected) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Selected hyperparameters (chosen by mean AUC on tuning seeds "
        r"\texttt{" + str(cfg["tuning"]["seeds"]) + r"}, reported on held-out seeds).}",
        r"\label{tab:hparams}",
        r"\begin{tabular}{llcc}", r"\toprule",
        r"Slip & Method & $\alpha$ & $\gamma$ / $\varepsilon$ / budget \\", r"\midrule",
    ]
    for slip in cfg["env"]["slips"]:
        key = round(float(slip), 6)
        for algo, name in (("sarsa", "SARSA"), ("q_learning", "Q-learning")):
            lines.append(
                rf"{slip_label(slip)} & {name} & {selected[(key, algo)]:.2f} & "
                rf"{cfg['agent']['gamma']} / {cfg['agent']['eps_start']}$\to$"
                rf"{cfg['agent']['eps_end']} / {cfg['train']['budget']} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main_results_table(summary, slips, cfg) -> str:
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Main results over " + str(len(cfg["evaluation"]["seeds"])) +
        r" seeds (mean [95\% bootstrap CI]). Steps-to-threshold is censored at the "
        r"budget; $n$ counts seeds that reached 90\% of $V^\star$.}",
        r"\label{tab:main}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Slip & Method & Final success & AUC & Steps to thr. & Late sd & Gap \\",
        r"\midrule",
    ]
    for slip in slips:
        d = summary[np.isclose(summary["slip"], slip)]
        for algo, name in (("sarsa", "SARSA"), ("q_learning", "Q-learning")):
            s = d[d["algo"] == algo]
            m = s["final_success"].mean()
            lo, hi = bootstrap_ci(s["final_success"].values, seed=0)
            a_m = s["auc"].mean(); a_lo, a_hi = bootstrap_ci(s["auc"].values, seed=0)
            st = s["steps_to_threshold"].dropna()
            st_txt = (rf"{st.median():.0f} ({st.size}/{len(s)})" if st.size else
                      rf"$-$ ({0}/{len(s)})")
            lines.append(
                rf"{slip_label(slip)} & {name} & {m:.3f} [{lo:.3f}, {hi:.3f}] & "
                rf"{a_m:.3f} [{a_lo:.3f}, {a_hi:.3f}] & {st_txt} & "
                rf"{s['late_std'].mean():.3f} & {s['optimality_gap'].mean():+.3f} \\")
        lines.append(r"\midrule" if slip != slips[-1] else "")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def stats_table(stats) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{SARSA vs Q-learning: two-sided Mann--Whitney $U$ with Holm "
        r"correction within each metric, and Cliff's $\delta$ ($n=10$ per group).}",
        r"\label{tab:stats}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Slip & Metric & $U$ & $p$ & $p_{\text{Holm}}$ & Cliff's $\delta$ \\", r"\midrule",
    ]
    for r in stats.itertuples():
        lines.append(rf"{slip_label(r.slip)} & {r.metric} & {r.U:.1f} & {r.p:.3f} & "
                     rf"{r.p_holm:.3f} & {r.cliffs_delta:+.2f} ({r.magnitude}) \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
