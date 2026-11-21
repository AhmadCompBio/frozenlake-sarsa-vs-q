"""End-to-end driver: tune -> main -> stats -> figures -> tables.

Usage:
    python -m scripts.run_all                 # full study (~25-35 min on CPU)
    python -m scripts.run_all --smoke         # ~1-2 min, writes results/smoke/
    python -m scripts.run_all --out DIR       # override output directory

Every artefact is written under <out>/{raw,figures,tables}/. Nothing is
hard-coded except the default config path; change behaviour via the YAML.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from src import experiment, plotting, tables
from src.fl_env import greedy_policy


def _prepare_dirs(out: Path) -> None:
    for sub in ("raw", "figures", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)


def _apply_smoke(cfg: dict) -> None:
    """Shrink the study so the whole pipeline runs end-to-end in ~1 minute."""
    cfg["env"]["slips"] = [0.0, 2.0 / 3.0]
    cfg["agent"]["alphas"] = [0.05, 0.1]
    cfg["train"].update(budget=3_000, eval_every=500, eval_episodes=10)
    cfg["tuning"].update(seeds=[0], budget=1_500, eval_every=500, eval_episodes=10)
    cfg["evaluation"].update(seeds=[100, 101], final_eval_episodes=20)


def _policy_value_figure(cfg, refs, summary, out: Path) -> None:
    """Behaviour visualisation: DP optimum vs SARSA vs Q-learning at the hardest slip."""
    slip = round(float(cfg["env"]["slips"][-1]), 6)
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


def _make_figures(cfg, refs, curves, summary, out: Path) -> None:
    slips = [round(float(s), 6) for s in cfg["env"]["slips"]]
    plotting.plot_learning_curves(
        curves, refs, slips, out / "figures" / "fig_learning_curves.pdf")
    plotting.plot_final_distribution(
        summary, slips, out / "figures" / "fig_final_distribution.pdf")
    plotting.plot_sensitivity(
        summary, refs, slips, out / "figures" / "fig_sensitivity.pdf")
    _policy_value_figure(cfg, refs, summary, out)


def _make_tables(cfg, refs, selected, summary, stats, out: Path) -> None:
    slips = [round(float(s), 6) for s in cfg["env"]["slips"]]
    tables.write(out / "tables" / "table_mdp.tex",
                 tables.mdp_table(cfg, refs))
    tables.write(out / "tables" / "table_hyperparams.tex",
                 tables.hyperparameter_table(cfg, selected))
    tables.write(out / "tables" / "table_main.tex",
                 tables.main_results_table(summary, slips, cfg))
    tables.write(out / "tables" / "table_stats.tex",
                 tables.stats_table(stats))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the full SARSA-vs-Q-learning study.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None,
                    help="output directory (default: results, or results/smoke with --smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny study to validate the pipeline end-to-end")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    if args.smoke:
        _apply_smoke(cfg)

    out = Path(args.out or ("results/smoke" if args.smoke else "results"))
    _prepare_dirs(out)
    cfg["_out"] = str(out)

    print(f"[1/5] reference values (DP + uniform random) -> {out/'raw'/'reference_values.json'}")
    refs = experiment.reference_values(cfg)

    print(f"[2/5] tuning: {len(cfg['agent']['alphas'])} alphas x "
          f"{len(cfg['env']['slips'])} slips x {len(cfg['tuning']['seeds'])} seeds")
    selected = experiment.run_tuning(cfg, out)
    print(f"      selected alphas -> {out/'raw'/'tuning_selected.csv'}")

    print(f"[3/5] main: {len(cfg['evaluation']['seeds'])} seeds x "
          f"{len(cfg['env']['slips'])} slips x 2 methods @ {cfg['train']['budget']} steps")
    curves, summary = experiment.run_main(cfg, selected, refs, out)
    print(f"      per-seed summary -> {out/'raw'/'summary_per_seed.csv'}")

    print(f"[4/5] statistical tests -> {out/'raw'/'statistics.csv'}")
    stats = experiment.statistical_tests(summary, cfg)

    print(f"[5/5] figures and tables -> {out/'figures'}, {out/'tables'}")
    _make_figures(cfg, refs, curves, summary, out)
    _make_tables(cfg, refs, selected, summary, stats, out)

    n_raw = len(list((out / "raw").iterdir()))
    n_fig = len(list((out / "figures").glob("*.pdf")))
    n_tab = len(list((out / "tables").glob("*.tex")))
    print(f"\ndone -> {out}")
    print(f"  raw:     {n_raw:3d} files")
    print(f"  figures: {n_fig:3d} PDFs")
    print(f"  tables:  {n_tab:3d} .tex files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
