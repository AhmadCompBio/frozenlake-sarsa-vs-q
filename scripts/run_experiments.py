"""Main experiment stage only, using alphas selected by run_tuning.

Supports --slips to split the study into shorter batches (e.g. on Colab):
    python -m scripts.run_experiments --slips 0.0 0.1
    python -m scripts.run_experiments --slips 0.2 0.6666666666666666

reference_values.json is MERGED across batches, not overwritten.
Requires tuning_selected.csv to exist (produced by run_tuning or run_all).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src import experiment
from src.fl_env import SlipFrozenLake, uniform_random_value, value_iteration


def _load_selected(path: Path) -> dict:
    df = pd.read_csv(path)
    return {(round(float(r.slip), 6), r.algo): float(r.alpha) for r in df.itertuples()}


def _build_refs(cfg) -> dict:
    """Same computation as experiment.reference_values, but without writing JSON."""
    refs = {}
    for slip in cfg["env"]["slips"]:
        key = round(float(slip), 6)
        env = SlipFrozenLake(slip=slip, desc=cfg["env"]["desc"],
                             max_steps=cfg["env"]["max_steps"])
        V, iters = value_iteration(env.P, env.n_states, env.n_actions, gamma=1.0)
        Vr = uniform_random_value(env.P, env.n_states, env.n_actions)
        refs[key] = dict(V=V, optimal=float(V[0]), random=float(Vr[0]),
                         iterations=int(iters), env=env)
    return refs


def _merge_reference_json(refs: dict, out: Path) -> None:
    path = out / "raw" / "reference_values.json"
    existing = {}
    if path.exists():
        with open(path) as fh:
            existing = json.load(fh)
    for k, v in refs.items():
        existing[str(k)] = dict(optimal=v["optimal"], random=v["random"],
                                iterations=v["iterations"])
    with open(path, "w") as fh:
        json.dump(existing, fh, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the main experiment stage.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--slips", nargs="+", type=float, default=None,
                    help="restrict to these slip values (default: all in config)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    if args.slips is not None:
        cfg["env"]["slips"] = args.slips
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfg["_out"] = str(out)

    selected_path = out / "raw" / "tuning_selected.csv"
    if not selected_path.exists():
        raise SystemExit(f"missing {selected_path}; run run_tuning or run_all first")
    selected = _load_selected(selected_path)

    refs = _build_refs(cfg)
    _merge_reference_json(refs, out)

    curves, summary = experiment.run_main(cfg, selected, refs, out)
    print(f"curves  -> {out/'raw'/'curves_long.csv'}      ({len(curves)} rows)")
    print(f"summary -> {out/'raw'/'summary_per_seed.csv'}  ({len(summary)} rows)")

    experiment.statistical_tests(summary, cfg)
    print(f"stats   -> {out/'raw'/'statistics.csv'}")
    return 0


if __name__ == "__main__":
    main()
