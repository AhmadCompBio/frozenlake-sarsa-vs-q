"""Regenerate ONE LaTeX table from committed raw results. Never retrains.

Usage:
    python -m scripts.make_tables --only mdp
    python -m scripts.make_tables --only hyperparams
    python -m scripts.make_tables --only main
    python -m scripts.make_tables --only stats

Requires:
    raw/reference_values.json (mdp)
    raw/tuning_selected.csv   (hyperparams)
    raw/summary_per_seed.csv  (main)
    raw/statistics.csv        (stats)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src import tables

ONLY = ("mdp", "hyperparams", "main", "stats")


def _load_refs(path: Path) -> dict:
    """reference_values.json keys are strings; return them as rounded floats."""
    with open(path) as fh:
        raw = json.load(fh)
    return {round(float(k), 6): v for k, v in raw.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate one LaTeX table from raw results.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", required=True, choices=ONLY)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    if args.only == "mdp":
        refs = _load_refs(out / "raw" / "reference_values.json")
        text = tables.mdp_table(cfg, refs)
        path = out / "tables" / "table_mdp.tex"

    elif args.only == "hyperparams":
        sel_df = pd.read_csv(out / "raw" / "tuning_selected.csv")
        selected = {(round(float(r.slip), 6), r.algo): float(r.alpha)
                    for r in sel_df.itertuples()}
        text = tables.hyperparameter_table(cfg, selected)
        path = out / "tables" / "table_hyperparams.tex"

    elif args.only == "main":
        summary = pd.read_csv(out / "raw" / "summary_per_seed.csv")
        slips = [round(float(s), 6) for s in cfg["env"]["slips"]]
        text = tables.main_results_table(summary, slips, cfg)
        path = out / "tables" / "table_main.tex"

    else:   # stats
        stats = pd.read_csv(out / "raw" / "statistics.csv")
        text = tables.stats_table(stats)
        path = out / "tables" / "table_stats.tex"

    tables.write(path, text)
    print(f"table written: {path}")
    return 0


if __name__ == "__main__":
    main()
