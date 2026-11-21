"""Tuning stage only. Writes tuning_runs.csv and tuning_selected.csv.

Safe to re-run without redoing main experiments. Reads the same config as
run_all.py; does not modify reference_values.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src import experiment


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run only the tuning stage.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfg["_out"] = str(out)

    selected = experiment.run_tuning(cfg, out)
    print("tuning selected alphas:")
    for (slip, algo), alpha in sorted(selected.items()):
        print(f"  slip={slip}  {algo:10s}  alpha={alpha}")
    print(f"-> {out/'raw'/'tuning_selected.csv'}")
    return 0


if __name__ == "__main__":
    main()
