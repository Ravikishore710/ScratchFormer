"""Run the ablation grid + built-in benchmark (reduced budget by default).

Usage:
    python scripts/run_experiments.py --config configs/baseline.json --epochs 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import Config
from src.data.dataset import prepare_datasets
from src.experiments.ablations import run_builtin_benchmark, run_experiments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()

    cfg = Config.load_json(args.config)
    bundle = prepare_datasets(cfg)          # built once, shared by all rows
    run_experiments(cfg, bundle=bundle, epochs=args.epochs)
    if not args.skip_benchmark:
        run_builtin_benchmark(cfg, bundle, epochs=args.epochs)

    out_csv = f"{cfg.output_dir}/experiments/experiment_results.csv"
    df = pd.read_csv(out_csv)
    print("\n===== experiment results =====")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
