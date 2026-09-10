"""Train ScratchFormer.

Usage:
    python scripts/train.py --config configs/baseline.json --epochs 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data.dataset import prepare_datasets
from src.model.transformer import Transformer, count_parameters
from src.training.trainer import Trainer
from src.visualization.plots import plot_training_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.load_json(args.config)
    if args.epochs:
        cfg.epochs = args.epochs
    cfg.validate()
    cfg.save_json(Path(cfg.output_dir) / f"config_{cfg.exp_name}.json")

    bundle = prepare_datasets(cfg)
    model = Transformer(cfg, len(bundle.src_tok), len(bundle.tgt_tok))
    print(f"Trainable parameters: {count_parameters(model):,}")

    trainer = Trainer(model, cfg, checkpoint_dir=f"{cfg.output_dir}/model/best_model")

    # ---- Phase 15: tiny-dataset overfit verification ----
    print("\n[phase 15] tiny-subset overfit check (100 steps on one batch)")
    trainer.overfit_check(bundle.train_ds, steps=100)

    # ---- Phase 16-17: full training ----
    print("\n[phase 16-17] full training")
    history = trainer.fit(bundle.train_ds, bundle.val_ds)
    Path(f"{cfg.output_dir}/training_curves").mkdir(parents=True, exist_ok=True)
    with open(f"{cfg.output_dir}/training_curves/history_{cfg.exp_name}.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, f"{cfg.output_dir}/training_curves")
    print("Done. Best checkpoint saved to", f"{cfg.output_dir}/model/best_model")


if __name__ == "__main__":
    main()
