"""Generate positional-encoding and attention visualizations.

Usage:
    python scripts/visualize.py --config configs/baseline.json --n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import Config
from src.data.dataset import prepare_datasets
from src.model.embedding import sinusoidal_position_encoding
from src.model.transformer import Transformer
from src.training.trainer import Trainer
from src.visualization.plots import plot_positional_encoding, visualize_attentions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    cfg = Config.load_json(args.config)
    bundle = prepare_datasets(cfg)

    pe = sinusoidal_position_encoding(cfg.max_seq_len, cfg.d_model)[np.newaxis, ...]
    print(plot_positional_encoding(pe, f"{cfg.output_dir}/positional_encoding/positional_encoding.png"))

    model = Transformer(cfg, len(bundle.src_tok), len(bundle.tgt_tok))
    trainer = Trainer(model, cfg, checkpoint_dir=f"{cfg.output_dir}/model/best_model")
    if not trainer.restore_latest():
        raise SystemExit('No checkpoint found -- train first (scripts/train.py).')

    for i in range(args.n):
        print(visualize_attentions(
            model, bundle, bundle.test_src_text[i], bundle.test_tgt_text[i],
            out_dir=f"{cfg.output_dir}/attention_maps", prefix=f"test_{i}"))


if __name__ == "__main__":
    main()
