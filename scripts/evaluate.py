"""Final test-set evaluation + failure case harvesting.

Usage:
    python scripts/evaluate.py --config configs/baseline.json --n 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data.dataset import prepare_datasets
from src.model.transformer import Transformer
from src.training.trainer import Trainer
from src.evaluation.evaluate import evaluate_test_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    cfg = Config.load_json(args.config)
    bundle = prepare_datasets(cfg)
    model = Transformer(cfg, len(bundle.src_tok), len(bundle.tgt_tok))
    trainer = Trainer(model, cfg, checkpoint_dir=f"{cfg.output_dir}/model/best_model")
    if not trainer.restore_latest():
        raise SystemExit("No checkpoint found -- train first (scripts/train.py).")

    test_loss, test_acc = trainer.evaluate(bundle.test_ds)
    metrics = evaluate_test_set(
        model, bundle, f"{cfg.output_dir}/predictions/test_predictions.csv", n=args.n)
    metrics["test_loss"] = test_loss
    metrics["test_token_accuracy"] = test_acc

    with open(f"{cfg.output_dir}/predictions/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
