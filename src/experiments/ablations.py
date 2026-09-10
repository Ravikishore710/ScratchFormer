"""Ablation study: train several configurations and record an experiment table.

The dataset + tokenizers are built ONCE and shared across runs so that every
row differs only in the varied hyperparameter.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ..config import Config
from ..data.dataset import prepare_datasets
from ..evaluation.evaluate import quick_gen_metrics
from ..model.builtin import BuiltinTransformer
from ..model.transformer import Transformer, count_parameters
from ..training.trainer import Trainer

# (experiment name, config overrides) -- d_model % num_heads must stay 0
EXPERIMENTS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ("no_positional_encoding", {"use_positional_encoding": False}),
    ("heads_1", {"num_heads": 1}),
    ("heads_2", {"num_heads": 2}),
    ("heads_4", {"num_heads": 4}),
    ("d_model_64", {"d_model": 64, "num_heads": 4}),
    ("d_model_256", {"d_model": 256}),
    ("layers_1", {"num_layers": 1}),
    ("layers_4", {"num_layers": 4}),
    ("d_ff_256", {"d_ff": 256}),
    ("d_ff_1024", {"d_ff": 1024}),
    ("warmup_1000", {"warmup_steps": 1000}),
    ("warmup_8000", {"warmup_steps": 8000}),
]


def _evaluate_row(name, cfg, model, bundle, train_time, epochs):
    em, bleu = quick_gen_metrics(model, bundle, bundle.val_src_text,
                                 bundle.val_tgt_text, n=100)
    return {
        "experiment": name,
        "d_model": cfg.d_model, "num_heads": cfg.num_heads,
        "num_layers": cfg.num_layers, "d_ff": cfg.d_ff,
        "use_pe": cfg.use_positional_encoding, "warmup_steps": cfg.warmup_steps,
        "params": count_parameters(model),
        "epochs": epochs,
        "train_time_s": round(train_time, 1),
        "val_exact_match": round(em, 2),
        "val_bleu": round(bleu, 2),
    }


def run_experiments(base_cfg: Config, bundle=None, epochs: int = 3,
                    experiments=None, out_csv: str = "outputs/experiments/experiment_results.csv"):
    experiments = experiments or EXPERIMENTS
    if bundle is None:
        bundle = prepare_datasets(base_cfg)
    rows = []
    for name, overrides in experiments:
        cfg = replace(base_cfg, **overrides).validate()
        cfg.epochs = epochs
        print(f"\n=== experiment: {name} | {cfg.d_model}d / {cfg.num_heads}h / "
              f"{cfg.num_layers}L / ff{cfg.d_ff} / pe={cfg.use_positional_encoding} ===")
        model = Transformer(cfg, len(bundle.src_tok), len(bundle.tgt_tok))
        trainer = Trainer(model, cfg, checkpoint_dir=None)
        t0 = time.time()
        trainer.fit(bundle.train_ds, bundle.val_ds, log_every=500)
        rows.append(_evaluate_row(name, cfg, model, bundle, time.time() - t0, epochs))
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


def run_builtin_benchmark(base_cfg: Config, bundle, epochs: int = 3,
                          out_csv: str = "outputs/experiments/experiment_results.csv") -> dict:
    """Train the Keras-MHA benchmark with identical hyperparameters."""
    cfg = replace(base_cfg)
    cfg.epochs = epochs
    print("\n=== benchmark: BuiltinTransformer (tf.keras.layers.MultiHeadAttention) ===")
    model = BuiltinTransformer(cfg, len(bundle.src_tok), len(bundle.tgt_tok))
    trainer = Trainer(model, cfg, checkpoint_dir=None)
    t0 = time.time()
    trainer.fit(bundle.train_ds, bundle.val_ds, log_every=500)
    row = _evaluate_row("builtin_keras_mha", cfg, model, bundle, time.time() - t0, epochs)
    df = pd.read_csv(out_csv) if Path(out_csv).exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(out_csv, index=False)
    return row
