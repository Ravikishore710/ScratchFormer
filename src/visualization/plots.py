"""All figures: training curves, positional encoding, attention heatmaps."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_training_curves(history: dict, out_dir: str) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for key, ylabel in [("loss", "Loss"), ("acc", "Accuracy")]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(history[f"train_{key}"], marker="o", label=f"train {key}")
        ax.plot(history[f"val_{key}"], marker="s", label=f"val {key}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Training / validation {key}")
        ax.grid(alpha=0.3)
        ax.legend()
        path = Path(out_dir) / f"{key}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_positional_encoding(pe: np.ndarray, out_path: str) -> str:
    """pe: (1, max_len, d_model)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    matrix = pe[0]
    im = axes[0].imshow(matrix, aspect="auto", cmap="viridis")
    axes[0].set_title("Positional encoding matrix")
    axes[0].set_xlabel("Embedding dimension")
    axes[0].set_ylabel("Position")
    fig.colorbar(im, ax=axes[0])
    axes[1].plot(matrix[:, : min(8, matrix.shape[1])])
    axes[1].set_title("First dimensions vs position")
    axes[1].set_xlabel("Position")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_attention_grid(weights: np.ndarray, row_tokens: list[str], col_tokens: list[str],
                        title: str, out_path: str) -> str:
    """weights: (num_heads, Lq, Lk). One heatmap per head in a grid."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    heads = weights.shape[0]
    cols = min(4, heads)
    rows = int(np.ceil(heads / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for h in range(heads):
        ax = axes[h]
        im = ax.imshow(weights[h], cmap="viridis", vmin=0, vmax=weights.max())
        ax.set_title(f"head {h}", fontsize=9)
        if len(col_tokens) <= 12:
            ax.set_xticks(range(len(col_tokens)), col_tokens, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(row_tokens)), row_tokens, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for h in range(heads, len(axes)):
        axes[h].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def visualize_attentions(model, bundle, src_text: str, tgt_text: str,
                         out_dir: str, prefix: str = "sample") -> list[str]:
    """Teacher-forced forward pass with attention capture; saves heatmaps."""
    src_ids = bundle.src_tok.encode(src_text, add_special=False, max_len=40)
    tgt_ids = bundle.tgt_tok.encode(tgt_text, add_special=True, max_len=40)
    n = len(tgt_ids)
    dec_in = [bundle.tgt_tok.sos_id] + tgt_ids[:-1]

    src = tf.constant([src_ids + [bundle.src_tok.pad_id] * (40 - len(src_ids))], tf.int64)
    dec = tf.constant([dec_in + [bundle.tgt_tok.pad_id] * (40 - n)], tf.int64)
    _, attn = model((src, dec), training=False, return_attention=True)

    src_tokens = bundle.src_tok.tokenize(src_text)[:40]
    tgt_tokens = bundle.tgt_tok.decode(tgt_ids, stop_at_eos=True, skip_special=True).split()

    paths = []
    for layer_i, w in enumerate(attn["encoder"]):
        w = w[0, :, : len(src_tokens), : len(src_tokens)].numpy()
        paths.append(plot_attention_grid(
            w, src_tokens, src_tokens,
            f"Encoder self-attention, layer {layer_i}",
            f"{out_dir}/encoder/{prefix}_layer{layer_i}.png"))
    for layer_i, w in enumerate(attn["decoder_self"]):
        w = w[0, :, : len(tgt_tokens), : len(tgt_tokens)].numpy()
        paths.append(plot_attention_grid(
            w, tgt_tokens, tgt_tokens,
            f"Decoder self-attention, layer {layer_i}",
            f"{out_dir}/decoder_self/{prefix}_layer{layer_i}.png"))
    for layer_i, w in enumerate(attn["cross"]):
        w = w[0, :, : len(tgt_tokens), : len(src_tokens)].numpy()
        paths.append(plot_attention_grid(
            w, tgt_tokens, src_tokens,
            f"Cross-attention (Q=target, K=source), layer {layer_i}",
            f"{out_dir}/cross/{prefix}_layer{layer_i}.png"))
    return paths
