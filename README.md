# ScratchFormer

**Implement, train, inspect, and ablate a Transformer from scratch in TensorFlow.**

ScratchFormer is a complete experimental deep-learning project: a small
encoder–decoder Transformer built entirely from low-level TensorFlow/Keras
building blocks — no `tf.keras.layers.MultiHeadAttention`, no Hugging Face, no
pretrained weights. It goes all the way from a raw parallel corpus to a trained
machine-translation model with attention visualizations and a full ablation study.

> I didn't just *use* Transformers — I understand the architecture well enough
> to implement, train, debug, inspect, experiment with, and evaluate one from
> the ground up.

---

## 1. Project overview

We build this pipeline end to end:

```
raw dataset → tokenizer → vocabulary → tensors → masks → embeddings
→ positional encoding → scaled dot-product attention → multi-head attention
→ encoder → decoder → Transformer → training → autoregressive inference
→ attention visualization → ablation study → evaluation
```

**Task:** English → French machine translation on the
[manythings fra-eng corpus](https://www.manythings.org/anki/) (a real, small,
public-domain parallel corpus that trains in minutes on a Colab GPU).

**Rule honored everywhere:** the Transformer itself is implemented by hand.
Keras' ready-made `MultiHeadAttention` appears only in a clearly-labeled
benchmark model (`src/model/builtin.py`) so we can compare against it.

## 2. Objective

* Demonstrate complete command of the Transformer architecture (Vaswani et al., 2017).
* Keep every design decision inspectable: masks, attention weights, loss masking,
  teacher forcing, and autoregressive decoding are all explicit.
* Turn the implementation into an *experimental study*: ablations, a benchmark
  against the framework implementation, error analysis, and reproducible artifacts.

## 3. Dataset

| Property        | Value                                    |
| --------------- | ---------------------------------------- |
| Source          | manythings `fra-eng` (English ↔ French)  |
| Examples used   | 60,000 pairs (configurable)              |
| Split           | 80% train / 10% validation / 10% test    |
| Max seq length  | 40 tokens (word-level)                   |
| Tokenizer       | custom regex word tokenizer (`src/data`) |
| Special tokens  | `<PAD>` `<UNK>` `<SOS>` `<EOS>`          |
| Leakage control | tokenizers are fit on the **train split only** |

* Download + statistics: `src/data/dataset.py`
* Vocabulary construction: `src/data/tokenizer.py` — explicit `token → id` / `id → token`
  maps, saved to JSON.

## 4. Architecture

```
                     INPUT (token ids)
                          │
            Token Embedding × sqrt(d_model)
                     + Positional Encoding
                          │
                       ENCODER × N
        ┌─────────────────────────────────────┐
        │  Multi-Head Self-Attention          │
        │  Add & LayerNorm                    │
        │  Feed-Forward (d_ff)                │
        │  Add & LayerNorm                    │
        └─────────────────────────────────────┘
                          │  encoder output
   TARGET (shifted right) │
            Token Embedding × sqrt(d_model)
                     + Positional Encoding
                          │
                       DECODER × N
        ┌─────────────────────────────────────┐
        │  Masked Multi-Head Self-Attention   │
        │  Add & LayerNorm                    │
        │  Multi-Head Cross-Attention         │
        │  Add & LayerNorm                    │
        │  Feed-Forward (d_ff)                │
        │  Add & LayerNorm                    │
        └─────────────────────────────────────┘
                          │
                Linear → vocab logits
```

| Hyperparameter | Default |
| -------------- | ------- |
| `d_model`      | 128     |
| `num_heads`    | 8       |
| `num_layers`   | 2 (enc) / 2 (dec) |
| `d_ff`         | 512     |
| `dropout`      | 0.1     |
| batch size     | 128     |
| optimizer      | Adam (β1=0.9, β2=0.98, ε=1e-9) |
| LR schedule    | Transformer warmup schedule (`warmup_steps=4000`) |
| loss           | padding-aware token cross-entropy (optional label smoothing) |

## 5. Implementation map

| Component | File |
| --- | --- |
| Padding / causal / combined masks | `src/model/masks.py` |
| Scaled dot-product attention | `src/model/attention.py` |
| Multi-head attention (manual head split/merge) | `src/model/layers.py` |
| Sinusoidal positional encoding | `src/model/embedding.py` |
| Encoder block + stack | `src/model/encoder.py` |
| Decoder block + stack (masked self-attn, cross-attn) | `src/model/decoder.py` |
| Complete Transformer | `src/model/transformer.py` |
| Keras-MHA benchmark model | `src/model/builtin.py` |
| Padding-aware loss / accuracy | `src/training/losses.py` |
| Warmup LR schedule | `src/training/lr_schedule.py` |
| `GradientTape` training loop, checkpointing, overfit check | `src/training/trainer.py` |
| Greedy autoregressive decoding | `src/inference/generate.py` |
| Exact match + corpus BLEU | `src/evaluation/metrics.py` |
| Test-set evaluation + failure harvest | `src/evaluation/evaluate.py` |
| Training curves, positional encoding, attention heatmaps | `src/visualization/plots.py` |
| Ablation grid + benchmark | `src/experiments/ablations.py` |

Masking convention used throughout: **1 = blocked, 0 = allowed**; masks are
added as `mask * -1e9` to attention logits and broadcast over
`(batch, heads, query_len, key_len)`.

## 6. Training methodology

1. **Sanity checks** (`tests/`): mask semantics, softmax normalization, forward
   shapes, finite loss, gradient flow to every layer.
2. **Tiny-subset overfit verification** (Phase 15): the model must drive the
   loss of a single batch to near zero within ~100 steps. If it cannot, training
   stops and we debug.
3. **Full training** with `tf.GradientTape` (no `model.fit`), gradient clipping,
   best-val-loss checkpointing, and per-epoch validation.
4. **Teacher forcing** during training (decoder input = `<SOS> A B C D`,
   target = `A B C D <EOS>`); **greedy autoregressive decoding** at inference —
   no cheating.

## 7. Quickstart (Google Colab)

```bash
# 1. clone the repo in a Colab cell and move into it
!git clone https://github.com/<your-username>/scratchformer.git
%cd scratchformer
!pip install -r requirements.txt

# 2. train (downloads ~10 MB of data automatically, uses the GPU)
!python scripts/train.py --config configs/baseline.json --epochs 10

# 3. evaluate on the held-out test set
!python scripts/evaluate.py --config configs/baseline.json --n 500

# 4. attention + positional-encoding visualizations
!python scripts/visualize.py --config configs/baseline.json --n 3

# 5. ablation study (reduced budget) + built-in benchmark
!python scripts/run_experiments.py --config configs/baseline.json --epochs 3
```

The notebook `notebooks/07_Transformer_From_Scratch.ipynb` walks through every
phase interactively, mirroring the 31 sections of the project blueprint.

## 8. Results

Results are written under `outputs/` and are **not** committed to git
(folder skeleton is kept via `.gitkeep` files). After running the pipeline you get:

```
outputs/
├── training_curves/     # loss.png, acc.png, history JSON
├── attention_maps/
│   ├── encoder/         # encoder self-attention heatmaps per layer
│   ├── decoder_self/    # decoder masked self-attention heatmaps
│   └── cross/           # cross-attention heatmaps (target rows × source cols)
├── positional_encoding/ # PE matrix + per-dimension curves
├── experiments/         # experiment_results.csv (ablation table)
├── predictions/         # test_predictions.csv + test_metrics.json
└── model/best_model/    # best-val-loss checkpoint
```

### Final experimental table (produced by `scripts/run_experiments.py`)

| Experiment    | What varies                     | Val EM | Val BLEU | Train time | Params |
| ------------- | ------------------------------- | -----: | -------: | ---------: | -----: |
| baseline      | 128d / 8h / 2L / ff512          |    —   |    —     |     —      |   —    |
| no PE         | positional encoding removed     |    —   |    —     |     —      |   —    |
| 1 head        | num_heads = 1                   |    —   |    —     |     —      |   —    |
| 2/4 heads     | num_heads = 2 / 4               |    —   |    —     |     —      |   —    |
| d_model 64/256| model width                     |    —   |    —     |     —      |   —    |
| 1/4 layers    | stack depth                     |    —   |    —     |     —      |   —    |
| d_ff 256/1024 | FFN capacity                    |    —   |    —     |     —      |   —    |
| warmup 1k/8k  | LR schedule                     |    —   |    —     |     —      |   —    |
| builtin       | Keras `MultiHeadAttention`      |    —   |    —     |     —      |   —    |

*(Numbers are produced by the runs — nothing is fabricated.)*

## 9. Ablation study & scientific questions

The experiment runner is designed to answer:

* Does positional information matter? (`baseline` vs `no PE`)
* Does multi-head beat single-head attention at this scale?
* How sensitive is the model to width (`d_model`), depth (`num_layers`),
  FFN capacity (`d_ff`), and the warmup schedule?
* What is the size/performance trade-off (params vs BLEU)?
* How does the hand-built model compare with Keras' fused `MultiHeadAttention`
  in speed and quality? (built-in benchmark row)

Sequence-length and masking-effect ablations are natural extensions: vary
`max_seq_len` / `max_decode_len` in `configs/baseline.json`, or train with a
corrupted mask by editing `src/model/masks.py` and compare rows.

## 10. Attention visualizations

Heatmaps show, per layer and head:

* **Encoder self-attention** — which source words attend to which.
* **Decoder self-attention** — strict causality: token *i* never attends beyond *i*.
* **Cross-attention** — which source words each generated target word uses.

Representative figures are saved to `outputs/attention_maps/`; the best ones
can be copied into this README (add them under `docs/` if you publish them).

## 11. Error analysis

`scripts/evaluate.py` writes `outputs/predictions/test_predictions.csv` with
`source / reference / hypothesis / exact_match / ref_len / hyp_len` per example,
plus a `test_metrics.json` summary including the count of *premature-EOS*
predictions (`hyp_len < 50% of ref_len`) — a quick proxy for degeneration
failure modes worth inspecting by hand.

## 12. Limitations

* Word-level tokenization (no subwords) caps translation quality on morphology.
* Greedy decoding only (no beam search).
* Small dataset / small model: this is a learning and experimentation project,
  not an SOTA MT system.
* The built-in benchmark uses eager Keras components with identical depth/width,
  but Keras' fused kernels can behave differently under XLA — treat timings
  as indicative, not benchmark-grade.

## 13. Reproducibility

* Every run is fully described by `configs/baseline.json` (saved into
  `outputs/config_<exp_name>.json` at train time).
* `seed=42` for numpy/shuffling; deterministic splits.
* Tests: `pytest tests/ -v`
* Environment: `pip install -r requirements.txt`, Python ≥ 3.10,
  TensorFlow ≥ 2.16 (GPU recommended for full training; tests run on CPU).

## 14. Repository structure

```
scratchformer/
├── notebooks/07_Transformer_From_Scratch.ipynb
├── configs/baseline.json
├── scripts/
│   ├── train.py            # sanity: overfit check → full training
│   ├── evaluate.py         # held-out test evaluation + failure harvest
│   ├── visualize.py        # positional encoding + attention heatmaps
│   └── run_experiments.py  # ablation grid + built-in benchmark
├── src/
│   ├── config.py
│   ├── data/               # download, dataset, tokenizer
│   ├── model/              # masks, attention, layers, embedding, encoder, decoder,
│   │                       # transformer, builtin benchmark
│   ├── training/           # losses, LR schedule, GradientTape trainer
│   ├── inference/          # greedy autoregressive decoding
│   ├── evaluation/         # BLEU / exact match / prediction harvest
│   ├── visualization/      # curves + heatmaps
│   └── experiments/        # ablation runner
├── tests/                  # mask, attention, model, gradient tests
├── outputs/                # artifacts (gitignored except .gitkeep)
├── requirements.txt
├── LICENSE                 # MIT
└── .gitignore
```

## 15. Definition of done

- [x] Real seq2seq dataset with clean train/val/test split
- [x] Custom tokenizer/vocabulary with `<PAD> <UNK> <SOS> <EOS>`
- [x] Efficient `tf.data` pipeline (batch, prefetch, teacher-forcing shift)
- [x] Sinusoidal positional encoding (visualized, ablatable)
- [x] Scaled dot-product attention from scratch
- [x] Padding mask, causal mask, combined decoder mask
- [x] Multi-head attention with manual head split/merge
- [x] FFN, residual connections, LayerNorm
- [x] Encoder stack + decoder stack (masked self-attn + cross-attn)
- [x] Complete Transformer with final vocabulary projection
- [x] Padding-aware loss + custom `GradientTape` loop + warmup LR
- [x] Tiny-subset overfitting verification gate
- [x] Autoregressive inference (greedy, stops at `<EOS>`)
- [x] Test evaluation + failure-case harvesting
- [x] Attention extraction + visualization (encoder / decoder / cross)
- [x] Ablation study (PE, heads, width, depth, FFN, warmup) + results CSV
- [x] Built-in Keras-MHA benchmark row
- [x] Unit tests (`pytest tests/`)
- [x] Professional README + reproducible configs

## 16. Reference

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
> Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need.*
> NeurIPS 2017. https://arxiv.org/abs/1706.03762
