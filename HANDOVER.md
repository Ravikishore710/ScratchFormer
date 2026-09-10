# HANDOVER.md — ScratchFormer Agent Handover Document

**Project:** ScratchFormer — a Transformer (encoder–decoder) implemented from
scratch in TensorFlow/Keras and trained on English → French translation.
**Repo:** `https://github.com/Ravikishore710/scratchformer`
**This document is the single source of truth.** Read it fully before touching code.

---

## 1. Mission

The code for the entire project is **written but has never been executed** —
not even once. There is no verified result anywhere in this repository.

Your job, in strict order:

1. Set up the environment (§5).
2. Run the test suite and make it pass (§6, §7).
3. Run the pipeline end to end: overfit gate → short training → evaluation →
   visualization (§8).
4. Only then run full training and (optionally) the ablation study (§9).
5. Update `README.md` §8 "Results" with **real measured numbers only** (§10).

**Do not refactor, "improve", or add features before the tests pass.**
If you believe code is wrong, say so and fix the minimal thing.

---

## 2. Hard rules (non-negotiable)

| # | Rule |
|---|------|
| 1 | The main model must NOT use `tf.keras.layers.MultiHeadAttention`, Hugging Face, or any pretrained weights. The only allowed use of the built-in MHA is the benchmark in `src/model/builtin.py`. |
| 2 | Do NOT replace the custom `GradientTape` training loop with `model.fit`. |
| 3 | Mask convention everywhere is **1 = blocked, 0 = allowed**. Never invert it in one place only. |
| 4 | Special-token IDs are fixed by construction: `<PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3`. `PAD=0` is load-bearing — `src/model/transformer.py` builds masks assuming `pad_id=0`. |
| 5 | Tokenizers must be fit on the **train split only**. Never touch test data for fitting anything. |
| 6 | Never write fabricated numbers into the README results table or any CSV. If a run failed, say so. |
| 7 | Do not commit `data/` or `outputs/` contents (they are gitignored; only `.gitkeep` files stay). |

---

## 3. Current status

| Component | Status |
|---|---|
| All source code (`src/`), scripts, configs, tests, notebook, README | ✅ written, syntax-compiled, statically reviewed |
| Unit tests executed | ❌ never run |
| Dataset download | ❌ never run |
| Training / overfit gate | ❌ never run |
| Evaluation, visualization, experiments | ❌ never run |
| Known environment risk | Colab reports **Python 3.13**; `requirements.txt` pins `tensorflow>=2.16` which may not exist for 3.13 — see §5 |

---

## 4. Big picture

```
raw fra-eng corpus (~10 MB, manythings)
  → parse pairs → filter by word length → subsample 60k → 80/10/10 split (seed 42)
  → WordTokenizer fit on train only (PAD/UNK/SOS/EOS + frequency vocab)
  → encode to ids, pad to max_seq_len=40, shift-right for teacher forcing
  → tf.data (batch 128, shuffle train only, prefetch)
  → Transformer (from scratch):
       embedding ×√d_model + sinusoidal PE
       encoder ×2 [MHA self-attn → Add&Norm → FFN → Add&Norm]
       decoder ×2 [masked MHA self-attn → Add&Norm → MHA cross-attn → Add&Norm → FFN → Add&Norm]
       → Dense projection to target vocab
  → padding-aware cross-entropy loss, Adam(0.9, 0.98, 1e-9) + warmup LR schedule,
    gradient clip 1.0, GradientTape loop, best-val checkpointing
  → greedy autoregressive inference (no teacher forcing) → BLEU / exact match
  → attention heatmaps, ablation grid, Keras-MHA benchmark row
```

**Task:** English → French word-level machine translation. Model is small
(~7–9M parameters) so a full epoch set trains in well under an hour on a Colab T4.

---

## 5. Environment setup (do this first)

Target: **Google Colab with GPU** (Runtime → Change runtime type → T4 GPU).

```bash
git clone https://github.com/Ravikishore710/scratchformer.git
cd scratchformer
```

Cell-by-cell:

```python
# cell 1 — check python and preinstalled TF FIRST, before pip
import sys
print(sys.version)
try:
    import tensorflow as tf
    print("tensorflow already installed:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
except ImportError:
    print("tensorflow NOT installed")
```

```bash
# cell 2 — install deps
# If cell 1 showed a working TF (Colab preinstalls one), SKIP this cell or run:
#   !pip install -q numpy pandas matplotlib tqdm pytest
# Only if TF is missing AND pip cannot resolve `tensorflow>=2.16` on Python 3.13:
#   !pip install -q tensorflow --upgrade     # latest release, supports 3.13
#   !pip install -q numpy pandas matplotlib tqdm pytest
```

```python
# cell 3 — sanity
import tensorflow as tf
assert tf.config.list_physical_devices("GPU"), "No GPU — enable T4 in Runtime settings"
print(tf.__version__, tf.config.list_physical_devices("GPU"))
```

**Failure mode to expect:** on Python 3.13, `pip install tensorflow>=2.16` may
say "no matching distribution". Cure: install latest `tensorflow` (≥2.20-era
builds support 3.13), never downgrade Python by hand, never install `tf-nightly`.

---

## 6. Repository map — every file, every tiny detail

### `configs/baseline.json`
One JSON = one full experiment. Loaded via `Config.load_json` (unknown keys are
silently dropped, so typos in this file are ignored — check spelling).
Fields: dataset sizes/splits/seed, vocab caps (`src_vocab_size=20000`,
`tgt_vocab_size=25000`, `min_freq=2`), model dims (`d_model=128`,
`num_heads=8`, `num_layers=2`, `d_ff=512`, `dropout=0.1`,
`use_positional_encoding=true`), training (`batch_size=128`, `epochs=10`,
`warmup_steps=4000`, `label_smoothing=0.0`, `clip_norm=1.0`), inference
(`max_decode_len=40`), io (`output_dir`, `exp_name`).

### `src/config.py`
`@dataclass Config`. `validate()` raises unless `d_model % num_heads == 0` —
the ablation runner relies on this to catch bad head/width combos.
`override(**kw)` returns a validated copy. `exp_name` is only used for
artifact file naming.

### `src/data/tokenizer.py` — `WordTokenizer`
- Regex: `\w+|[^\w\s]`, applied to **lower-cased** text. French accents are
  preserved (`\w` is Unicode-aware). "l'ami" → `["l", "'", "ami"]` — fine,
  that's the intended word-level granularity.
- `build(texts)`: counts frequencies, keeps specials first
  (`PAD, UNK, SOS, EOS` → ids 0–3), then tokens with `freq >= min_freq`
  up to `vocab_size`. **Fitting happens only on train texts** (enforced by
  `dataset.py`, not by the class — don't bypass it).
- `encode(text, add_special, max_len)`: truncates content to `max_len`
  first, then adds specials if asked. Callers therefore pass
  `max_len=max_seq_len-2` for targets so `SOS+content+EOS ≤ max_seq_len`.
- `decode(ids, stop_at_eos=True, skip_special=True)`: stops at EOS, drops
  PAD/SOS/UNK when `skip_special=True` (used for printing predictions).
- `save`/`load` JSON: `id_to_token` list only; the map is rebuilt on load.

### `src/data/dataset.py`
- `download_fra_eng(cfg)`: idempotent; downloads `fra-eng.zip` (~10 MB) from
  manythings, extracts the first `.txt` inside, renames to `data/fra.txt`.
  **If the network blocks the download:** manually download
  `https://www.manythings.org/anki/fra-eng.zip`, extract `fra.txt`, place it at
  `data/fra.txt` — the function then skips downloading.
- `load_pairs`: splits each line on `\t`, needs ≥2 fields, strips, skips the
  header line (`src.lower() == "english"`).
- `prepare_datasets(cfg, verbose=True)`:
  1. filters pairs to word lengths `≤ max_seq_len - 2` (room for SOS/EOS),
  2. subsamples `max_pairs=60000` with `np.random.default_rng(seed=42)`,
  3. splits 80/10/10 with the same rng (train indices first — so the split is
     stable when `max_pairs` changes),
  4. fits `src_tok`/`tgt_tok` on train texts only,
  5. encodes+pads everything to exactly `max_seq_len` with zeros (PAD=0),
  6. builds `tf.data` pipelines.
- `_make_tf_dataset`: `shift_right(tgt) = concat([SOS], tgt[:-1])` — this is
  teacher forcing; emits batches of shape `((src, dec_in), tgt)` where
  `tgt = content + EOS` (right-padded). `shuffle` only on train;
  `.prefetch(tf.data.AUTOTUNE)`.
- Returns `DataBundle(train_ds, val_ds, test_ds, src_tok, tgt_tok,
  val_src_text, val_tgt_text, test_src_text, test_tgt_text, stats)`.
  Raw text lists are kept for generation-based metrics (BLEU needs strings).

### `src/model/masks.py` — convention: **1 = blocked**
- `create_padding_mask(seq, pad_id=0)` → `(B, 1, 1, L)`. Broadcasts over heads
  and queries.
- `create_look_ahead_mask(size)` → `(size, size)`, strictly lower-triangular
  allow. 1 = future.
- `create_decoder_mask(dec_ids, pad_id=0)` → `(B, 1, L, L)` =
  `max(padding_mask, look_ahead)` — uses `maximum`, not `add`, so values stay
  exactly 0/1 (matters for the Keras-MHA boolean conversion in `builtin.py`).

### `src/model/attention.py` — `scaled_dot_product_attention(q, k, v, mask)`
- `scores = q @ kᵀ / sqrt(d_k)`; `d_k` is `float32`-cast from the **key**
  depth.
- Mask added as `mask * -1e9` **before** softmax (so softmax still
  normalizes over allowed positions even if an entire row is masked — never
  produces NaNs for PAD query rows).
- Returns `(output, attention_weights)`; weights are kept purely for
  visualization.

### `src/model/layers.py`
- `MultiHeadAttention`: `wq/wk/wv/wo` Dense layers; `_split_heads`
  reshape→transpose `(B,L,d)→(B,h,L,depth)`; `_merge_heads` the inverse;
  asserts `d_model % num_heads == 0` at construction. `call(q,k,v,mask,
  return_attention=False, training=None)`.
- `PointWiseFFN`: Dense(`d_ff`, relu) → Dropout → Dense(`d_model`).
- `ResidualLayerNorm`: **post-norm** — `LayerNorm(x + Dropout(sublayer(x)))`,
  matching the original paper (not pre-norm). Don't "modernize" it; the
  warmup schedule is what makes post-norm train here.

### `src/model/embedding.py`
- `sinusoidal_position_encoding(max_len, d_model)`: sin on even dims, cos on
  odd dims, `1/10000^(2i/d_model)`. Pure numpy, unit-testable.
- `TokenAndPositionEmbedding`: `Embedding(vocab, d_model) * sqrt(d_model)`
  then `+ pos_encoding[:, :seq_len, :]` then Dropout. The PE matrix is a
  **non-trainable `add_weight` with a constant initializer** — chosen
  specifically because Keras 3 serializes it cleanly (a bare `tf.constant`
  attribute can cause tracing/serialization quirks). `use_positional_encoding
  =False` skips the add (ablation only; embedding scale still applied).

### `src/model/encoder.py` / `src/model/decoder.py`
- Blocks are post-norm: self-attn → Add&Norm → FFN → Add&Norm (encoder);
  masked self-attn → Add&Norm → cross-attn → Add&Norm → FFN → Add&Norm
  (decoder).
- **Decoder wiring (verify if debugging translation quality):**
  self-attention gets the *combined* mask (`create_decoder_mask`); cross-attention
  gets the *encoder padding* mask; cross-attention Q comes from decoder state,
  K/V from encoder output.
- Stacks accept `return_attention=True` and return per-layer weight lists —
  the visualization code depends on this exact structure
  (lists indexed by layer, each `(B, heads, Lq, Lk)`).

### `src/model/transformer.py` — `Transformer(cfg, src_vocab_size, tgt_vocab_size)`
- `call((src_ids, dec_in_ids), training=False, return_attention=False)`
  → logits `(B, Lt, tgt_vocab)` or `(logits, {"encoder": [...], "decoder_self":
  [...], "cross": [...]})`.
- Masks are built **inside** the model from raw ids (PAD=0 hardcoded — see
  Rule 4).
- `count_parameters(model)`: forces one dummy forward on seq length 8, then
  sums weight sizes. Requires `max_seq_len ≥ 8` (always true here).
- ⚠️ **Checkpoint restore constraint:** restoring requires rebuilding the
  model with identical vocab sizes and architecture. `scripts/evaluate.py`
  and `scripts/visualize.py` rebuild tokenizers by re-running
  `prepare_datasets(cfg)` with the **same config JSON** — never change the
  config between train and evaluate, or restore will fail/corrupt.

### `src/model/builtin.py` — benchmark only
`BuiltinTransformer` with the same call signature (minus attention capture —
it raises on `return_attention=True`). `_keras_mask` converts our float
0/1 masks to boolean **True = attend** for Keras MHA. The block shares `norm2`
between cross-attn and FFN residuals when cross-attn is absent — intentional,
benchmarks still get their own parameters.

### `src/training/losses.py`
- `masked_loss`: per-token CE (or label-smoothed CE), multiplied by the
  non-PAD mask, **divided by the number of non-PAD tokens** (per-batch
  token mean — this is deliberate, not per-sequence).
- `masked_accuracy`: same masking logic, argmax vs label.

### `src/training/lr_schedule.py`
Original transformer schedule: `lr = d_model^-0.5 · min(step^-0.5, step ·
warmup^-1.5)`. Implements `get_config` (Keras 3 requirement for schedules).

### `src/training/trainer.py` — `Trainer(model, cfg, checkpoint_dir=None)`
- Optimizer: `Adam(schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)` —
  β2=0.98 and ε=1e-9 are from the paper, don't "fix" them to defaults.
- Steps are wrapped in `tf.function`. Shapes are fixed by the dataset, so
  one trace per step function — good.
- `fit(train_ds, val_ds, log_every)`: per-epoch history dict, prints epoch
  summaries, **saves checkpoint whenever val_loss improves**.
- `evaluate(ds, max_batches=None)`: teacher-forced val loss/accuracy.
- `overfit_check(train_ds, steps=100)`: trains on the FIRST batch only.
  **Gate:** raises `RuntimeError` if `losses[-1] > 0.5 × losses[0]`.
  Expected on a working setup: loss falls from ~10 (≈ln 25000) to <1.0.
- Checkpointing via `tf.train.CheckpointManager`, restore with
  `.expect_partial()` (the optimizer's schedule step counter is allowed to be
  missing).

### `src/inference/generate.py`
- `translate(model, src_tok, tgt_tok, text, max_len)`: greedy loop —
  append `argmax(logits[:, -1])` until EOS or `max_len`.
- **Deliberately eager** (no `@tf.function`): a traced version would re-trace
  on every growing sequence length (up to 40 re-traces per sentence). Eager is
  ~40 small forwards per sentence — slow-ish but correct; do not "optimize"
  this without measuring.
- `translate_batch` is a simple python loop over `translate`.

### `src/evaluation/metrics.py`
- `corpus_bleu`: dependency-free BLEU-4 with **+1 smoothing** on modified
  precisions and brevity penalty. It is a *simple* BLEU, not sacreBLEU —
  compare numbers only within this project.
- `exact_match`: fraction of identical strings (after strip), 0–100.

### `src/evaluation/evaluate.py`
- `quick_gen_metrics(model, bundle, src_texts, tgt_texts, n, max_len)`:
  greedy-translates `n` examples, returns `(exact_match, bleu)`.
- `evaluate_test_set(model, bundle, out_csv, n, max_len)`: greedy-translates
  `n` test examples, writes CSV with `source/reference/hypothesis/
  exact_match/ref_len/hyp_len`, returns metrics including
  `n_premature_eos` (heuristic: `hyp_len < 0.5 × ref_len`) — a degeneration
  signal worth reporting, not a hard failure.

### `src/visualization/plots.py`
- `matplotlib` forced to `Agg` (headless-safe).
- `plot_training_curves(history, out_dir)` → `loss.png`, `acc.png`.
- `plot_positional_encoding(pe, path)` — expects `(1, max_len, d_model)`.
- `plot_attention_grid(weights, row_tokens, col_tokens, title, path)`:
  weights `(heads, Lq, Lk)`; token labels drawn only when ≤12 tokens (keeps
  heatmaps readable; longer sentences still render, just unlabeled).
- `visualize_attentions(model, bundle, src_text, tgt_text, out_dir, prefix)`:
  runs ONE teacher-forced forward with `return_attention=True` and writes
  per-layer grids into `encoder/`, `decoder_self/`, `cross/`. Note it uses a
  fixed window of 40 tokens.

### `src/experiments/ablations.py`
- `EXPERIMENTS`: 13 named configs — baseline, `no_positional_encoding`,
  `heads_1/2/4`, `d_model_64` (heads auto-set to 4), `d_model_256`,
  `layers_1/4`, `d_ff_256/1024`, `warmup_1000/8000`.
- `run_experiments(base_cfg, bundle=None, epochs=3, experiments=None)`:
  builds the dataset ONCE if not provided (shared tokenizers → every row
  differs only in the varied hyperparameter). Trains each config, evaluates
  100 greedy generations on val, **rewrites the CSV after every experiment**
  (crash-safe), columns: `experiment, d_model, num_heads, num_layers, d_ff,
  use_pe, warmup_steps, params, epochs, train_time_s, val_exact_match, val_bleu`.
- `run_builtin_benchmark(base_cfg, bundle, epochs)`: trains
  `BuiltinTransformer` identically and appends the `builtin_keras_mha` row.

### `scripts/`
Thin CLIs; each inserts the repo root into `sys.path` so they work from
anywhere: `train.py` (overfit gate → full fit → curves; saves config copy +
history JSON), `evaluate.py` (requires checkpoint), `visualize.py` (requires
checkpoint), `run_experiments.py` (`--skip-benchmark` flag exists).

### `tests/` — 11 tests, what each proves
- `test_masks.py` (3): padding mask shape/values; causal mask equals the exact
  expected matrix; combined decoder mask blocks future keys AND pad keys.
- `test_attention.py` (3): output shapes; softmax rows sum to 1; masked key
  gets ≈0 weight and remaining rows renormalize; scaling keeps weights
  well-conditioned.
- `test_model.py` (5): forward shapes; finite loss + bounded accuracy;
  **gradients exist and are finite for every trainable variable**; attention
  capture shapes per layer (`encoder (1,h,Ls,Ls)`, `decoder_self (1,h,Lt,Lt)`,
  `cross (1,h,Lt,Ls)`); parameter count positive.

---

## 7. Step 1 — run the tests

From the **repo root** (the directory containing `src/`, `tests/`, `pytest
must see `src` as an importable package):

```bash
cd scratchformer          # wherever you cloned it
python -m pytest tests/ -v
```

**Expected: 11 passed, 0 failed.** Runtime is seconds on CPU.

If `python -m pytest tests/ -v` fails with `ModuleNotFoundError: src`, you are
not in the repo root — `cd` first (the earlier `rootdir: /content` incident
was exactly this). If your pytest setup insists on a conftest, create an empty
`conftest.py` at the repo root (its presence puts the root on `sys.path`).

**If a test fails:** read the test — each asserts a specific invariant listed
in §6. Fix the code (not the test) unless the test itself contradicts §2/§6,
in which case stop and report the contradiction.

---

## 8. Step 2 — smoke-test the full pipeline (short budget)

Do these before any long run. Each prints progress; expected behavior in
italics.

```bash
# 2a. data pipeline (downloads ~10 MB; prints stats + batch shapes)
python -c "from src.config import Config; from src.data.dataset import prepare_datasets; \
b = prepare_datasets(Config(max_pairs=2000)); \
print({k: v for k, v in b.stats.items()}); \
print(next(iter(b.train_ds.take(1)))[0][0].shape)"
# expect: stats with train=1600, val=200, test=200; batch src (128, 40)
```

```bash
# 2b. training smoke: 1 epoch on 2k pairs (~1-2 min on T4).
# Expect: '[checkpoint] saved ...' each epoch; overfit gate PASS; val loss finite.
python - <<'EOF'
from src.config import Config
from src.data.dataset import prepare_datasets
from src.model.transformer import Transformer, count_parameters
from src.training.trainer import Trainer
cfg = Config(max_pairs=2000, epochs=1, batch_size=64)
b = prepare_datasets(cfg, verbose=False)
m = Transformer(cfg, len(b.src_tok), len(b.tgt_tok))
print("params:", count_parameters(m))
t = Trainer(m, cfg, checkpoint_dir="outputs/model/smoke")
t.overfit_check(b.train_ds, steps=50)
h = t.fit(b.train_ds, b.val_ds, log_every=10)
print("history:", h)
EOF
```

Sanity numbers for 2b: initial loss ≈ 9–11 (ln of vocab ≈ 10.1); after 1 epoch
val loss typically 4–7; exact values vary — what matters is **monotone-ish
decrease and no NaN**.

```bash
# 2c. inference + eval smoke (~1 min: 20 greedy translations, small model)
python - <<'EOF'
from src.config import Config
from src.data.dataset import prepare_datasets
from src.model.transformer import Transformer
from src.training.trainer import Trainer
from src.inference.generate import translate
from src.evaluation.evaluate import evaluate_test_set
cfg = Config(max_pairs=2000)
b = prepare_datasets(cfg, verbose=False)
m = Transformer(cfg, len(b.src_tok), len(b.tgt_tok))
t = Trainer(m, cfg, checkpoint_dir="outputs/model/smoke"); t.restore_latest()
print(translate(m, b.src_tok, b.tgt_tok, b.test_src_text[0]))
print(evaluate_test_set(m, b, "outputs/predictions/smoke.csv", n=20))
EOF
```

```bash
# 2d. visualization smoke (requires the smoke checkpoint from 2b)
python - <<'EOF'
from src.config import Config
from src.data.dataset import prepare_datasets
from src.model.transformer import Transformer
from src.training.trainer import Trainer
from src.visualization.plots import visualize_attentions
from src.model.embedding import sinusoidal_position_encoding
import numpy as np
cfg = Config(max_pairs=2000)
b = prepare_datasets(cfg, verbose=False)
m = Transformer(cfg, len(b.src_tok), len(b.tgt_tok))
t = Trainer(m, cfg, checkpoint_dir="outputs/model/smoke"); t.restore_latest()
pe = sinusoidal_position_encoding(cfg.max_seq_len, cfg.d_model)
paths = visualize_attentions(m, b, b.test_src_text[0], b.test_tgt_text[0],
                             out_dir="outputs/attention_maps", prefix="smoke")
print(len(paths), "figures written")
EOF
```

If 2a–2d all behave, the project is healthy. Delete `outputs/model/smoke` and
the smoke CSVs before the real run.

---

## 9. Step 3 — full runs

```bash
# 3a. full baseline training (60k pairs, 10 epochs). Expect ~15-40 min on T4.
python scripts/train.py --config configs/baseline.json --epochs 10

# 3b. held-out test evaluation (uses best checkpoint; ~5-10 min for 500 gens)
python scripts/evaluate.py --config configs/baseline.json --n 500

# 3c. attention visualizations for 3 test examples
python scripts/visualize.py --config configs/baseline.json --n 3

# 3d. ablation grid + Keras-MHA benchmark, reduced budget (several hours).
#     Run overnight or in a second Colab session in parallel if desired.
python scripts/run_experiments.py --config configs/baseline.json --epochs 3
```

Expected quality ballpark (word-level MT, small model, greedy decoding —
**targets, not guarantees**): val token accuracy 40–60%, test BLEU 10–25,
exact match low single digits. Higher is better, but any clean monotone
training curve + finite metrics = success for this project.

---

## 10. Step 4 — report & finish

1. Fill `README.md` §8's table with the **measured** numbers from
   `outputs/experiments/experiment_results.csv` and §7 metrics.
2. Optionally copy the best figures from `outputs/` into a `docs/` folder and
   embed 2–3 in the README (`outputs/` itself stays gitignored).
3. Commit with message: `test: verified pipeline end-to-end; results in README`
4. Write a short `REPORT.md` at the repo root: environment (TF version, GPU),
   final metrics, any test/training failures seen and how they were resolved,
   and which ablations surprised you. **If anything in this handover turned
   out to be wrong, correct the handover too.**

---

## 11. Troubleshooting playbook (ordered by likelihood)

| Symptom | Likely cause | Fix |
|---|---|---|
| `No matching distribution found for tensorflow` on py3.13 | TF 2.16 has no cp313 wheel | Use Colab's preinstalled TF, or `pip install --upgrade tensorflow` (§5) |
| `pytest` collects 0 items / `ModuleNotFoundError: src` | not in repo root | `cd scratchformer`; run `python -m pytest tests/ -v` (§7) |
| `OSError`/404 downloading fra-eng.zip | network block | manual download → place `fra.txt` in `data/` (§6 dataset.py) |
| Loss = NaN in first epochs | rare with clip=1.0 + warmup | confirm warmup_steps>0, clip_norm=1.0, dropout 0.1; try batch_size=64 |
| Overfit gate RuntimeError | real bug OR batch too small | verify with steps=200; if still failing, inspect gradient flow test — report, don't patch blindly |
| Checkpoint restore silent mismatch | config changed between train/evaluate | use the same `--config` file; vocab sizes derive from the same seed/data |
| `BuiltinTransformer` mask error | Keras version changed mask API | report exact TF/Keras version + traceback; do not rewrite the main model |
| Inference feels slow | by design (eager, ~40 fw/sentence) | acceptable; do not add tf.function blindly (see §6 generate.py) |
| OOM on T4 | batch 128 × seq 40 × d_model 256 rows | lower `batch_size` in config for big ablation rows |
| GPU not found | CPU runtime | Runtime → Change runtime type → T4 GPU |

---

## 12. Conventions cheat-sheet

| Thing | Value / rule |
|---|---|
| Mask semantics | `1 = blocked`, added as `mask * -1e9` pre-softmax |
| Special token ids | PAD=0, UNK=1, SOS=2, EOS=3 |
| Teacher forcing | dec_in = `[SOS] + tgt[:-1]`; loss target = `tgt` (content+EOS) |
| Loss | token-mean CE over non-PAD positions |
| Optimizer | Adam(β1=0.9, β2=0.98, ε=1e-9), warmup schedule, clip 1.0 |
| Norm style | post-norm residual (original paper) |
| Attention scale | `1/sqrt(d_k)`, d_k = key depth per head |
| Inference | greedy, stops at EOS or `max_decode_len=40` |
| Seeds | numpy `default_rng(42)` for splits/subsample; TF seed set in notebook |
| Split | 80/10/10, train-only tokenizer fitting, test untouched until §9 step 3b |

---

*End of handover. When in doubt: tests (§7) → smoke (§8) → full (§9) → report (§10).*
