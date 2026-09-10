# ScratchFormer: Technical & Empirical Report

**Project:** From-Scratch Encoder-Decoder Transformer for Machine Translation (English → French)  
**Repository:** [https://github.com/Ravikishore710/ScratchFormer](https://github.com/Ravikishore710/ScratchFormer)  
**Author:** Venkata Ravi Kishore ([LinkedIn](https://www.linkedin.com/in/ravii-kishorre) | [GitHub](https://github.com/Ravikishore710) | [Email](mailto:venkataravikishore710@gmail.com))  
**Date:** September 2026  

---

## 1. Executive Summary

**ScratchFormer** is a complete, mathematically rigorous Transformer encoder-decoder built entirely from first principles in TensorFlow/Keras without relying on high-level library abstractions (such as `tf.keras.layers.MultiHeadAttention` in the primary model). 

The project encompasses:
- Custom **Scaled Dot-Product Attention** with manual projection matrices, dynamic causal lookahead masks, and key/query padding masks.
- Fixed **Sinusoidal Positional Encoding** (Vaswani et al., 2017).
- Post-norm residual layer normalization stacks for both encoder and decoder blocks.
- End-to-end custom training loop implemented via `tf.GradientTape` with warmup-decay learning rate scheduling, label smoothing, and gradient clipping.
- Invariant unit testing verifying core mathematical properties (`11 passed, 0 failed`).
- Empirical study on 60,000 English–French sentence pairs trained on modern accelerator hardware (Kaggle Tesla T4 GPU).
- Comprehensive 14-experiment ablation grid evaluating width, depth, head count, positional encoding, FFN capacity, warmup schedules, and benchmarking against Keras' native `MultiHeadAttention`.

---

## 2. Invariant Unit Test Suite Verification

The project is governed by an invariant test suite (`tests/`) that verifies fundamental mathematical, architectural, and optimization guarantees without compromising the implementation to artificially satisfy tests.

All tests execute via `pytest tests/ -v`:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-9.0.2, pluggy-1.6.0
rootdir: scratchformer
collected 11 items

tests/test_attention.py::test_attention_output_and_weight_shapes PASSED  [  9%]
tests/test_attention.py::test_softmax_normalization_and_masking PASSED   [ 18%]
tests/test_attention.py::test_scaling_by_sqrt_d_k PASSED                 [ 27%]
tests/test_masks.py::test_padding_mask_shape_and_values PASSED           [ 36%]
tests/test_masks.py::test_look_ahead_mask_is_causal PASSED               [ 45%]
tests/test_masks.py::test_decoder_mask_combines_padding_and_causality PASSED [ 54%]
tests/test_model.py::test_transformer_forward_pass_shapes PASSED         [ 63%]
tests/test_model.py::test_loss_is_finite_and_accuracy_is_bounded PASSED  [ 72%]
tests/test_model.py::test_gradients_exist_and_are_finite PASSED          [ 81%]
tests/test_model.py::test_attention_capture_shapes PASSED                [ 90%]
tests/test_model.py::test_parameter_count PASSED                         [100%]

============================= 11 passed in 21.89s =============================
```

### Invariant Checks Summary
1. **Scaled Dot-Product Attention**:
   - Softmax rows strictly sum to $1.0$ across all heads.
   - Attention to masked positions is zeroed ($< 10^{-6}$).
   - Pre-softmax dot-product logits are explicitly scaled by $1 / \sqrt{d_k}$.
2. **Masking Properties**:
   - Padding masks broadcast cleanly over `(B, 1, 1, S)` and `(B, 1, T, T)`.
   - Causal look-ahead masks strictly mask future steps ($j > i$) with upper-triangular structure.
   - Combined decoder masks block both causal futures and padded query/key positions.
3. **Model & Optimization**:
   - Forward pass shapes match `(batch_size, tgt_seq_len, tgt_vocab_size)`.
   - Cross-entropy loss with label smoothing remains finite; token accuracy is bounded in $[0, 1]$.
   - Backward pass produces non-zero, finite gradients across all $88$ trainable variables.
   - Attention weight capture returns exact head and sequence dimensions for inspection.

---

## 3. Engineering Challenges & Root-Cause Resolutions

During the implementation, testing, and scaling of ScratchFormer, five critical bugs were diagnosed and resolved:

### 3.1 Decoder Post-Norm Residual Chaining
- **Issue:** In `src/model/decoder.py`, the output of the first residual block `res1(x, self_out)` was stored in `x`, but the cross-attention block mistakenly used the un-normalized input tensor.
- **Resolution:** Re-routed `res1` output directly into `cross_mha(x, enc_output, ...)`, ensuring post-layer normalization and residual signals propagate through the cross-attention sublayer as specified in Vaswani et al.

### 3.2 Combined Decoder Causal & Padding Mask Broadcast
- **Issue:** Padding tokens in the decoder input were receiving attention weights because `create_decoder_mask` only applied key padding masks alongside lookahead masks, leaving query padding positions unmasked.
- **Resolution:** Updated `create_decoder_mask` in `src/model/masks.py` to compute both `key_pad_mask` and `query_pad_mask` and merge them with `look_ahead_mask`, ensuring `<PAD>` positions receive full masking `[1, 1, 1, 1]`.

### 3.3 Target Sequence Token Alignment for Teacher Forcing
- **Issue:** In `src/data/dataset.py`, target sequences were encoded with `add_special=True` (`<SOS> + tokens + <EOS>`), and then teacher-forcing sliced `dec_in = tgt_ids[:, :-1]` and `labels = tgt_ids[:, 1:]`. This misaligned token boundaries with padding masks.
- **Resolution:** Fixed target encoding to store `tokens + <EOS>`, explicitly prefixing `<SOS>` for `dec_in` and aligning label sequences with exact sequence length.

### 3.4 TensorFlow / Keras 3 Unbuilt State Checkpoint Failure
- **Issue:** When running `evaluate.py` or `visualize.py`, `Trainer` called `self.ckpt.restore()` on an unbuilt subclassed model. Keras 3 emitted:
  `UserWarning: Layer 'scratchformer' looks like it has unbuilt state... Exception: 'EagerTensor' object has no attribute '_serialize_to_tensors'`
  TensorFlow silently failed to map checkpoint variables to layers created on subsequent calls, causing the model to instantiate brand-new random weights (`test_loss: 8.84`).
- **Resolution:** In `src/training/trainer.py`, `Trainer.__init__` now performs a lightweight dummy forward pass (`model((dummy_src, dummy_dec))`) if `len(model.trainable_variables) == 0` prior to constructing `tf.train.Checkpoint`. This guarantees all variables are allocated in memory before restoration.

### 3.5 Checkpoint Serialization of Sublayer Lists (`self.blocks`)
- **Issue:** Despite reaching 73.4% validation accuracy in training, evaluating restored checkpoints yielded random loss (`8.4966`). Deep inspection of the checkpoint index revealed that only $8$ out of $92$ tensors were saved! In TensorFlow Keras, assigning layers to a plain Python list (`self.blocks = [EncoderBlock(...) for ...]`) fails to register them in `_checkpoint_dependencies`. All 84 weight tensors for attention heads, feed-forwards, and layer norms were omitted from the checkpoint file on disk.
- **Resolution:** In both `Encoder` and `Decoder`, each block is explicitly assigned to the layer instance via `setattr(self, f"block_{i}", block)`. Re-testing with `tf.train.Checkpoint.restore().assert_consumed()` confirmed 100% of the 92 model tensors are serialized and restored.

### 3.6 Autoregressive Decoding Latency Optimization
- **Issue:** Initial greedy translation took ~4.98 seconds per sentence (~41 minutes for 500 sentences) due to Python-to-C++ dispatch and eager GPU kernel invocation for each generated token.
- **Resolution:** Decorated `_last_logits` in `src/inference/generate.py` with `@tf.function(reduce_retracing=True)`. Autoregressive decoding speed increased to **15.11 sentences/second** (500 sentences completed in **33.1 seconds**, a **~75× speedup**).

---

## 4. Baseline Training & Empirical Performance

The baseline model was trained for 10 epochs on 60,000 sentence pairs (48,000 train, 6,000 val, 6,000 test) using a single Kaggle Tesla T4 GPU.

### 4.1 Training Progression
- **Total Parameters:** 3,729,929
- **Training Time:** ~355 seconds total (~34s/epoch)
- **Batch Size:** 128 | **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$) with Transformer warmup schedule (4,000 steps).

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Epoch Time |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 7.6155 | 12.06% | 5.5706 | 22.92% | 51s |
| 2 | 4.6127 | 31.95% | 3.6541 | 40.93% | 34s |
| 3 | 3.4196 | 43.41% | 2.8854 | 49.13% | 34s |
| 4 | 2.8089 | 50.38% | 2.3561 | 56.58% | 34s |
| 5 | 2.3312 | 56.95% | 1.9483 | 62.96% | 34s |
| 6 | 1.9486 | 62.53% | 1.6617 | 67.46% | 34s |
| 7 | 1.6626 | 66.68% | 1.4898 | 70.23% | 34s |
| 8 | 1.4565 | 69.54% | 1.3757 | 71.56% | 34s |
| 9 | 1.3040 | 71.64% | 1.2950 | 72.87% | 34s |
| **10** | **1.1917** | **73.22%** | **1.2579** | **73.52%** | **34s** |

### 4.2 Held-Out Test Evaluation (N=500 Sentences)
- **Test Loss (Teacher-Forced):** `1.2484`
- **Test Token Accuracy (Teacher-Forced):** `73.54%`
- **Corpus BLEU (Greedy Autoregressive):** **36.55**
- **Mean Reference Length:** `6.44` tokens
- **Mean Hypothesis Length:** `8.16` tokens
- **Premature `<EOS>` Count:** `0 / 500` (zero premature termination)
- **Inference Speed:** **15.11 sent/s** (500 sentences in 33.1s)

---

## 5. Ablation Study & Scientific Findings

To systematically examine architectural design decisions, 13 variations plus 1 Keras native benchmark were evaluated under identical conditions (3 epochs each, shared vocabulary and tokenization):

| Experiment | Variation Details | Val EM (%) | Val BLEU | Train Time | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **baseline** | $d_m=128, h=8, L=2, d_{ff}=512, \text{PE}=\text{True}, w=4000$ | 0.0% | 6.50 | 125.1s | 3,729,929 |
| **no_positional_encoding**| Positional encoding removed ($\text{PE}=\text{False}$) | 0.0% | 8.15 | 122.2s | 3,729,929 |
| **heads_1** | Single attention head ($h=1$) | 0.0% | 7.40 | 114.1s | 3,729,929 |
| **heads_2** | Two attention heads ($h=2$) | 0.0% | 8.28 | 116.4s | 3,729,929 |
| **heads_4** | Four attention heads ($h=4$) | 0.0% | 7.46 | 116.6s | 3,729,929 |
| **d_model_64** | Narrower embedding width ($d_m=64$) | 0.0% | 2.22 | 92.4s | 1,771,721 |
| **d_model_256** | Wider embedding width ($d_m=256$) | 0.0% | 12.10 | 186.6s | 8,236,169 |
| **layers_1** | Single layer stack ($L=1$) | 0.0% | 7.99 | 86.8s | 3,267,081 |
| **layers_4** | Deeper layer stack ($L=4$) | 0.0% | 2.41 | 196.2s | 4,655,625 |
| **d_ff_256** | Narrower FFN capacity ($d_{ff}=256$) | 0.0% | 6.85 | 117.4s | 3,466,761 |
| **d_ff_1024** | Wider FFN capacity ($d_{ff}=1024$) | 0.0% | 7.41 | 135.0s | 4,256,265 |
| **warmup_1000** | Rapid learning rate ramp ($w=1000$) | 0.0% | **29.06** | 121.5s | 3,729,929 |
| **warmup_8000** | Slow learning rate ramp ($w=8000$) | 0.0% | 1.54 | 122.1s | 3,729,929 |
| **builtin_keras_mha** | Standard `tf.keras.layers.MultiHeadAttention` | 0.0% | 12.11 | 120.1s | 3,729,929 |

### Key Scientific Insights

1. **Learning Rate Warmup Dynamics (`warmup_1000` vs `warmup_8000`)**:
   - In a 3-epoch budget (approx. 1,125 total steps), `warmup_1000` achieved an impressive **29.06 BLEU** because the learning rate reached its peak quickly, enabling effective gradient updates.
   - Conversely, `warmup_8000` was still on the initial ramp ($\text{LR} \approx 10^{-4}$), yielding only **1.54 BLEU**.
2. **Model Width vs Depth (`d_model_256` vs `layers_4`)**:
   - Increasing width to $d_{\text{model}}=256$ delivered the highest BLEU among capacity modifications (**12.10 BLEU**), whereas increasing depth to $L=4$ under-performed (**2.41 BLEU**) within 3 epochs due to gradient propagation lag and cold warmup.
3. **Multi-Head Partitioning**:
   - At $d_m=128$, head configurations $h \in \{1, 2, 4, 8\}$ achieved competitive results ($7.40$ to $8.28$ BLEU), demonstrating that smaller hidden sizes benefit from moderate head dimensions ($d_k = 32$ or $64$).
4. **ScratchFormer vs Built-in Keras Benchmark**:
   - ScratchFormer trained in **125.1 seconds** compared to **120.1 seconds** for Keras' native `MultiHeadAttention`.
   - Our hand-crafted implementation with explicit splitting, matrix operations, and manual masking runs within **4% of native optimized C++ library routines** on GPU.

---

## 6. Reproducibility & Deployment Guide

To reproduce the exact findings:

1. **Local Verification (CPU)**:
   ```bash
   python -m pytest tests/ -v
   ```
2. **Kaggle GPU Execution (Tesla T4)**:
   ```bash
   git clone https://github.com/Ravikishore710/ScratchFormer.git
   cd ScratchFormer
   pip install -r requirements.txt

   # Full 10-epoch training (~5.5 minutes)
   python scripts/train.py --config configs/baseline.json --epochs 10 --skip-overfit

   # Held-out evaluation (500 sentences, ~33 seconds)
   python scripts/evaluate.py --config configs/baseline.json --n 500

   # Generate attention heatmaps & positional encoding visualizations
   python scripts/visualize.py --config configs/baseline.json --n 3

   # Run ablation suite & Keras benchmark (3 epochs each)
   python scripts/run_experiments.py --config configs/baseline.json --epochs 3
   ```

---

## 7. Conclusion

ScratchFormer validates that an industrial-strength Transformer encoder-decoder can be built from raw linear algebra and foundational operations in TensorFlow without sacrificing training stability, numerical precision, or execution throughput. The model achieves **36.55 BLEU** with zero premature degeneration and near-native GPU efficiency.
