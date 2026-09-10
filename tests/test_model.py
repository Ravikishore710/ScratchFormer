"""Full-model shape / loss / gradient / attention capture sanity checks."""
import numpy as np
import tensorflow as tf

from src.config import Config
from src.model.transformer import Transformer, count_parameters
from src.training.losses import masked_accuracy, masked_loss


def _test_config_and_model():
    cfg = Config(
        src_vocab_size=30,
        tgt_vocab_size=35,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        max_seq_len=8,
        dropout=0.0,
    )
    model = Transformer(cfg, cfg.src_vocab_size, cfg.tgt_vocab_size)
    return cfg, model


def test_transformer_forward_pass_shapes():
    cfg, model = _test_config_and_model()
    b, ls, lt = 2, 7, 5  # B > 1 and Ls != Lt
    src_ids = tf.random.uniform((b, ls), 0, cfg.src_vocab_size, dtype=tf.int64)
    dec_in_ids = tf.random.uniform((b, lt), 0, cfg.tgt_vocab_size, dtype=tf.int64)

    logits = model((src_ids, dec_in_ids), training=False)
    assert logits.shape == (b, lt, cfg.tgt_vocab_size)


def test_loss_is_finite_and_accuracy_is_bounded():
    cfg, _ = _test_config_and_model()
    b, lt = 2, 5
    # Include PAD tokens (0) to explicitly exercise padding mask paths
    tgt = tf.constant([
        [5, 8, 12, 0, 0],
        [9, 14, 0, 0, 0]
    ], dtype=tf.int64)
    logits = tf.random.normal((b, lt, cfg.tgt_vocab_size))

    loss = masked_loss(tgt, logits, pad_id=0)
    assert np.isfinite(float(loss))
    assert not np.isnan(float(loss))
    assert not np.isinf(float(loss))

    acc = masked_accuracy(tgt, logits, pad_id=0)
    acc_val = float(acc)
    assert 0.0 <= acc_val <= 1.0


def test_gradients_exist_and_are_finite():
    cfg, model = _test_config_and_model()
    b, ls, lt = 2, 6, 6
    src = tf.random.uniform((b, ls), 0, cfg.src_vocab_size, dtype=tf.int64)
    dec = tf.random.uniform((b, lt), 0, cfg.tgt_vocab_size, dtype=tf.int64)
    tgt = tf.random.uniform((b, lt), 0, cfg.tgt_vocab_size, dtype=tf.int64)

    with tf.GradientTape() as tape:
        logits = model((src, dec), training=True)
        loss = masked_loss(tgt, logits, pad_id=0)

    grads = tape.gradient(loss, model.trainable_variables)

    assert len(grads) == len(model.trainable_variables)
    for g, var in zip(grads, model.trainable_variables):
        assert g is not None, f"Variable {var.name} received no gradient."
        assert tf.reduce_all(tf.math.is_finite(g)).numpy(), f"Variable {var.name} gradient has non-finite values."


def test_attention_capture_shapes():
    cfg, model = _test_config_and_model()
    b, ls, lt = 2, 5, 6
    heads = cfg.num_heads

    src = tf.random.uniform((b, ls), 1, cfg.src_vocab_size, dtype=tf.int64)
    dec = tf.random.uniform((b, lt), 1, cfg.tgt_vocab_size, dtype=tf.int64)

    logits, attention = model((src, dec), training=False, return_attention=True)
    assert logits.shape == (b, lt, cfg.tgt_vocab_size)

    assert len(attention["encoder"]) == cfg.num_layers == 2
    assert len(attention["decoder_self"]) == cfg.num_layers == 2
    assert len(attention["cross"]) == cfg.num_layers == 2

    for l_idx in range(cfg.num_layers):
        assert attention["encoder"][l_idx].shape == (b, heads, ls, ls)
        assert attention["decoder_self"][l_idx].shape == (b, heads, lt, lt)
        assert attention["cross"][l_idx].shape == (b, heads, lt, ls)


def test_parameter_count():
    cfg, model = _test_config_and_model()
    counted = count_parameters(model)

    assert counted > 0
    expected = sum(int(tf.size(v).numpy()) for v in model.trainable_variables)
    assert counted == expected
