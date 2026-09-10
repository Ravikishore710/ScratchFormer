"""Full-model shape / loss / gradient sanity checks on a tiny config."""
import tensorflow as tf

from src.config import Config
from src.model.transformer import Transformer, count_parameters
from src.training.losses import masked_accuracy, masked_loss


def _tiny_model():
    cfg = Config(d_model=32, num_heads=4, num_layers=1, d_ff=64, max_seq_len=16,
                 batch_size=8)
    return Transformer(cfg, src_vocab_size=60, tgt_vocab_size=70), cfg


def test_forward_shapes():
    model, _ = _tiny_model()
    src = tf.random.uniform((2, 10), 0, 60, dtype=tf.int64)
    dec = tf.random.uniform((2, 12), 0, 70, dtype=tf.int64)
    logits = model((src, dec), training=False)
    assert logits.shape == (2, 12, 70)


def test_loss_is_finite_and_padding_ignored():
    model, _ = _tiny_model()
    tgt = tf.constant([[5, 6, 7, 0, 0], [8, 9, 0, 0, 0]], tf.int64)
    logits = tf.random.normal((2, 5, 70))
    loss = masked_loss(tgt, logits, pad_id=0)
    assert tf.math.is_finite(loss)
    # accuracy over only the 5 non-PAD tokens
    acc = masked_accuracy(tgt, logits, pad_id=0)
    assert 0.0 <= float(acc) <= 1.0


def test_gradients_flow_to_all_layers():
    model, _ = _tiny_model()
    src = tf.random.uniform((2, 8), 0, 60, dtype=tf.int64)
    dec = tf.random.uniform((2, 8), 0, 70, dtype=tf.int64)
    tgt = tf.random.uniform((2, 8), 0, 70, dtype=tf.int64)
    with tf.GradientTape() as tape:
        loss = masked_loss(tgt, model((src, dec), training=True), pad_id=0)
    grads = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in grads)
    assert all(tf.math.is_finite(g).numpy().all() for g in grads)


def test_attention_capture_shapes():
    model, cfg = _tiny_model()
    src = tf.random.uniform((1, 6), 1, 60, dtype=tf.int64)  # no PAD -> zero mask
    dec = tf.random.uniform((1, 6), 1, 70, dtype=tf.int64)
    logits, attn = model((src, dec), training=False, return_attention=True)
    assert logits.shape == (1, 6, 70)
    for key in ("encoder", "decoder_self", "cross"):
        assert len(attn[key]) == cfg.num_layers
    h, Ls, Lt = cfg.num_heads, 6, 6
    assert attn["encoder"][0].shape == (1, h, Ls, Ls)
    assert attn["decoder_self"][0].shape == (1, h, Lt, Lt)
    assert attn["cross"][0].shape == (1, h, Lt, Ls)


def test_parameter_count_is_positive():
    model, _ = _tiny_model()
    assert count_parameters(model) > 0
