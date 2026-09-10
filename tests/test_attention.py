"""Scaled dot-product attention correctness checks."""
import numpy as np
import tensorflow as tf

from src.model.attention import scaled_dot_product_attention


def test_output_shape_and_softmax_rows():
    q = tf.random.normal((2, 3, 5, 8))
    k = tf.random.normal((2, 3, 6, 8))
    v = tf.random.normal((2, 3, 6, 16))
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 3, 5, 16)
    assert w.shape == (2, 3, 5, 6)
    np.testing.assert_allclose(w.numpy().sum(-1), 1.0, atol=1e-5)


def test_mask_blocks_key_positions():
    q = tf.random.normal((1, 1, 2, 4))
    k = tf.random.normal((1, 1, 3, 4))
    v = tf.random.normal((1, 1, 3, 4))
    mask = tf.constant([[[[0.0, 0.0, 1.0]]]])  # block key 2
    _, w = scaled_dot_product_attention(q, k, v, mask)
    assert float(tf.reduce_max(w[..., 2])) < 1e-6
    np.testing.assert_allclose(w.numpy()[..., :2].sum(-1), 1.0, atol=1e-5)


def test_scaling_prevents_explosion():
    # without 1/sqrt(d_k) the logits would be huge; softmax saturates and
    # gradients vanish -- verify weights stay well-conditioned
    q = k = tf.random.normal((1, 1, 10, 64))
    v = tf.random.normal((1, 1, 10, 64))
    _, w = scaled_dot_product_attention(q, k, v)
    assert float(tf.reduce_max(w)) < 1.0
    assert float(tf.reduce_min(w)) > 0.0
