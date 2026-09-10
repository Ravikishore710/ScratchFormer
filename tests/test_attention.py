"""Scaled dot-product attention correctness and invariant checks."""
import numpy as np
import tensorflow as tf

from src.model.attention import scaled_dot_product_attention


def test_attention_output_and_weight_shapes():
    b = 2
    h = 4
    lq = 3
    lk = 5
    depth = 8

    q = tf.random.normal((b, h, lq, depth))
    k = tf.random.normal((b, h, lk, depth))
    v = tf.random.normal((b, h, lk, depth))

    out, weights = scaled_dot_product_attention(q, k, v)

    assert out.shape == (b, h, lq, depth)
    assert weights.shape == (b, h, lq, lk)


def test_softmax_normalization_and_masking():
    b, h, lq, lk, depth = 2, 2, 3, 4, 8
    q = tf.random.normal((b, h, lq, depth))
    k = tf.random.normal((b, h, lk, depth))
    v = tf.random.normal((b, h, lk, depth))

    # Without mask: all rows sum to 1.0
    _, w_unmasked = scaled_dot_product_attention(q, k, v)
    row_sums = tf.reduce_sum(w_unmasked, axis=-1)
    np.testing.assert_allclose(row_sums.numpy(), 1.0, atol=1e-5)

    # Mask key position 2 (0=allowed, 1=blocked)
    mask = tf.constant([[[[0.0, 0.0, 1.0, 0.0]]]])  # (1, 1, 1, 4) broadcastable
    out_masked, w_masked = scaled_dot_product_attention(q, k, v, mask=mask)

    # 1. Blocked key position receives ~0 attention
    assert float(tf.reduce_max(w_masked[..., 2])) < 1e-6
    # 2. Allowed positions renormalize and sum to 1.0
    np.testing.assert_allclose(tf.reduce_sum(w_masked, axis=-1).numpy(), 1.0, atol=1e-5)
    # 3. Allowed positions alone sum to 1.0
    allowed_sum = w_masked[..., 0] + w_masked[..., 1] + w_masked[..., 3]
    np.testing.assert_allclose(allowed_sum.numpy(), 1.0, atol=1e-5)
    # 4. No NaNs produced
    assert not np.isnan(w_masked.numpy()).any()
    assert not np.isnan(out_masked.numpy()).any()


def test_scaling_by_sqrt_d_k():
    # Small deterministic values for exact mathematical verification
    q_np = np.array([[[[1.0, 2.0],
                       [3.0, 4.0]]]], dtype=np.float32)  # (1, 1, 2, 2)
    k_np = np.array([[[[1.0, 0.0],
                       [0.0, 2.0]]]], dtype=np.float32)  # (1, 1, 2, 2)
    v_np = np.array([[[[5.0, 6.0],
                       [7.0, 8.0]]]], dtype=np.float32)  # (1, 1, 2, 2)

    d_k = 2.0  # key depth

    # Manual computation using NumPy: scores = Q K^T / sqrt(d_k)
    scores_np = np.matmul(q_np, np.swapaxes(k_np, -1, -2)) / np.sqrt(d_k)
    # Softmax over last axis
    exp_scores = np.exp(scores_np - np.max(scores_np, axis=-1, keepdims=True))
    expected_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    expected_out = np.matmul(expected_weights, v_np)

    q = tf.constant(q_np)
    k = tf.constant(k_np)
    v = tf.constant(v_np)
    actual_out, actual_weights = scaled_dot_product_attention(q, k, v)

    np.testing.assert_allclose(actual_weights.numpy(), expected_weights, atol=1e-5)
    np.testing.assert_allclose(actual_out.numpy(), expected_out, atol=1e-5)
