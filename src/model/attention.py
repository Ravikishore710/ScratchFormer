"""Scaled dot-product attention implemented from first principles.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Everything is explicit: we build Q, K, V projections elsewhere, and here we
compute scores, scale, mask, softmax, and the weighted sum of V -- returning
both the output and the attention weights so they can be visualized later.
"""
from __future__ import annotations

import tensorflow as tf


def scaled_dot_product_attention(
    q: tf.Tensor,
    k: tf.Tensor,
    v: tf.Tensor,
    mask: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Args:
        q: (B, H, Lq, depth) queries
        k: (B, H, Lk, depth) keys
        v: (B, H, Lk, dv)  values
        mask: broadcastable to (B, H, Lq, Lk); 1 = blocked, 0 = allowed
    Returns:
        output: (B, H, Lq, dv)
        attention_weights: (B, H, Lq, Lk) -- rows sum to 1
    """
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(d_k)

    if mask is not None:
        scores += (mask * -1e9)

    attention_weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights
