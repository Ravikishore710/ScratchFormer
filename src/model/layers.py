"""Core Transformer layers: Multi-Head Attention, FFN, and residual/LayerNorm
building blocks. No tf.keras.layers.MultiHeadAttention is used anywhere here.
"""
from __future__ import annotations

import tensorflow as tf

from .attention import scaled_dot_product_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention built on our own scaled dot-product attention.

    d_model = num_heads * head_dim is enforced at construction time.
    """

    def __init__(self, d_model: int, num_heads: int, name: str = "multi_head_attention"):
        super().__init__(name=name)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model, name="wq")
        self.wk = tf.keras.layers.Dense(d_model, name="wk")
        self.wv = tf.keras.layers.Dense(d_model, name="wv")
        self.wo = tf.keras.layers.Dense(d_model, name="wo")

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """(B, L, d_model) -> (B, num_heads, L, depth)."""
        b = tf.shape(x)[0]
        x = tf.reshape(x, (b, tf.shape(x)[1], self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _merge_heads(self, x: tf.Tensor) -> tf.Tensor:
        """(B, num_heads, L, depth) -> (B, L, d_model)."""
        b = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (b, tf.shape(x)[1], self.d_model))

    def call(self, q, k, v, mask=None, return_attention: bool = False, training=None):
        q = self._split_heads(self.wq(q))
        k = self._split_heads(self.wk(k))
        v = self._split_heads(self.wv(v))

        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        concat = self._merge_heads(attn_output)
        output = self.wo(concat)
        if return_attention:
            return output, attn_weights
        return output


class PointWiseFFN(tf.keras.layers.Layer):
    """Position-wise feed-forward network: Dense(d_ff) -> activation -> Dense(d_model)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, name: str = "ffn"):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(d_ff, activation="relu", name="dense_1")
        self.dense2 = tf.keras.layers.Dense(d_model, name="dense_2")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=None):
        return self.dense2(self.dropout(self.dense1(x), training=training))


class ResidualLayerNorm(tf.keras.layers.Layer):
    """out = LayerNorm(x + Dropout(sublayer(x)))  -- the post-norm residual block."""

    def __init__(self, d_model: int, dropout: float, name: str):
        super().__init__(name=name)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, sublayer_output, training=None):
        return self.norm(x + self.dropout(sublayer_output, training=training))
