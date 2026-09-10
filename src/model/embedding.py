"""Token embedding + sinusoidal positional encoding."""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def sinusoidal_position_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Classic Vaswani et al. (2017) sinusoidal positional encoding matrix."""
    pos = np.arange(max_len)[:, np.newaxis]              # (max_len, 1)
    i = np.arange(d_model // 2)[np.newaxis, :]           # (1, d_model/2)
    angle_rates = 1.0 / np.power(10000.0, (2 * i) / np.float32(d_model))
    angles = pos * angle_rates                           # (max_len, d_model/2)
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """Embedding * sqrt(d_model) + positional encoding (+ dropout).

    `use_positional_encoding=False` is supported for the PE ablation study.
    The positional matrix is a non-trainable weight so it serializes cleanly.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int,
                 dropout: float, use_positional_encoding: bool = True,
                 name: str = "token_position_embedding"):
        super().__init__(name=name)
        self.d_model = tf.cast(d_model, tf.float32)
        self.use_pe = use_positional_encoding
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, name="token_embedding")
        self.dropout = tf.keras.layers.Dropout(dropout)
        pe = sinusoidal_position_encoding(max_seq_len, d_model)[np.newaxis, ...]
        self.pos_encoding = self.add_weight(
            shape=pe.shape, initializer=tf.constant_initializer(pe),
            trainable=False, name="positional_encoding")

    def call(self, x, training=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) * tf.sqrt(self.d_model)
        if self.use_pe:
            x = x + self.pos_encoding[:, :seq_len, :]
        return self.dropout(x, training=training)
