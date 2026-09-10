"""Attention masking utilities.

Convention: a mask value of 1.0 means "blocked" (set to -inf before softmax),
0.0 means "allowed". Masks are float tensors so they can be added directly to
attention logits, and they broadcast over (batch, heads, query_len, key_len).
"""
from __future__ import annotations

import tensorflow as tf


def create_padding_mask(seq: tf.Tensor, pad_id: int = 0) -> tf.Tensor:
    """(B, L) -> (B, 1, 1, L); 1 where the token is <PAD>."""
    mask = tf.cast(tf.math.equal(seq, pad_id), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size: int) -> tf.Tensor:
    """(size, size) strictly-lower-triangular allow mask; 1 = future position."""
    return 1.0 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def create_decoder_mask(dec_ids: tf.Tensor, pad_id: int = 0) -> tf.Tensor:
    """Combine padding + causality for decoder self-attention.

    (B, L) -> (B, 1, L, L). A query position can never attend to a <PAD> key
    nor to any future key.
    """
    pad_mask = create_padding_mask(dec_ids, pad_id)          # (B, 1, 1, L)
    look_ahead = create_look_ahead_mask(tf.shape(dec_ids)[1])  # (L, L)
    return tf.maximum(pad_mask, look_ahead[tf.newaxis, tf.newaxis])
