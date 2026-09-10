"""Padding-aware loss and accuracy. <PAD> positions never contribute."""
from __future__ import annotations

import tensorflow as tf


def masked_loss(real: tf.Tensor, pred: tf.Tensor, pad_id: int = 0,
                label_smoothing: float = 0.0) -> tf.Tensor:
    """Token-level cross-entropy over non-PAD positions only."""
    mask = tf.cast(tf.not_equal(real, pad_id), tf.float32)
    if label_smoothing and label_smoothing > 0.0:
        vocab_size = tf.cast(tf.shape(pred)[-1], tf.float32)
        one_hot = tf.one_hot(real, tf.shape(pred)[-1], dtype=tf.float32)
        one_hot = one_hot * (1.0 - label_smoothing) + label_smoothing / vocab_size
        log_probs = tf.nn.log_softmax(pred, axis=-1)
        loss = -tf.reduce_sum(one_hot * log_probs, axis=-1)
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(real: tf.Tensor, pred: tf.Tensor, pad_id: int = 0) -> tf.Tensor:
    """Token accuracy over non-PAD positions only."""
    mask = tf.cast(tf.not_equal(real, pad_id), tf.float32)
    correct = tf.cast(tf.equal(real, tf.argmax(pred, axis=-1, output_type=real.dtype)),
                      tf.float32) * mask
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)
