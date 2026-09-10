"""Benchmark model built from Keras' ready-made MultiHeadAttention.

Architecture is kept identical to ScratchFormer (same depths, heads, layers,
FFN sizes and positional encoding) so the comparison is apples-to-apples.
This is ONLY a benchmark, not the project itself.
"""
from __future__ import annotations

import tensorflow as tf

from ..config import Config
from .embedding import TokenAndPositionEmbedding
from .masks import create_decoder_mask, create_padding_mask


def _keras_mask(mask: tf.Tensor) -> tf.Tensor:
    """Our masks: 1=blocked. Keras MHA boolean masks: True=attend."""
    return tf.logical_not(tf.cast(mask, tf.bool))


class _BuiltinBlock(tf.keras.layers.Layer):
    def __init__(self, cfg: Config, is_decoder: bool, name: str):
        super().__init__(name=name)
        depth = cfg.d_model // cfg.num_heads
        self.self_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=cfg.num_heads, key_dim=depth, dropout=cfg.dropout, name="self_mha")
        self.cross_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=cfg.num_heads, key_dim=depth, dropout=cfg.dropout,
            name="cross_mha") if is_decoder else None
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(cfg.d_ff, activation="relu"),
            tf.keras.layers.Dense(cfg.d_model),
        ], name="ffn")
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(cfg.dropout)

    def call(self, x, enc_output=None, self_mask=None, cross_mask=None, training=None):
        x = self.norm1(x + self.dropout(
            self.self_mha(x, x, attention_mask=_keras_mask(self_mask),
                          training=training), training=training))
        if self.cross_mha is not None:
            x = self.norm2(x + self.dropout(
                self.cross_mha(x, enc_output, attention_mask=_keras_mask(cross_mask),
                               training=training), training=training))
        x = (self.norm3 if self.cross_mha is not None else self.norm2)(
            x + self.dropout(self.ffn(x, training=training), training=training))
        return x


class BuiltinTransformer(tf.keras.Model):
    """Drop-in replacement for ScratchFormer with the same call signature."""

    def __init__(self, cfg: Config, src_vocab_size: int, tgt_vocab_size: int,
                 name: str = "builtin_transformer"):
        super().__init__(name=name)
        cfg.validate()
        self.cfg = cfg
        self.src_embed = TokenAndPositionEmbedding(
            src_vocab_size, cfg.d_model, cfg.max_seq_len, cfg.dropout,
            cfg.use_positional_encoding, name="src_embedding")
        self.tgt_embed = TokenAndPositionEmbedding(
            tgt_vocab_size, cfg.d_model, cfg.max_seq_len, cfg.dropout,
            cfg.use_positional_encoding, name="tgt_embedding")
        self.enc_blocks = [_BuiltinBlock(cfg, False, f"enc_block_{i}") for i in range(cfg.num_layers)]
        self.dec_blocks = [_BuiltinBlock(cfg, True, f"dec_block_{i}") for i in range(cfg.num_layers)]
        self.final_layer = tf.keras.layers.Dense(tgt_vocab_size)

    def call(self, inputs, training=False, return_attention: bool = False):
        if return_attention:
            raise NotImplementedError("Benchmark model does not expose attention weights.")
        src_ids, dec_in_ids = inputs
        enc_mask = create_padding_mask(src_ids)
        combined_mask = create_decoder_mask(dec_in_ids)
        x = self.src_embed(src_ids, training=training)
        for block in self.enc_blocks:
            x = block(x, self_mask=enc_mask, training=training)
        enc_out = x
        y = self.tgt_embed(dec_in_ids, training=training)
        for block in self.dec_blocks:
            y = block(y, enc_output=enc_out, self_mask=combined_mask,
                      cross_mask=enc_mask, training=training)
        return self.final_layer(y)
