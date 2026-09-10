"""Transformer encoder: a stack of identical encoder blocks."""
from __future__ import annotations

import tensorflow as tf

from ..config import Config
from .layers import MultiHeadAttention, PointWiseFFN, ResidualLayerNorm


class EncoderBlock(tf.keras.layers.Layer):
    """Self-attention -> Add&Norm -> FFN -> Add&Norm (post-norm, Vaswani et al.)."""

    def __init__(self, cfg: Config, name: str = "encoder_block"):
        super().__init__(name=name)
        self.mha = MultiHeadAttention(cfg.d_model, cfg.num_heads)
        self.ffn = PointWiseFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.res1 = ResidualLayerNorm(cfg.d_model, cfg.dropout, name="residual_1")
        self.res2 = ResidualLayerNorm(cfg.d_model, cfg.dropout, name="residual_2")

    def call(self, x, mask=None, training=None, return_attention: bool = False):
        if return_attention:
            attn_out, attn_weights = self.mha(x, x, x, mask, return_attention=True)
        else:
            attn_out, attn_weights = self.mha(x, x, x, mask), None
        x = self.res1(x, attn_out, training=training)
        x = self.res2(x, self.ffn(x, training=training), training=training)
        if return_attention:
            return x, attn_weights
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, cfg: Config, name: str = "encoder"):
        super().__init__(name=name)
        self.blocks = [EncoderBlock(cfg, name=f"block_{i}") for i in range(cfg.num_layers)]
        self.dropout = tf.keras.layers.Dropout(cfg.dropout)

    def call(self, x, mask=None, training=None, return_attention: bool = False):
        attentions = []
        x = self.dropout(x, training=training)
        for block in self.blocks:
            if return_attention:
                x, w = block(x, mask, training=training, return_attention=True)
                attentions.append(w)
            else:
                x = block(x, mask, training=training)
        if return_attention:
            return x, attentions
        return x
