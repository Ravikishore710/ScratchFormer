"""Transformer decoder: masked self-attention + cross-attention + FFN blocks."""
from __future__ import annotations

import tensorflow as tf

from ..config import Config
from .layers import MultiHeadAttention, PointWiseFFN, ResidualLayerNorm


class DecoderBlock(tf.keras.layers.Layer):
    """Masked self-attn -> Add&Norm -> cross-attn -> Add&Norm -> FFN -> Add&Norm.

    Self-attention:  Q=K=V=decoder state   (causal + padding mask)
    Cross-attention: Q=decoder state, K=V=encoder output  (encoder padding mask)
    """

    def __init__(self, cfg: Config, name: str = "decoder_block"):
        super().__init__(name=name)
        self.self_mha = MultiHeadAttention(cfg.d_model, cfg.num_heads, name="masked_self_attention")
        self.cross_mha = MultiHeadAttention(cfg.d_model, cfg.num_heads, name="cross_attention")
        self.ffn = PointWiseFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.res1 = ResidualLayerNorm(cfg.d_model, cfg.dropout, name="residual_1")
        self.res2 = ResidualLayerNorm(cfg.d_model, cfg.dropout, name="residual_2")
        self.res3 = ResidualLayerNorm(cfg.d_model, cfg.dropout, name="residual_3")

    def call(self, x, enc_output, combined_mask=None, enc_padding_mask=None,
             training=None, return_attention: bool = False):
        # 1) masked self-attention over the decoder state so far
        if return_attention:
            self_out, self_w = self.self_mha(x, x, x, combined_mask, return_attention=True)
            x = self.res1(x, self_out, training=training)
            cross_out, cross_w = self.cross_mha(x, enc_output, enc_output,
                                                enc_padding_mask, return_attention=True)
            x = self.res2(x, cross_out, training=training)
        else:
            self_out = self.self_mha(x, x, x, combined_mask)
            x = self.res1(x, self_out, training=training)
            cross_out = self.cross_mha(x, enc_output, enc_output, enc_padding_mask)
            x = self.res2(x, cross_out, training=training)
            self_w = cross_w = None
        x = self.res3(x, self.ffn(x, training=training), training=training)
        if return_attention:
            return x, self_w, cross_w
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, cfg: Config, name: str = "decoder"):
        super().__init__(name=name)
        self.blocks = [DecoderBlock(cfg, name=f"block_{i}") for i in range(cfg.num_layers)]
        self.dropout = tf.keras.layers.Dropout(cfg.dropout)

    def call(self, x, enc_output, combined_mask=None, enc_padding_mask=None,
             training=None, return_attention: bool = False):
        self_attentions, cross_attentions = [], []
        x = self.dropout(x, training=training)
        for block in self.blocks:
            if return_attention:
                x, sw, cw = block(x, enc_output, combined_mask, enc_padding_mask,
                                  training=training, return_attention=True)
                self_attentions.append(sw)
                cross_attentions.append(cw)
            else:
                x = block(x, enc_output, combined_mask, enc_padding_mask, training=training)
        if return_attention:
            return x, self_attentions, cross_attentions
        return x
