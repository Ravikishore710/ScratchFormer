"""The complete from-scratch encoder-decoder Transformer."""
from __future__ import annotations

import tensorflow as tf

from ..config import Config
from .decoder import Decoder
from .embedding import TokenAndPositionEmbedding
from .encoder import Encoder
from .masks import create_decoder_mask, create_padding_mask


class Transformer(tf.keras.Model):
    """ScratchFormer: encoder-decoder Transformer implemented from scratch.

    call(inputs=(src_ids, dec_in_ids), training=..., return_attention=...)
        -> logits (B, Lt, tgt_vocab)  [+ attention dict if requested]

    Masks are created inside the model from the raw token ids (<PAD> id is 0,
    guaranteed by WordTokenizer). Teacher forcing is expressed by passing the
    shifted-right target as dec_in_ids.
    """

    def __init__(self, cfg: Config, src_vocab_size: int, tgt_vocab_size: int,
                 name: str = "scratchformer"):
        super().__init__(name=name)
        cfg.validate()
        self.cfg = cfg
        self.src_embed = TokenAndPositionEmbedding(
            src_vocab_size, cfg.d_model, cfg.max_seq_len, cfg.dropout,
            cfg.use_positional_encoding, name="src_embedding")
        self.tgt_embed = TokenAndPositionEmbedding(
            tgt_vocab_size, cfg.d_model, cfg.max_seq_len, cfg.dropout,
            cfg.use_positional_encoding, name="tgt_embedding")
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.final_layer = tf.keras.layers.Dense(tgt_vocab_size, name="final_projection")

    def call(self, inputs, training=False, return_attention: bool = False):
        src_ids, dec_in_ids = inputs

        enc_padding_mask = create_padding_mask(src_ids)       # (B,1,1,Ls)
        combined_mask = create_decoder_mask(dec_in_ids)       # (B,1,Lt,Lt)

        enc_in = self.src_embed(src_ids, training=training)
        if return_attention:
            enc_out, enc_attn = self.encoder(enc_in, enc_padding_mask,
                                             training=training, return_attention=True)
        else:
            enc_out = self.encoder(enc_in, enc_padding_mask, training=training)
            enc_attn = None

        dec_in = self.tgt_embed(dec_in_ids, training=training)
        if return_attention:
            dec_out, dec_self, dec_cross = self.decoder(
                dec_in, enc_out, combined_mask, enc_padding_mask,
                training=training, return_attention=True)
        else:
            dec_out = self.decoder(dec_in, enc_out, combined_mask,
                                   enc_padding_mask, training=training)
            dec_self = dec_cross = None

        logits = self.final_layer(dec_out)
        if return_attention:
            return logits, {
                "encoder": enc_attn,
                "decoder_self": dec_self,
                "cross": dec_cross,
            }
        return logits


def count_parameters(model: tf.keras.Model) -> int:
    """Parameter count (forces one dummy forward pass to build weights)."""
    model((tf.zeros((1, 8), tf.int64), tf.zeros((1, 8), tf.int64)), training=False)
    return int(sum(int(tf.size(vv).numpy()) for vv in model.trainable_variables))
