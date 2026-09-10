"""Unit tests for individual Transformer layers, positional encoding, and BuiltinTransformer."""
import numpy as np
import tensorflow as tf

from src.config import Config
from src.model.builtin import BuiltinTransformer
from src.model.embedding import TokenAndPositionEmbedding, sinusoidal_position_encoding
from src.model.layers import PointWiseFFN, ResidualLayerNorm
from src.model.transformer import Transformer, count_parameters


def test_sinusoidal_positional_encoding():
    max_len = 50
    d_model = 64
    pe = sinusoidal_position_encoding(max_len, d_model)
    assert pe.shape == (max_len, d_model)
    # Sinusoids are bounded within [-1, 1]
    assert np.all(pe >= -1.0)
    assert np.all(pe <= 1.0)
    # Even dimensions are sine, odd dimensions are cosine
    assert np.allclose(pe[0, 1::2], 1.0)  # cos(0) = 1


def test_pointwise_ffn():
    d_model = 32
    d_ff = 128
    ffn = PointWiseFFN(d_model, d_ff, dropout=0.0)
    x = tf.random.normal((2, 10, d_model))
    out = ffn(x, training=False)
    assert out.shape == (2, 10, d_model)


def test_residual_layernorm():
    d_model = 32
    norm_layer = ResidualLayerNorm(d_model, dropout=0.0, name="res_norm")
    x = tf.random.normal((2, 8, d_model))
    sub_out = tf.random.normal((2, 8, d_model))
    out = norm_layer(x, sub_out, training=False)
    assert out.shape == (2, 8, d_model)
    # Output of LayerNorm along last axis should have zero mean and unit variance
    mean = tf.reduce_mean(out, axis=-1)
    np.testing.assert_allclose(mean.numpy(), 0.0, atol=1e-5)


def test_builtin_transformer_shape_parity():
    cfg = Config(d_model=32, num_heads=4, num_layers=1, d_ff=64, max_seq_len=16)
    src_vocab = 50
    tgt_vocab = 60

    custom_model = Transformer(cfg, src_vocab, tgt_vocab)
    builtin_model = BuiltinTransformer(cfg, src_vocab, tgt_vocab)

    src = tf.random.uniform((2, 16), 0, src_vocab, dtype=tf.int64)
    dec = tf.random.uniform((2, 16), 0, tgt_vocab, dtype=tf.int64)

    out_custom = custom_model((src, dec), training=False)
    out_builtin = builtin_model((src, dec), training=False)

    assert out_custom.shape == (2, 16, tgt_vocab)
    assert out_builtin.shape == (2, 16, tgt_vocab)
    assert count_parameters(custom_model) > 0
    assert count_parameters(builtin_model) > 0
