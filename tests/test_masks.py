"""Mask semantics: 1 = blocked, 0 = allowed."""
import tensorflow as tf

from src.model.masks import (create_decoder_mask, create_look_ahead_mask,
                             create_padding_mask)


def test_padding_mask_shape_and_values():
    seq = tf.constant([[5, 0, 7, 0]])
    m = create_padding_mask(seq)
    assert m.shape == (1, 1, 1, 4)
    assert m[0, 0, 0, 0] == 0 and m[0, 0, 0, 1] == 1 and m[0, 0, 0, 3] == 1


def test_look_ahead_mask_is_causal():
    m = create_look_ahead_mask(4)
    expected = tf.constant([[0., 1, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]])
    assert tf.reduce_all(tf.equal(m, expected))


def test_decoder_mask_combines_padding_and_causality():
    tgt = tf.constant([[1, 2, 0, 0]])  # last two positions are <PAD>
    m = create_decoder_mask(tgt)
    assert m.shape == (1, 1, 4, 4)
    # query 0 may attend only to itself
    assert m[0, 0, 0, 0] == 0 and m[0, 0, 0, 1] == 1
    # query 1 may attend to keys 0 and 1, never to PAD keys 2/3
    assert m[0, 0, 1, 0] == 0 and m[0, 0, 1, 1] == 0 and m[0, 0, 1, 2] == 1
