"""Mask semantics: 1 = blocked, 0 = allowed."""
import numpy as np
import tensorflow as tf

from src.model.masks import (create_decoder_mask, create_look_ahead_mask,
                             create_padding_mask)


def test_padding_mask_shape_and_values():
    seq = tf.constant([
        [5, 8, 0, 0],
        [7, 3, 9, 0]
    ])
    m = create_padding_mask(seq, pad_id=0)

    # 1. Output shape must be (B, 1, 1, L)
    assert m.shape == (2, 1, 1, 4)

    # 2. Non-padding tokens produce 0, padding tokens produce 1
    expected_sample_1 = np.array([[[[0.0, 0.0, 1.0, 1.0]]]])
    expected_sample_2 = np.array([[[[0.0, 0.0, 0.0, 1.0]]]])
    np.testing.assert_array_equal(m[0:1].numpy(), expected_sample_1)
    np.testing.assert_array_equal(m[1:2].numpy(), expected_sample_2)


def test_look_ahead_mask_is_causal():
    size = 4
    m = create_look_ahead_mask(size)
    expected = tf.constant([
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
    ])

    assert m.shape == (4, 4)
    # Diagonal is allowed (0), previous allowed (0), future blocked (1)
    assert tf.reduce_all(tf.equal(m, expected))


def test_decoder_mask_combines_padding_and_causality():
    dec_ids = tf.constant([[2, 5, 8, 0]])
    m = create_decoder_mask(dec_ids, pad_id=0)

    # Shape: (B, 1, L, L)
    assert m.shape == (1, 1, 4, 4)

    expected = np.array([[
        [
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ]
    ]])
    np.testing.assert_array_equal(m.numpy(), expected)

    # Output contains only 0 or 1 (combined via maximum, not addition)
    unique_vals = set(np.unique(m.numpy()))
    assert unique_vals.issubset({0.0, 1.0})
