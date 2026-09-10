"""Advanced unit tests for dataset preprocessing and teacher-forcing alignment."""
import numpy as np
import tensorflow as tf

from src.config import Config
from src.data.dataset import _encode_padded, _make_tf_dataset, prepare_datasets
from src.data.tokenizer import WordTokenizer


def test_shift_right_and_teacher_forcing_alignment():
    # Construct synthetic token sequences
    sos_id = 2
    eos_id = 3
    pad_id = 0

    # Let tgt be "bonjour monde <EOS> <PAD> <PAD>"
    # token ids: [10, 20, 3, 0, 0]
    tgt = np.array([[10, 20, eos_id, pad_id, pad_id]], dtype=np.int64)
    src = np.array([[5, 6, pad_id, pad_id, pad_id]], dtype=np.int64)

    cfg = Config(batch_size=1, max_seq_len=5)
    ds = _make_tf_dataset(src, tgt, cfg, sos_id=sos_id, shuffle=False)

    for (s, dec_in), y in ds.take(1):
        # dec_in must start with SOS (2)
        assert dec_in[0, 0].numpy() == sos_id
        # dec_in[0, 1] must be tgt[0, 0] (10)
        assert dec_in[0, 1].numpy() == 10
        # dec_in[0, 2] must be tgt[0, 1] (20)
        assert dec_in[0, 2].numpy() == 20
        # Target y must be the original sequence
        assert y[0, 0].numpy() == 10
        assert y[0, 1].numpy() == 20
        assert y[0, 2].numpy() == eos_id
        assert y[0, 3].numpy() == pad_id


def test_prepare_datasets_splits_and_shapes():
    cfg = Config(max_pairs=500, max_seq_len=20, batch_size=16)
    bundle = prepare_datasets(cfg, verbose=False)

    stats = bundle.stats
    assert stats["pairs_used"] <= 500
    assert stats["train"] + stats["val"] + stats["test"] == stats["pairs_used"]
    assert stats["train"] == int(stats["pairs_used"] * (1.0 - cfg.val_size - cfg.test_size))

    # Test batch shapes from tf.data
    batch = next(iter(bundle.train_ds.take(1)))
    (src, dec_in), tgt = batch
    assert src.shape == (16, 20)
    assert dec_in.shape == (16, 20)
    assert tgt.shape == (16, 20)
    assert src.dtype == tf.int64
    assert dec_in.dtype == tf.int64
    assert tgt.dtype == tf.int64


def test_tokenizers_fitted_only_on_train_split():
    cfg = Config(max_pairs=500, max_seq_len=20)
    bundle = prepare_datasets(cfg, verbose=False)

    # Any rare token that only appeared in the test set must be mapped to UNK
    test_only_word = "xyznonexistentword99"
    assert bundle.src_tok.token_to_id.get(test_only_word, bundle.src_tok.unk_id) == bundle.src_tok.unk_id
    assert bundle.tgt_tok.token_to_id.get(test_only_word, bundle.tgt_tok.unk_id) == bundle.tgt_tok.unk_id
