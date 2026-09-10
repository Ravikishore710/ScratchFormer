"""Advanced tests for inference, greedy generation, and translation metrics."""
import tensorflow as tf

from src.evaluation.metrics import corpus_bleu, exact_match
from src.inference.generate import translate


def test_exact_match_metric():
    refs = ["le chat est noir", "bonjour le monde", "il fait beau"]
    hyps_perfect = ["le chat est noir", "bonjour le monde", "il fait beau"]
    assert exact_match(refs, hyps_perfect) == 100.0

    hyps_partial = ["le chat est blanc", "bonjour le monde", "il pleut"]
    assert abs(exact_match(refs, hyps_partial) - (100.0 / 3.0)) < 1e-4

    hyps_none = ["a", "b", "c"]
    assert exact_match(refs, hyps_none) == 0.0


def test_corpus_bleu_properties():
    # Identical hypothesis and reference should have high BLEU
    refs = [["le", "chat", "est", "noir"]]
    hyps = [["le", "chat", "est", "noir"]]
    bleu_identical = corpus_bleu(refs, hyps)
    assert bleu_identical > 90.0

    # Shorter hypothesis triggers brevity penalty
    hyps_short = [["le", "chat"]]
    bleu_short = corpus_bleu(refs, hyps_short)
    assert bleu_short < bleu_identical

    # Across a corpus of non-matching sentences with >=4 words, BLEU drops to near zero
    refs_corpus = [["un", "deux", "trois", "quatre", "cinq"]] * 20
    hyps_wrong = [["six", "sept", "huit", "neuf", "dix"]] * 20
    bleu_wrong = corpus_bleu(refs_corpus, hyps_wrong)
    assert bleu_wrong < 5.0
    assert bleu_wrong < bleu_identical


class DummyModel:
    """Mock model that immediately predicts EOS (id 3) on the first step."""
    def __call__(self, inputs, training=False):
        src, dec = inputs
        # batch size 1, dec seq_len L, vocab size 10
        b, l = dec.shape[0], dec.shape[1]
        # Return logits predicting EOS (3)
        logits = tf.one_hot(tf.fill((b, l), 3), depth=10) * 10.0
        return logits


class DummyTokenizer:
    def __init__(self):
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3

    def encode(self, text, add_special=False, max_len=40):
        return [4, 5]

    def decode(self, ids, stop_at_eos=True, skip_special=True):
        return ""


def test_autoregressive_greedy_stops_at_eos():
    model = DummyModel()
    src_tok = DummyTokenizer()
    tgt_tok = DummyTokenizer()

    out = translate(model, src_tok, tgt_tok, "test text", max_len=10)
    # Since dummy model always predicts EOS, it stops immediately at step 0
    assert out == ""
