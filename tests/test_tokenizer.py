"""Advanced unit tests for WordTokenizer."""
import json
import tempfile
from pathlib import Path

from src.data.tokenizer import WordTokenizer


def test_special_tokens_ids_and_immutability():
    tok = WordTokenizer().build(["bonjour le monde"])
    assert tok.pad_id == 0, "PAD must be id 0"
    assert tok.unk_id == 1, "UNK must be id 1"
    assert tok.sos_id == 2, "SOS must be id 2"
    assert tok.eos_id == 3, "EOS must be id 3"
    assert tok.id_to_token[:4] == [tok.PAD, tok.UNK, tok.SOS, tok.EOS]


def test_tokenization_french_accents_and_punctuation():
    tok = WordTokenizer()
    text = "L'élève, où va-t-il à la fête ?"
    tokens = tok.tokenize(text)
    assert "l" in tokens
    assert "'" in tokens
    assert "élève" in tokens
    assert "où" in tokens
    assert "fête" in tokens
    assert "?" in tokens


def test_min_frequency_and_vocab_size_cap():
    texts = [
        "apple banana apple orange apple",
        "banana orange banana",
        "pear",
    ]
    tok = WordTokenizer(vocab_size=10, min_freq=2).build(texts)
    assert "apple" in tok.token_to_id
    assert "banana" in tok.token_to_id
    assert "orange" in tok.token_to_id
    assert "pear" not in tok.token_to_id

    tok_capped = WordTokenizer(vocab_size=6, min_freq=1).build(texts)
    assert len(tok_capped) <= 6


def test_unknown_words_mapped_to_unk():
    tok = WordTokenizer().build(["hello world"])
    encoded = tok.encode("hello unknown_token world", add_special=False)
    assert encoded[0] == tok.token_to_id["hello"]
    assert encoded[1] == tok.unk_id
    assert encoded[2] == tok.token_to_id["world"]


def test_encode_and_decode_with_specials():
    tok = WordTokenizer().build(["je suis étudiant"])
    text = "je suis étudiant"
    encoded_special = tok.encode(text, add_special=True, max_len=10)
    assert encoded_special[0] == tok.sos_id
    assert encoded_special[-1] == tok.eos_id

    decoded = tok.decode(encoded_special, stop_at_eos=True, skip_special=True)
    assert decoded == text


def test_tokenizer_serialization_roundtrip():
    tok = WordTokenizer(vocab_size=50, min_freq=1).build(["le chat noir dort sur le tapis"])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tokenizer.json"
        tok.save(str(path))

        loaded = WordTokenizer.load(str(path))
        assert loaded.id_to_token == tok.id_to_token
        assert loaded.token_to_id == tok.token_to_id
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.min_freq == tok.min_freq
