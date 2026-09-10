"""Explicit word-level tokenizer.

We deliberately build the token -> id / id -> token mapping ourselves so the
project does not hide the sequence representation behind a library.
"""
from __future__ import annotations

import json
import re
from collections import Counter

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class WordTokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"
    SOS = "<SOS>"
    EOS = "<EOS>"

    def __init__(self, vocab_size: int = 20000, min_freq: int = 1):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.id_to_token: list[str] = []
        self.token_to_id: dict[str, int] = {}

    # ---------------- special ids (fixed: PAD must be 0) ----------------
    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[self.SOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    def __len__(self) -> int:
        return len(self.id_to_token)

    # ---------------- build / encode / decode ----------------
    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def build(self, texts: list[str]) -> "WordTokenizer":
        counter: Counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))
        self.id_to_token = [self.PAD, self.UNK, self.SOS, self.EOS]
        for token, freq in counter.most_common():
            if len(self.id_to_token) >= self.vocab_size:
                break
            if freq < self.min_freq:
                continue
            self.id_to_token.append(token)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        return self

    def encode(self, text: str, add_special: bool = False,
               max_len: int | None = None) -> list[int]:
        ids = [self.token_to_id.get(t, self.unk_id) for t in self.tokenize(text)]
        if max_len is not None:
            ids = ids[:max_len]
        if add_special:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, stop_at_eos: bool = True,
               skip_special: bool = True) -> str:
        tokens: list[str] = []
        for i in ids:
            i = int(i)
            if i == self.eos_id and stop_at_eos:
                break
            if not (0 <= i < len(self.id_to_token)):
                tokens.append(self.UNK)
                continue
            tok = self.id_to_token[i]
            if skip_special and tok in (self.PAD, self.SOS, self.UNK):
                continue
            tokens.append(tok)
        return " ".join(tokens)

    # ---------------- persistence ----------------
    def save(self, path: str) -> None:
        payload = {"vocab_size": self.vocab_size, "min_freq": self.min_freq,
                   "id_to_token": self.id_to_token}
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path) as f:
            payload = json.load(f)
        tok = cls(payload["vocab_size"], payload["min_freq"])
        tok.id_to_token = payload["id_to_token"]
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        return tok
