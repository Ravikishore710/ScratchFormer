"""Dataset construction for the English -> French machine-translation task.

Pipeline:
    download -> parse pairs -> filter/subsample -> split -> tokenize (train only)
             -> pad -> tf.data
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..config import Config
from .tokenizer import WordTokenizer


@dataclass
class DataBundle:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    src_tok: WordTokenizer
    tgt_tok: WordTokenizer
    val_src_text: list
    val_tgt_text: list
    test_src_text: list
    test_tgt_text: list
    stats: dict = field(default_factory=dict)


def download_fra_eng(cfg: Config) -> Path:
    """Download the manythings fra-eng zip and extract fra.txt (idempotent)."""
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    txt_path = data_dir / "fra.txt"
    if txt_path.exists():
        return txt_path
    zip_path = data_dir / "fra-eng.zip"
    if not zip_path.exists():
        print(f"Downloading {cfg.dataset_url} ...")
        req = urllib.request.Request(
            cfg.dataset_url,
            headers={"User-Agent": "curl/8.4.0", "Accept": "*/*"},
        )
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out:
            shutil.copyfileobj(resp, out)
    with zipfile.ZipFile(zip_path) as zf:
        target_name = "fra.txt" if "fra.txt" in zf.namelist() else next(
            n for n in zf.namelist() if n.endswith(".txt") and not n.startswith("_")
        )
        zf.extract(target_name, data_dir)
        extracted = data_dir / target_name
        if extracted != txt_path:
            extracted.rename(txt_path)
    return txt_path


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            src, tgt = parts[0].strip(), parts[1].strip()
            if not src or not tgt or src.lower() == "english":
                continue
            pairs.append((src, tgt))
    return pairs


def _encode_padded(texts: list[str], tok: WordTokenizer,
                   max_len: int, add_special: bool) -> np.ndarray:
    """Encode to token ids and pad/truncate every sequence to exactly max_len."""
    arr = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        if add_special:
            ids = tok.encode(text, add_special=False, max_len=max_len - 1) + [tok.eos_id]
        else:
            ids = tok.encode(text, add_special=False, max_len=max_len)
        ids = ids[:max_len]
        arr[i, : len(ids)] = ids
    return arr


def _make_tf_dataset(src_ids: np.ndarray, tgt_ids: np.ndarray, cfg: Config,
                     sos_id: int, shuffle: bool) -> tf.data.Dataset:
    sos = tf.constant(sos_id, dtype=tf.int64)

    def shift_right(tgt):
        return tf.concat([tf.fill([1], sos), tgt[:-1]], axis=0)

    ds = tf.data.Dataset.from_tensor_slices((src_ids, tgt_ids))
    if shuffle:
        ds = ds.shuffle(len(src_ids), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda s, t: ((s, shift_right(t)), t))
    return ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


def prepare_datasets(cfg: Config, verbose: bool = True) -> DataBundle:
    cfg.validate()
    txt_path = download_fra_eng(cfg)
    pairs = load_pairs(txt_path)

    # ---- filter by word length (leave room for <SOS>/<EOS>) ----
    filtered = [
        (s, t) for s, t in pairs
        if 0 < len(s.split()) <= cfg.max_seq_len - 2
        and 0 < len(t.split()) <= cfg.max_seq_len - 2
    ]
    rng = np.random.default_rng(cfg.seed)
    if len(filtered) > cfg.max_pairs:
        idx = rng.choice(len(filtered), size=cfg.max_pairs, replace=False)
        filtered = [filtered[i] for i in idx]

    # ---- split (test stays untouched until final evaluation) ----
    n = len(filtered)
    n_val = int(n * cfg.val_size)
    n_test = int(n * cfg.test_size)
    n_train = n - n_val - n_test
    perm = rng.permutation(n)
    train = [filtered[i] for i in perm[:n_train]]
    val = [filtered[i] for i in perm[n_train:n_train + n_val]]
    test = [filtered[i] for i in perm[n_train + n_val:]]

    # ---- tokenizers are fit ONLY on the training split ----
    src_tok = WordTokenizer(cfg.src_vocab_size, cfg.min_freq).build([s for s, _ in train])
    tgt_tok = WordTokenizer(cfg.tgt_vocab_size, cfg.min_freq).build([t for _, t in train])

    src_train = _encode_padded([s for s, _ in train], src_tok, cfg.max_seq_len, add_special=False)
    tgt_train = _encode_padded([t for _, t in train], tgt_tok, cfg.max_seq_len, add_special=True)
    src_val = _encode_padded([s for s, _ in val], src_tok, cfg.max_seq_len, add_special=False)
    tgt_val = _encode_padded([t for _, t in val], tgt_tok, cfg.max_seq_len, add_special=True)
    src_test = _encode_padded([s for s, _ in test], src_tok, cfg.max_seq_len, add_special=False)
    tgt_test = _encode_padded([t for _, t in test], tgt_tok, cfg.max_seq_len, add_special=True)

    train_ds = _make_tf_dataset(src_train, tgt_train, cfg, tgt_tok.sos_id, shuffle=True)
    val_ds = _make_tf_dataset(src_val, tgt_val, cfg, tgt_tok.sos_id, shuffle=False)
    test_ds = _make_tf_dataset(src_test, tgt_test, cfg, tgt_tok.sos_id, shuffle=False)

    stats = {
        "total_pairs_available": len(pairs),
        "pairs_used": n,
        "train": n_train, "val": n_val, "test": n_test,
        "src_vocab_size": len(src_tok), "tgt_vocab_size": len(tgt_tok),
        "max_seq_len": cfg.max_seq_len,
        "avg_src_words": float(np.mean([len(s.split()) for s, _ in filtered])),
        "avg_tgt_words": float(np.mean([len(t.split()) for _, t in filtered])),
    }
    if verbose:
        for k, v in stats.items():
            print(f"{k:>24}: {v}")

    return DataBundle(
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        src_tok=src_tok, tgt_tok=tgt_tok,
        val_src_text=[s for s, _ in val], val_tgt_text=[t for _, t in val],
        test_src_text=[s for s, _ in test], test_tgt_text=[t for _, t in test],
        stats=stats,
    )
