"""Autoregressive greedy decoding -- NO teacher forcing at inference time."""
from __future__ import annotations

import tensorflow as tf


@tf.function(reduce_retracing=True)
def _last_logits(model, src, dec_in):
    return model((src, dec_in), training=False)[:, -1, :]


def translate(model, src_tok, tgt_tok, text: str,
              max_len: int = 40, strip_special: bool = True) -> str:
    """Greedy autoregressive translation: <SOS> -> token -> token ... -> <EOS>."""
    ids = src_tok.encode(text, add_special=False, max_len=max_len)
    src = tf.constant([ids], dtype=tf.int64)
    dec = [[tgt_tok.sos_id]]
    for _ in range(max_len):
        logits = _last_logits(model, src, tf.constant(dec, dtype=tf.int64))
        next_id = int(tf.argmax(logits[0]))
        if next_id == tgt_tok.eos_id:
            break
        dec[0].append(next_id)
    return tgt_tok.decode(dec[0][1:], stop_at_eos=True, skip_special=strip_special)


def translate_batch(model, src_tok, tgt_tok, texts: list[str],
                    max_len: int = 40) -> list[str]:
    return [translate(model, src_tok, tgt_tok, t, max_len) for t in texts]
