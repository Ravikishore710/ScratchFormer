"""Task metrics: exact match + a dependency-free corpus BLEU with smoothing."""
from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(references: list[list[str]], hypotheses: list[list[str]],
                max_n: int = 4) -> float:
    """Simple smoothed corpus BLEU (add-one smoothing on modified precisions)."""
    clipped = [0] * max_n
    totals = [0] * max_n
    ref_len = hyp_len = 0
    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            hyp_counts = _ngrams(hyp, n)
            ref_counts = _ngrams(ref, n)
            clipped[n - 1] += sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
            totals[n - 1] += max(len(hyp) - n + 1, 0)
    precisions = [(clipped[i] + 1.0) / (totals[i] + 1.0) for i in range(max_n)]
    if min(precisions) == 0:
        return 0.0
    log_bleu = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(hyp_len, 1))
    return 100.0 * bp * math.exp(log_bleu)


def exact_match(references: list[str], hypotheses: list[str]) -> float:
    """Fraction of exactly-correct sentence translations (0-100)."""
    if not references:
        return 0.0
    return 100.0 * sum(r.strip() == h.strip() for r, h in zip(references, hypotheses)) / len(references)
