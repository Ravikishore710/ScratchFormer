"""End-to-end evaluation on a held-out split + failure-case harvesting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.dataset import DataBundle
from .metrics import corpus_bleu, exact_match


def _reference_tokens(tgt_tok, text: str) -> list[str]:
    return tgt_tok.tokenize(text)


def quick_gen_metrics(model, bundle: DataBundle, src_texts, tgt_texts,
                      n: int = 200, max_len: int = 40) -> tuple[float, float]:
    """Exact-match + corpus BLEU on the first n examples (greedy decoding)."""
    from ..inference.generate import translate

    n = min(n, len(src_texts))
    hyps, refs = [], []
    for i in range(n):
        hyp = translate(model, bundle.src_tok, bundle.tgt_tok,
                        src_texts[i], max_len=max_len)
        hyps.append(hyp)
        refs.append(tgt_texts[i])
    em = exact_match(refs, hyps)
    bleu = corpus_bleu([_reference_tokens(bundle.tgt_tok, r) for r in refs],
                       [h.split() for h in hyps])
    return em, bleu


def evaluate_test_set(model, bundle: DataBundle, out_csv: str,
                      n: int = 500, max_len: int = 40) -> dict:
    """Teacher-forced metrics come from Trainer.evaluate; here we generate
    predictions autoregressively and store success/failure cases."""
    import sys
    import time
    from ..inference.generate import translate

    n = min(n, len(bundle.test_src_text))
    rows = []

    use_tqdm = True
    try:
        from tqdm import tqdm
        iterator = tqdm(
            range(n),
            desc="Generating translations",
            unit="sent",
            ncols=100,
            file=sys.stdout,
            mininterval=0.5,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, ETA: {remaining}, {rate_fmt}]"
        )
    except ImportError:
        use_tqdm = False
        iterator = range(n)

    t0 = time.time()
    for i in iterator:
        src = bundle.test_src_text[i]
        ref = bundle.test_tgt_text[i]
        hyp = translate(model, bundle.src_tok, bundle.tgt_tok, src, max_len=max_len)
        rows.append({
            "source": src, "reference": ref, "hypothesis": hyp,
            "exact_match": ref.strip() == hyp.strip(),
            "ref_len": len(ref.split()), "hyp_len": len(hyp.split()),
        })
        if not use_tqdm and ((i + 1) % 25 == 0 or (i + 1) == n):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i+1:>3}/{n}] ({(i+1)/n*100:5.1f}%) | "
                  f"Elapsed: {int(elapsed//60):02d}:{int(elapsed%60):02d} | "
                  f"ETA: {int(remaining//60):02d}:{int(remaining%60):02d} | "
                  f"{rate:.2f} sent/s", flush=True)

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    em = 100.0 * df["exact_match"].mean()
    bleu = corpus_bleu([_reference_tokens(bundle.tgt_tok, r) for r in df["reference"]],
                       [h.split() for h in df["hypothesis"]])
    premature = df[df["hyp_len"] < df["ref_len"] * 0.5]
    return {
        "n_examples": n,
        "exact_match": em,
        "bleu": bleu,
        "mean_ref_len": float(df["ref_len"].mean()),
        "mean_hyp_len": float(df["hyp_len"].mean()),
        "n_premature_eos": int(len(premature)),
        "predictions_csv": out_csv,
    }
