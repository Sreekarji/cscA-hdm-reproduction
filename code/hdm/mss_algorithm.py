"""
Minimum Synonymous Subsequence (MSS) — paper Algorithm 1.

Given a source sequence psi_S and a semantic-similarity threshold epsilon, MSS
returns the shortest subsequence whose meaning is still within epsilon of the
original. This is the paper's semantic compression operator: what actually goes
over the air is the MSS, not the full sentence, so the compression ratio is a
MEASURED quantity rather than a fixed truncation constant.

Greedy realisation of Algorithm 1: repeatedly delete the single token whose
removal costs the least similarity, and stop as soon as the best available
deletion would push similarity below epsilon. Deletion order is therefore
importance-ordered, which is what distinguishes MSS from the eta=0.73 prefix
truncation used elsewhere in the repo (that keeps the FIRST 73% of words
regardless of which words carry the meaning).

Similarity is BERT-family cosine (MiniLM by default), matching the paper's use
of a pretrained sentence encoder to define synonymy.

Batching note: the greedy loop is round-synchronous ACROSS sentences — every
active sentence takes one deletion step per round and all candidate variants
are scored in a single sim_fn call. Cost is O(max_len) encoder calls total, not
O(n_sentences * max_len), which matters because callers may route sim_fn
through a subprocess (see multimodal_eval.batch_similarity).

compute_mim is re-exported here so the semantic-layer primitives (Eq. 5 message
importance and Algorithm 1 compression) can be imported from one place.
"""

import numpy as np

try:                                     # package-relative when imported as a module
    from mcs_table import compute_mim
except ImportError:                      # path-insert style used by the experiments
    import os, sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "channel"))
    from mcs_table import compute_mim

__all__ = [
    "bert_similarity",
    "minimum_synonymous_subsequence",
    "minimum_synonymous_subsequence_batch",
    "compute_mim",
    "DEFAULT_EPSILON",
]

DEFAULT_EPSILON = 0.95

_encoder = None


def _get_encoder(model_dir=None):
    """Lazily load MiniLM on CPU. Returns None if unavailable."""
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from sentence_transformers import SentenceTransformer
        if model_dir is None:
            from config import MINIML_PATH
            model_dir = str(MINIML_PATH)
        _encoder = SentenceTransformer(model_dir, device="cpu")
    except Exception:
        _encoder = None
    return _encoder


def _jaccard(originals, candidates):
    """Encoder-free fallback so MSS still runs without sentence_transformers."""
    out = []
    for o, c in zip(originals, candidates):
        so, sc = set(o.lower().split()), set(c.lower().split())
        out.append(len(so & sc) / max(len(so), 1))
    return np.array(out, dtype=float)


def _default_sim_fn(originals, candidates):
    model = _get_encoder()
    if model is None:
        return _jaccard(originals, candidates)
    embs = model.encode(list(originals) + list(candidates),
                        batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True)
    n = len(originals)
    a, b = embs[:n], embs[n:]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.clip((a * b).sum(axis=1), 0.0, 1.0)


def bert_similarity(text_a, text_b, sim_fn=None):
    """Cosine similarity in [0, 1] between two strings."""
    fn = sim_fn or _default_sim_fn
    return float(fn([text_a], [text_b])[0])


def minimum_synonymous_subsequence_batch(
    sentences, epsilon=DEFAULT_EPSILON, sim_fn=None, min_tokens=1,
):
    """Algorithm 1 over a list of sentences.

    Returns list of (mss_text, compression_ratio), ratio = kept / original
    tokens. A sentence that cannot lose any token without dropping below
    epsilon is returned unchanged with ratio 1.0.
    """
    fn = sim_fn or _default_sim_fn
    tokens = [s.split() for s in sentences]
    active = [i for i, t in enumerate(tokens) if len(t) > min_tokens]

    while active:
        # Build every one-token-deletion variant for every still-active sentence.
        owners, drop_at, origs, cands = [], [], [], []
        for i in active:
            toks = tokens[i]
            for j in range(len(toks)):
                owners.append(i)
                drop_at.append(j)
                origs.append(sentences[i])
                cands.append(" ".join(toks[:j] + toks[j + 1:]))

        sims = np.asarray(fn(origs, cands), dtype=float)

        # Per sentence, keep the cheapest deletion if it stays above epsilon.
        best = {}
        for k, i in enumerate(owners):
            if i not in best or sims[k] > sims[best[i]]:
                best[i] = k
        still_active = []
        for i, k in best.items():
            if sims[k] < epsilon:
                continue                      # any further deletion breaks synonymy
            del tokens[i][drop_at[k]]
            if len(tokens[i]) > min_tokens:
                still_active.append(i)
        active = still_active

    out = []
    for s, toks in zip(sentences, tokens):
        n_orig = max(len(s.split()), 1)
        out.append((" ".join(toks), len(toks) / n_orig))
    return out


def minimum_synonymous_subsequence(
    sentence, epsilon=DEFAULT_EPSILON, sim_fn=None, min_tokens=1,
):
    """Single-sentence Algorithm 1. Returns (mss_text, compression_ratio)."""
    return minimum_synonymous_subsequence_batch(
        [sentence], epsilon=epsilon, sim_fn=sim_fn, min_tokens=min_tokens)[0]


if __name__ == "__main__":
    demo = [
        "The proposed semantic communication system transmits the information very efficiently.",
        "Bandwidth allocation is optimized by the heterogeneous graph attention network.",
    ]
    for eps in (0.99, 0.95, 0.90):
        res = minimum_synonymous_subsequence_batch(demo, epsilon=eps)
        print(f"\nepsilon={eps}")
        for (mss, ratio), src in zip(res, demo):
            print(f"  ratio={ratio:.3f}  {mss!r}")
            assert 0.0 < ratio <= 1.0
            assert len(mss.split()) <= len(src.split())
    print(f"\ncompute_mim([0.5, 0.3, 0.2]) = {compute_mim([0.5, 0.3, 0.2]):.4f}")
    print("MSS smoke test PASSED")
