from __future__ import annotations
from rank_bm25 import BM25Okapi
import numpy as np

def keyword_search(corpus_texts, query, k = 5):
    if not corpus_texts:
        return []
    tokenized_corpus = [t.lower().split() for t in corpus_texts]
    if not any(tokenized_corpus):
        return []
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"score": float(scores[i]), "text": corpus_texts[i], "corpus_idx": i} for i in idxs]

def _safe_key(*parts):
    return "::".join("" if p is None else str(p) for p in parts)

def fuse_results(vec_results, kw_results, w_vec = 0.6):
    all_items = {}

    def add_or_accumulate(key, score, payload):
        cur = all_items.get(key)
        if cur is None:
            all_items[key] = {"score": float(score), **payload}
        else:
            cur["score"] += float(score)

    v_scores = np.array([r.get("score", 0.0) for r in vec_results] or [1.0], dtype="float32")
    k_scores = np.array([r.get("score", 0.0) for r in kw_results] or [1.0], dtype="float32")
    v_norm = (v_scores - v_scores.min()) / (v_scores.ptp() + 1e-9)
    k_norm = (k_scores - k_scores.min()) / (k_scores.ptp() + 1e-9)

    for r, s in zip(vec_results, v_norm):
        key = _safe_key(r.get("doc_id"), r.get("chunk_id"), (r.get("text") or "")[:30])
        add_or_accumulate(key, w_vec * float(s), r)

    for r, s in zip(kw_results, k_norm):
        key = _safe_key(r.get("doc_id"), r.get("corpus_idx"), (r.get("text") or "")[:30])
        add_or_accumulate(key, (1 - w_vec) * float(s), r)

    return sorted(all_items.values(), key=lambda x: x["score"], reverse=True)
