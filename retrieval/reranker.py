from __future__ import annotations

def topical_rerank(results, topic_hint = ""):
    hint = set(topic_hint.lower().split())
    boosted = []
    for r in results:
        score = r["score"]
        txt = r.get("text", "").lower()
        overlap = len(hint.intersection(set(txt.split())))
        boosted.append({**r, "score": score + 0.05 * overlap})
    return sorted(boosted, key=lambda x: x["score"], reverse=True)
