import time
from storage.vector_db import VectorStore
from retrieval.hybrid_search import keyword_search, fuse_results

CORPUS = [
    "John Smith approved the Q3 budget on July 8, 2024.",
    "The Phoenix project uses Weaviate for vector search and Neo4j for graph storage.",
    "Audio transcript: The team discussed latency improvements and reduced hallucinations.",
]
META = [{"doc_id":"d1","chunk_id":0,"modality":"text","source":"synthetic"},
        {"doc_id":"d2","chunk_id":0,"modality":"text","source":"synthetic"},
        {"doc_id":"d3","chunk_id":0,"modality":"text","source":"synthetic"}]

QA = [
    ("Who approved the Q3 budget?", "John Smith"),
    ("Which vector DB is used by the Phoenix project?", "Weaviate"),
    ("What did the team discuss in audio?", "latency"),
]

def main():
    vs = VectorStore(use_qdrant=False, dim=768)
    vs.index_texts(CORPUS, META)

    hits = 0
    t0 = time.time()
    for q, gold in QA:
        vec = vs.search(q, k=3)
        kw = keyword_search(CORPUS, q, k=3)
        fused = fuse_results(vec, kw)
        ctx = " ".join([r["text"] for r in fused[:3]])
        if gold.lower() in ctx.lower():
            hits += 1
    dt = time.time() - t0

    print(f"Eval: {hits}/{len(QA)} retrieved gold in context; latency {dt:.3f}s")

if __name__ == "__main__":
    main()
