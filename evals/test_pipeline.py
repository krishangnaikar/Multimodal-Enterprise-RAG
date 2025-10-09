import pytest
from utils.chunking import chunk_text
from retrieval.hybrid_search import keyword_search, fuse_results

def test_chunking_overlap():
    text = " ".join([f"w{i}" for i in range(3000)])
    chunks = chunk_text(text, max_tokens=200, overlap=50)
    assert len(chunks) > 5

def test_keyword_search():
    corpus = ["apples and bananas", "cats and dogs", "oranges and pears"]
    res = keyword_search(corpus, "apples", k=2)
    assert len(res) == 2
    assert "apples" in res[0]["text"]

def test_fusion_basic():
    vec = [{"score": 0.8, "text": "A"}, {"score": 0.1, "text": "B"}]
    kw  = [{"score": 5.0, "text": "B"}, {"score": 0.5, "text": "C"}]
    fused = fuse_results(vec, kw)
    assert len(fused) >= 2
