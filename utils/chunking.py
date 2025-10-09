
def chunk_text(text, max_tokens = 800, overlap = 100):
    words = text.split()
    chunks = []
    i = 0
    step = max_tokens - overlap
    while i < len(words):
        chunk = words[i:i+max_tokens]
        chunks.append(" ".join(chunk))
        i += step
    return chunks

def enrich_metadata(doc_id, modality, source_path, extra= None):
    base = {"doc_id": doc_id, "modality": modality, "source": source_path}
    if extra:
        base.update(extra)
    return base
