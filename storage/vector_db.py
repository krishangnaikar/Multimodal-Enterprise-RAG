from __future__ import annotations
import numpy as np
from utils.ollama_client import ollama_embed
import uuid
import json
from pathlib import Path
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:
    QdrantClient = None
    qmodels = None

try:
    import faiss
except Exception:
    faiss = None


class VectorStore:
    def __init__(self, use_qdrant = True, collection = "mmrag", dim = 768):
        self.collection = collection
        self.dim = dim
        self.backend = "numpy"
        self.texts = []
        self.payloads = []
        self._next_id = 0

        if use_qdrant and QdrantClient is not None:
            try:
                self.client = QdrantClient(host="localhost", port=6333, timeout=2.0)
                _ = self.client.get_collections()
                self.backend = "qdrant"
                try:
                    self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config=qmodels.VectorParams(size=self.dim, distance=qmodels.Distance.COSINE),
                    )
                except Exception:
                    pass
            except Exception:
                self.backend = "numpy"


        if self.backend == "numpy":
            self.vecs = np.zeros((0, self.dim), dtype="float32")

    def delete_by_doc(self, doc_id):
        if self.backend == "qdrant":
            flt = qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
            )
            self.client.delete(collection_name=self.collection,
                               points_selector=qmodels.FilterSelector(filter=flt))

    def save_local(self, user):
        base = Path("state") / "users" / user
        base.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss":
            import faiss
            faiss.write_index(self.index, str(base / "faiss.index"))
            (base / "payloads.json").write_text(json.dumps(self.payloads), encoding="utf-8")
        elif self.backend == "numpy":
            np.save(str(base / "vecs.npy"), getattr(self, "vecs", np.zeros((0, self.dim), "float32")))
            (base / "payloads.json").write_text(json.dumps(self.payloads), encoding="utf-8")

    def load_local(self, user):
        base = Path("state") / "users" / user
        if self.backend == "faiss":
            try:
                import faiss
                self.index = faiss.read_index(str(base / "faiss.index"))
                self.payloads = json.loads((base / "payloads.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        elif self.backend == "numpy":
            try:
                self.vecs = np.load(str(base / "vecs.npy"))
                self.payloads = json.loads((base / "payloads.json").read_text(encoding="utf-8"))
            except Exception:
                pass


    def reset(self):
        if self.backend == "qdrant":
            try:
                self.client.recreate_collection(
                    collection_name=self.collection,
                    vectors_config=qmodels.VectorParams(size=self.dim, distance=qmodels.Distance.COSINE),
                )
            except Exception:
                try:
                    self.client.delete_collection(self.collection)
                except Exception:
                    pass
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qmodels.VectorParams(size=self.dim, distance=qmodels.Distance.COSINE),
                )
            return
        if self.backend == "faiss":
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.dim)
            except Exception:
                self.backend = "numpy"
                self.vecs = np.zeros((0, self.dim), dtype="float32")
            self.payloads = []
            return
        self.vecs = np.zeros((0, self.dim), dtype="float32")
        self.payloads = []
    def _normalize(self, vecs):
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.size == 0:
            return vecs
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms

    def _filter_good(self, texts, metadatas, embs):
        good_t, good_m, good_v = [], [], []
        for t, m, e in zip(texts, metadatas, embs):
            if isinstance(e, (list, tuple, np.ndarray)) and len(e) == self.dim:
                good_t.append(t)
                good_m.append(m)
                good_v.append(e)
        if not good_v:
            return [], [], np.zeros((0, self.dim), dtype="float32")
        vecs = np.array(good_v, dtype="float32")
        return good_t, good_m, vecs

    def index_texts(self, texts, metadatas, embed_model = "nomic-embed-text"):
        if not texts:
            return

        embs = ollama_embed(embed_model, texts)

        good_texts, good_metas, vecs = self._filter_good(texts, metadatas, embs)
        if vecs.shape[0] == 0:
            return
        vecs = self._normalize(vecs)

        if self.backend == "qdrant":
            points = []
            start = self._next_id
            for i, (v, md, txt) in enumerate(zip(vecs, good_metas, good_texts), start=start):
                points.append(
                    qmodels.PointStruct(
                        id=int(i),
                        vector=v.tolist(),
                        payload=md | {"text": txt},
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points)
            self._next_id += len(points)
            return

        if self.backend == "faiss":
            self.index.add(vecs)
        else:
            self.vecs = vecs if self.vecs.shape[0] == 0 else np.vstack([self.vecs, vecs])

        self.payloads.extend([md | {"text": t} for md, t in zip(good_metas, good_texts)])

    def search(self, query, k = 5, embed_model = "nomic-embed-text"):
        q_embs = ollama_embed(embed_model, [query])
        if not q_embs or not isinstance(q_embs[0], (list, tuple, np.ndarray)) or len(q_embs[0]) != self.dim:
            return []
        q = np.array(q_embs[0], dtype="float32").reshape(1, -1)
        q = self._normalize(q)

        if self.backend == "qdrant":
            res = self.client.search(self.collection, query_vector=q[0].tolist(), limit=k)
            return [{"score": r.score, **(r.payload or {})} for r in res]

        if self.backend == "faiss":
            if getattr(self, "index", None) is None or self.index.ntotal == 0:
                return []
            D, I = self.index.search(q, k)
            out = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx == -1 or idx >= len(self.payloads):
                    continue
                out.append({"score": float(score), **self.payloads[idx]})
            return out

        if getattr(self, "vecs", None) is None or self.vecs.shape[0] == 0:
            return []
        sims = (self.vecs @ q.T).ravel()
        idxs = sims.argsort()[::-1][:k]
        return [{"score": float(sims[i]), **self.payloads[i]} for i in idxs if i < len(self.payloads)]
