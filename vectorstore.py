"""Step 3 + 4 + 5: embeddings, FAISS index, retriever."""
import os

import faiss
import numpy as np
from fastembed import TextEmbedding

from .ingest import Chunk

_MODEL = None


def get_embedder() -> TextEmbedding:
    global _MODEL
    if _MODEL is None:
        name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _MODEL = TextEmbedding(name)
    return _MODEL


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.index: faiss.IndexFlatIP | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(get_embedder().embed(texts)), dtype="float32")
        faiss.normalize_L2(vecs)  # so inner product = cosine similarity
        return vecs

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self._embed([c.text for c in chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])  # cosine sim (vectors are normalized)
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def search(self, query: str, k: int = 5, only_sources: list[str] | None = None):
        """Return top-k [(Chunk, score)], optionally limited to certain papers."""
        if self.index is None or not self.chunks:
            return []
        qv = self._embed([query])
        # over-fetch, then filter, so the paper filter still leaves k results
        fetch = len(self.chunks) if only_sources else min(k, len(self.chunks))
        scores, ids = self.index.search(qv, fetch)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if only_sources and chunk.source not in only_sources:
                continue
            results.append((chunk, float(score)))
            if len(results) == k:
                break
        return results