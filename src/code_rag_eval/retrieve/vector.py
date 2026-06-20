from __future__ import annotations
from code_rag_eval.types import RetrievedChunk


class VectorRetriever:
    def __init__(self, store, embed_client):
        self._store = store
        self._embed = embed_client

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        qv = self._embed.embed([query])[0]
        hits = self._store.query(qv, k)
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=i + 1)
            for i, (chunk, score) in enumerate(hits)
        ]
