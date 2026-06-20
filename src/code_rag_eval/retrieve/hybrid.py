from __future__ import annotations
from code_rag_eval.types import Chunk, RetrievedChunk


class HybridRetriever:
    """Fuses a vector retriever and a lexical retriever with Reciprocal Rank Fusion."""

    def __init__(self, vector_retriever, bm25_retriever, rrf_k: int = 60):
        self._vec = vector_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        pool = max(k, 20)  # fuse a wider pool from each arm than we finally return
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for results in (self._vec.retrieve(query, pool), self._bm25.retrieve(query, pool)):
            for r in results:
                cid = r.chunk.chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + r.rank)
                chunks[cid] = r.chunk
        order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [RetrievedChunk(chunk=chunks[cid], score=score, rank=i + 1)
                for i, (cid, score) in enumerate(order)]
