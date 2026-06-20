from __future__ import annotations
from rank_bm25 import BM25Okapi
from code_rag_eval.types import Chunk, RetrievedChunk
from code_rag_eval.retrieve.tokenize import tokenize_code


class BM25Retriever:
    """Lexical retriever over code-aware tokens. Builds its index from a chunk list."""

    def __init__(self, chunks: list[Chunk]):
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize_code(c.text) for c in self._chunks]) if self._chunks else None

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize_code(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]), rank=rank + 1)
                for rank, i in enumerate(order)]
