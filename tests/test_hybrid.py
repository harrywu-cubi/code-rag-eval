from code_rag_eval.types import Chunk, RetrievedChunk
from code_rag_eval.retrieve.hybrid import HybridRetriever


class _FixedRetriever:
    def __init__(self, ordered_chunks):
        self._ordered = ordered_chunks

    def retrieve(self, query, k):
        return [RetrievedChunk(chunk=c, score=1.0 / (i + 1), rank=i + 1)
                for i, c in enumerate(self._ordered[:k])]


def _c(name):
    return Chunk(text=name, file=f"{name}.py", start_line=1, end_line=1, symbol=name)


def test_hybrid_rrf_rewards_agreement():
    a, b, d = _c("a"), _c("b"), _c("d")
    # vector ranks a,b,d ; bm25 ranks b,a,d -> a and b both rank high in both arms
    vec = _FixedRetriever([a, b, d])
    bm25 = _FixedRetriever([b, a, d])
    results = HybridRetriever(vec, bm25, rrf_k=60).retrieve("q", k=3)
    assert {r.chunk.symbol for r in results} == {"a", "b", "d"}
    assert results[0].chunk.symbol in {"a", "b"}      # an agreed-upon top item wins
    assert results[-1].chunk.symbol == "d"            # ranked low in both -> last
    assert [r.rank for r in results] == [1, 2, 3]


def test_hybrid_dedupes_across_arms():
    a = _c("a")
    results = HybridRetriever(_FixedRetriever([a]), _FixedRetriever([a]), rrf_k=60).retrieve("q", k=5)
    assert len(results) == 1 and results[0].chunk.symbol == "a"
