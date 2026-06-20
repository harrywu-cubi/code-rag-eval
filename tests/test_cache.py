from pathlib import Path
from code_rag_eval.ingest.cache import EmbeddingCache, CachedEmbeddingClient
from tests.fakes import FakeEmbeddingClient


class _CountingClient(FakeEmbeddingClient):
    def __init__(self):
        super().__init__(model="counting")
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return super().embed(texts)


def test_cache_avoids_recompute(tmp_path: Path):
    inner = _CountingClient()
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    client = CachedEmbeddingClient(inner, cache)

    v1 = client.embed(["alpha", "beta"])
    assert inner.calls == 1                # one batch for two misses
    v2 = client.embed(["alpha", "beta"])   # both hits
    assert inner.calls == 1                # no new inner call
    assert v1 == v2

    mixed = client.embed(["alpha", "gamma"])  # one hit, one miss
    assert inner.calls == 2                    # only the miss triggered a call
    assert mixed[0] == v1[0]
