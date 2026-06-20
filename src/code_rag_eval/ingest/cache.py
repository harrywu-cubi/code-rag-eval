from __future__ import annotations
import hashlib
import json
import sqlite3
from pathlib import Path


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: str | Path):
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v TEXT)")
        self._conn.commit()

    def get(self, model: str, text: str) -> list[float] | None:
        row = self._conn.execute("SELECT v FROM emb WHERE k=?", (_key(model, text),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, text: str, vector: list[float]) -> None:
        self._conn.execute("INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)",
                           (_key(model, text), json.dumps(vector)))
        self._conn.commit()


class CachedEmbeddingClient:
    """Wraps any EmbeddingClient; only misses hit the inner client."""

    def __init__(self, inner, cache: EmbeddingCache):
        self._inner = inner
        self._cache = cache
        self.model = inner.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [self._cache.get(self.model, t) for t in texts]
        misses = [t for t, r in zip(texts, results) if r is None]
        if misses:
            computed = self._inner.embed(misses)
            it = iter(computed)
            for i, r in enumerate(results):
                if r is None:
                    vec = next(it)
                    self._cache.put(self.model, texts[i], vec)
                    results[i] = vec
        return [r for r in results if r is not None]
