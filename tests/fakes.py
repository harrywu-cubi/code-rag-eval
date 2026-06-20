from __future__ import annotations
import hashlib


class FakeEmbeddingClient:
    """Deterministic small-dim embeddings derived from text bytes. No network."""

    def __init__(self, model: str = "fake-embed", dim: int = 16):
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [h[i % len(h)] / 255.0 for i in range(self.dim)]
            out.append(vec)
        return out


class FakeLLMClient:
    """Returns a canned answer; records the last prompt it received."""

    def __init__(self, answer: str = "See fastapi/routing.py:1 for details."):
        self.answer = answer
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self.answer


class FakeJudge:
    """Returns fixed verdicts for judge-based metrics."""

    def __init__(self, relevant: bool = True, faithful: float = 1.0, relevancy: float = 1.0):
        self._relevant = relevant
        self._faithful = faithful
        self._relevancy = relevancy

    def is_relevant(self, question: str, chunk_text: str) -> bool:
        return self._relevant

    def faithfulness(self, answer: str, context: str) -> float:
        return self._faithful

    def answer_relevancy(self, question: str, answer: str) -> float:
        return self._relevancy
