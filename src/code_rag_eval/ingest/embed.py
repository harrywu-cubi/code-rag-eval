from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingClient:
    """Thin wrapper over the OpenAI embeddings API. Exercised manually / via CLI."""

    def __init__(self, model: str = "text-embedding-3-large"):
        from openai import OpenAI
        self._client = OpenAI()
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class VoyageEmbeddingClient:
    """Thin wrapper over the Voyage AI embeddings API. Exercised manually / via CLI."""

    def __init__(self, model: str = "voyage-code-3"):
        import voyageai
        self._client = voyageai.Client()
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(texts, model=self.model).embeddings
