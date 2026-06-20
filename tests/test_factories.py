import pytest
from code_rag_eval.config import EmbeddingConfig, GenerationConfig
from code_rag_eval.factories import make_embedding_client, make_llm_client


def _stub_factory():
    created = {}

    class _Stub:
        def __init__(self, model):
            created["model"] = model
            self.model = model

    return created, _Stub


def test_openai_embedding_selected(monkeypatch):
    created, stub = _stub_factory()
    monkeypatch.setattr("code_rag_eval.factories.OpenAIEmbeddingClient", stub)
    client = make_embedding_client(EmbeddingConfig(provider="openai", model="text-embedding-3-large"))
    assert created["model"] == "text-embedding-3-large" and client.model == "text-embedding-3-large"


def test_voyage_embedding_selected(monkeypatch):
    created, stub = _stub_factory()
    monkeypatch.setattr("code_rag_eval.factories.VoyageEmbeddingClient", stub)
    client = make_embedding_client(EmbeddingConfig(provider="voyage", model="voyage-code-3"))
    assert created["model"] == "voyage-code-3" and client.model == "voyage-code-3"


def test_unknown_llm_provider_raises():
    with pytest.raises(ValueError):
        make_llm_client(GenerationConfig(provider="openai", model="gpt-x"))
