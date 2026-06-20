import pytest
from code_rag_eval.config import EmbeddingConfig, GenerationConfig
from code_rag_eval.factories import make_embedding_client, make_llm_client


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        make_embedding_client(EmbeddingConfig(provider="voyage", model="voyage-code-3"))
    with pytest.raises(ValueError):
        make_llm_client(GenerationConfig(provider="openai", model="gpt-x"))


def test_openai_embedding_selected(monkeypatch):
    created = {}

    class _Stub:
        def __init__(self, model):
            created["model"] = model
            self.model = model

    monkeypatch.setattr("code_rag_eval.factories.OpenAIEmbeddingClient", _Stub)
    client = make_embedding_client(EmbeddingConfig(provider="openai", model="text-embedding-3-large"))
    assert created["model"] == "text-embedding-3-large"
    assert client.model == "text-embedding-3-large"
