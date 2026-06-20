from __future__ import annotations
from code_rag_eval.config import EmbeddingConfig, GenerationConfig
from code_rag_eval.ingest.embed import OpenAIEmbeddingClient, VoyageEmbeddingClient
from code_rag_eval.generate.answer import AnthropicClient


def make_embedding_client(cfg: EmbeddingConfig):
    if cfg.provider == "openai":
        return OpenAIEmbeddingClient(model=cfg.model)
    if cfg.provider == "voyage":
        return VoyageEmbeddingClient(model=cfg.model)
    raise ValueError(f"unknown embedding provider: {cfg.provider}")


def make_llm_client(cfg: GenerationConfig):
    if cfg.provider == "anthropic":
        return AnthropicClient(model=cfg.model)
    raise ValueError(f"llm provider not supported: {cfg.provider}")
