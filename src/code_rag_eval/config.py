from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "ast"] = "fixed"
    window_lines: int = 40       # used by fixed
    overlap_lines: int = 10      # used by fixed


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "voyage"] = "openai"
    model: str = "text-embedding-3-large"


class RetrievalConfig(BaseModel):
    method: Literal["vector", "bm25", "hybrid"] = "vector"
    top_k: int = 5
    rrf_k: int = 60


class GenerationConfig(BaseModel):
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-sonnet-4-6"


class ExperimentConfig(BaseModel):
    name: str
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(**data)
