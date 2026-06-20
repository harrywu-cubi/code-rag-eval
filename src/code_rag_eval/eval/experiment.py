from __future__ import annotations
from code_rag_eval.config import ExperimentConfig, EmbeddingConfig

_EMBEDDINGS = {
    "openai": "text-embedding-3-large",
    "voyage": "voyage-code-3",
}


def matrix_configs(base: ExperimentConfig) -> list[ExperimentConfig]:
    """The 8-config sweep: chunking{fixed,ast} x embedding{openai,voyage} x retrieval{vector,hybrid}."""
    configs: list[ExperimentConfig] = []
    for strategy in ("fixed", "ast"):
        for provider in ("openai", "voyage"):
            for method in ("vector", "hybrid"):
                configs.append(base.model_copy(update={
                    "name": f"{strategy}_{provider}_{method}",
                    "chunking": base.chunking.model_copy(update={"strategy": strategy}),
                    "embedding": EmbeddingConfig(provider=provider, model=_EMBEDDINGS[provider]),
                    "retrieval": base.retrieval.model_copy(update={"method": method}),
                }))
    return configs
