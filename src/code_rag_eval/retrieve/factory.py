from __future__ import annotations
from code_rag_eval.config import ExperimentConfig
from code_rag_eval.retrieve.vector import VectorRetriever
from code_rag_eval.retrieve.bm25 import BM25Retriever
from code_rag_eval.retrieve.hybrid import HybridRetriever


def make_retriever(cfg: ExperimentConfig, store, embed_client):
    method = cfg.retrieval.method
    vector = VectorRetriever(store, embed_client)
    if method == "vector":
        return vector
    bm25 = BM25Retriever(store.all_chunks())
    if method == "bm25":
        return bm25
    if method == "hybrid":
        return HybridRetriever(vector, bm25, rrf_k=cfg.retrieval.rrf_k)
    raise ValueError(f"unknown retrieval method: {method}")
