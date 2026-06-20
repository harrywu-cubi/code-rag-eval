from code_rag_eval.config import ExperimentConfig, RetrievalConfig
from code_rag_eval.types import Chunk
from code_rag_eval.ingest.store import ChromaStore
from code_rag_eval.retrieve.factory import make_retriever
from code_rag_eval.retrieve.vector import VectorRetriever
from code_rag_eval.retrieve.bm25 import BM25Retriever
from code_rag_eval.retrieve.hybrid import HybridRetriever
from tests.fakes import FakeEmbeddingClient


def _store():
    s = ChromaStore(collection_name="factory-test")
    embed = FakeEmbeddingClient()
    chunks = [Chunk(text="def login(): pass", file="a.py", start_line=1, end_line=1)]
    s.add(chunks, embed.embed([c.text for c in chunks]))
    return s


def _cfg(method):
    return ExperimentConfig(name="t", retrieval=RetrievalConfig(method=method))


def test_make_retriever_selects_impl():
    store, embed = _store(), FakeEmbeddingClient()
    assert isinstance(make_retriever(_cfg("vector"), store, embed), VectorRetriever)
    assert isinstance(make_retriever(_cfg("bm25"), store, embed), BM25Retriever)
    assert isinstance(make_retriever(_cfg("hybrid"), store, embed), HybridRetriever)
