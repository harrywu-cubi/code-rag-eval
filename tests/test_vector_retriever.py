from code_rag_eval.types import Chunk
from code_rag_eval.ingest.store import ChromaStore
from code_rag_eval.retrieve.vector import VectorRetriever
from tests.fakes import FakeEmbeddingClient


def test_retrieve_returns_ranked_chunks():
    embed = FakeEmbeddingClient()
    store = ChromaStore(collection_name="retr-test")
    chunks = [
        Chunk(text="def login(): pass", file="auth.py", start_line=1, end_line=1),
        Chunk(text="def logout(): pass", file="auth.py", start_line=2, end_line=2),
        Chunk(text="def render(): pass", file="view.py", start_line=1, end_line=1),
    ]
    store.add(chunks, embed.embed([c.text for c in chunks]))
    retr = VectorRetriever(store, embed)
    results = retr.retrieve("def login(): pass", k=2)
    assert len(results) == 2
    assert results[0].rank == 1 and results[1].rank == 2
    assert results[0].chunk.text == "def login(): pass"  # identical query -> top hit
