from code_rag_eval.types import Chunk
from code_rag_eval.retrieve.bm25 import BM25Retriever


def _chunks():
    return [
        Chunk(text="def login(user): pass", file="auth.py", start_line=1, end_line=1),
        Chunk(text="def logout(user): pass", file="auth.py", start_line=2, end_line=2),
        Chunk(text="def render_template(name): pass", file="view.py", start_line=1, end_line=1),
    ]


def test_bm25_ranks_exact_symbol_first():
    retr = BM25Retriever(_chunks())
    results = retr.retrieve("login", k=2)
    assert len(results) == 2
    assert results[0].rank == 1 and results[1].rank == 2
    assert results[0].chunk.text.startswith("def login")


def test_bm25_empty_corpus_returns_empty():
    assert BM25Retriever([]).retrieve("login", k=5) == []
