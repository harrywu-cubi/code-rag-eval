from code_rag_eval.types import Chunk, RetrievedChunk
from code_rag_eval.eval.dataset import EvalRecord
from code_rag_eval.eval.coverage import chunk_covers
from code_rag_eval.eval.retrieval_metrics import hit_at_k, recall_at_k, reciprocal_rank


def _r(file, s, e, rank):
    return RetrievedChunk(chunk=Chunk(text="x", file=file, start_line=s, end_line=e), score=1.0, rank=rank)


def test_chunk_covers_requires_same_file_and_overlap():
    c = Chunk(text="x", file="a.py", start_line=10, end_line=20)
    assert chunk_covers(c, "a.py", (15, 16)) is True
    assert chunk_covers(c, "a.py", (20, 25)) is True   # touching boundary overlaps
    assert chunk_covers(c, "a.py", (21, 30)) is False
    assert chunk_covers(c, "b.py", (15, 16)) is False


def _record(symbols, files, ranges):
    return EvalRecord(id="q", category="locate", question="q",
                      gold_symbols=symbols, gold_files=files, gold_line_ranges=ranges,
                      reference_answer="a")


def test_hit_recall_mrr():
    rec = _record(["s1", "s2"], ["a.py", "a.py"], [(10, 20), (100, 110)])
    retrieved = [_r("a.py", 12, 14, 1), _r("a.py", 200, 210, 2)]  # covers gold #1 only, at rank 1
    assert hit_at_k(retrieved, rec, k=2) == 1
    assert recall_at_k(retrieved, rec, k=2) == 0.5
    assert reciprocal_rank(retrieved, rec) == 1.0

    none = [_r("a.py", 200, 210, 1)]
    assert hit_at_k(none, rec, k=1) == 0
    assert recall_at_k(none, rec, k=1) == 0.0
    assert reciprocal_rank(none, rec) == 0.0
