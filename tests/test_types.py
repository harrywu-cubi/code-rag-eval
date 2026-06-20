from code_rag_eval.types import Chunk, RetrievedChunk


def test_chunk_id_is_stable_and_unique():
    c = Chunk(text="x", file="fastapi/routing.py", start_line=10, end_line=20)
    assert c.chunk_id == "fastapi/routing.py:10-20"
    assert c.kind == "fixed"


def test_retrieved_chunk_holds_score_and_rank():
    c = Chunk(text="x", file="a.py", start_line=1, end_line=2)
    rc = RetrievedChunk(chunk=c, score=0.9, rank=1)
    assert rc.score == 0.9 and rc.rank == 1 and rc.chunk is c
