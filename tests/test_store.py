from code_rag_eval.types import Chunk
from code_rag_eval.ingest.store import ChromaStore


def test_add_then_query_round_trips_chunk():
    store = ChromaStore(collection_name="store-test")  # ephemeral, in-memory
    chunks = [
        Chunk(text="def login(): ...", file="auth.py", start_line=1, end_line=3, symbol="auth.login"),
        Chunk(text="def logout(): ...", file="auth.py", start_line=5, end_line=7, symbol="auth.logout"),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    store.add(chunks, vectors)
    hits = store.query([1.0, 0.0], n=1)
    assert len(hits) == 1
    chunk, score = hits[0]
    assert chunk.file == "auth.py"
    assert chunk.symbol == "auth.login"
    assert chunk.start_line == 1 and chunk.end_line == 3
    assert 0.0 <= score <= 1.0
