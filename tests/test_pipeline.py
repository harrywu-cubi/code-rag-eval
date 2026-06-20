from pathlib import Path
from code_rag_eval.config import ChunkingConfig
from code_rag_eval.ingest.pipeline import build_chunks, ingest
from code_rag_eval.ingest.store import ChromaStore
from tests.fakes import FakeEmbeddingClient


def _make_corpus(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "fastapi"
    src = root / "fastapi"
    src.mkdir(parents=True)
    (src / "routing.py").write_text("\n".join(f"line{i}" for i in range(1, 51)), encoding="utf-8")
    return root, src


def test_build_chunks_uses_posix_relative_paths(tmp_path: Path):
    root, src = _make_corpus(tmp_path)
    chunks = build_chunks(src, root, ChunkingConfig(strategy="fixed", window_lines=40, overlap_lines=10))
    assert chunks
    assert all(c.file == "fastapi/routing.py" for c in chunks)


def test_ingest_embeds_and_stores(tmp_path: Path):
    root, src = _make_corpus(tmp_path)
    store = ChromaStore(collection_name="ingest-test")
    n = ingest(src, root, ChunkingConfig(strategy="fixed", window_lines=40, overlap_lines=10),
               FakeEmbeddingClient(), store, batch_size=2)
    assert n >= 1
    hits = store.query(FakeEmbeddingClient().embed(["line1"])[0], n=1)
    assert hits and hits[0][0].file == "fastapi/routing.py"


def test_build_chunks_ast_strategy(tmp_path):
    from code_rag_eval.config import ChunkingConfig
    from code_rag_eval.ingest.pipeline import build_chunks
    root = tmp_path / "fastapi"
    src = root / "fastapi"
    src.mkdir(parents=True)
    (src / "m.py").write_text("def alpha():\n    return 1\n\n\nclass C:\n    def beta(self):\n        return 2\n", encoding="utf-8")
    chunks = build_chunks(src, root, ChunkingConfig(strategy="ast"))
    syms = {c.symbol for c in chunks}
    assert "alpha" in syms and "C.beta" in syms
    assert all(c.file == "fastapi/m.py" for c in chunks)
