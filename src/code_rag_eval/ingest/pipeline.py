from __future__ import annotations
from pathlib import Path
from code_rag_eval.types import Chunk
from code_rag_eval.config import ChunkingConfig
from code_rag_eval.ingest.walk import iter_python_files
from code_rag_eval.ingest.chunkers import chunk_fixed
from code_rag_eval.paths import relpath


def build_chunks(source_dir: Path, corpus_root: Path, chunking: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_python_files(source_dir):
        text = path.read_text(encoding="utf-8")
        rel = relpath(path, corpus_root)
        if chunking.strategy == "fixed":
            chunks.extend(chunk_fixed(text, rel, chunking.window_lines, chunking.overlap_lines))
        else:
            raise ValueError(f"unknown chunking strategy: {chunking.strategy} (ast lands in Phase 4)")
    return chunks


def ingest(source_dir: Path, corpus_root: Path, chunking: ChunkingConfig,
           embed_client, store, batch_size: int = 100) -> int:
    chunks = build_chunks(source_dir, corpus_root, chunking)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = embed_client.embed([c.text for c in batch])
        store.add(batch, vectors)
    return len(chunks)
