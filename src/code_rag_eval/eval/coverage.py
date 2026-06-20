from __future__ import annotations
from code_rag_eval.types import Chunk


def chunk_covers(chunk: Chunk, gold_file: str, gold_range: tuple[int, int]) -> bool:
    """A chunk covers a gold symbol if same file and line ranges overlap (inclusive)."""
    if chunk.file != gold_file:
        return False
    gs, ge = gold_range
    return chunk.start_line <= ge and gs <= chunk.end_line
