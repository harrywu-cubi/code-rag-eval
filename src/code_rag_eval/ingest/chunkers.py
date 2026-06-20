from __future__ import annotations
from code_rag_eval.types import Chunk


def chunk_fixed(text: str, file: str, window_lines: int, overlap_lines: int) -> list[Chunk]:
    """Deliberately naive: fixed line windows that ignore code structure.

    This is the baseline the AST chunker (Phase 4) is measured against.
    """
    lines = text.splitlines()
    if not lines:
        return []
    step = max(1, window_lines - overlap_lines)
    chunks: list[Chunk] = []
    i = 0
    n = len(lines)
    while i < n:
        window = lines[i:i + window_lines]
        chunks.append(Chunk(
            text="\n".join(window),
            file=file,
            start_line=i + 1,
            end_line=i + len(window),
            kind="fixed",
        ))
        if i + window_lines >= n:
            break
        i += step
    return chunks
