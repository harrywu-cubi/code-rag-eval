from __future__ import annotations
from code_rag_eval.types import RetrievedChunk

SYSTEM = (
    "You are a precise code assistant. Answer the question using ONLY the code "
    "context provided. Cite the source of every claim inline as `file:line` (use a "
    "line number from inside the cited chunk's range). If the context does not "
    "contain the answer, say you cannot find it in the provided code."
)


def build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for r in retrieved:
        c = r.chunk
        blocks.append(f"# {c.file}:{c.start_line}-{c.end_line}\n{c.text}")
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (cite file:line):"
