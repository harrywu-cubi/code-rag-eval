from __future__ import annotations
from code_rag_eval.types import RetrievedChunk
from code_rag_eval.eval.dataset import EvalRecord
from code_rag_eval.eval.coverage import chunk_covers


def _gold_pairs(record: EvalRecord) -> list[tuple[str, tuple[int, int]]]:
    return list(zip(record.gold_files, record.gold_line_ranges))


def _covered_gold_indices(retrieved: list[RetrievedChunk], record: EvalRecord, k: int) -> set[int]:
    golds = _gold_pairs(record)
    covered: set[int] = set()
    for r in retrieved[:k]:
        for gi, (gf, gr) in enumerate(golds):
            if chunk_covers(r.chunk, gf, gr):
                covered.add(gi)
    return covered


def hit_at_k(retrieved: list[RetrievedChunk], record: EvalRecord, k: int) -> int:
    return 1 if _covered_gold_indices(retrieved, record, k) else 0


def recall_at_k(retrieved: list[RetrievedChunk], record: EvalRecord, k: int) -> float:
    golds = _gold_pairs(record)
    if not golds:
        return 0.0
    return len(_covered_gold_indices(retrieved, record, k)) / len(golds)


def reciprocal_rank(retrieved: list[RetrievedChunk], record: EvalRecord) -> float:
    golds = _gold_pairs(record)
    for r in retrieved:
        for gf, gr in golds:
            if chunk_covers(r.chunk, gf, gr):
                return 1.0 / r.rank
    return 0.0
