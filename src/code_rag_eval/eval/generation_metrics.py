from __future__ import annotations
from code_rag_eval.types import RetrievedChunk
from code_rag_eval.eval.dataset import EvalRecord
from code_rag_eval.generate.answer import Answer
from code_rag_eval.eval.judge import LLMJudge


def context_precision(question: str, retrieved: list[RetrievedChunk], judge: LLMJudge) -> float:
    if not retrieved:
        return 0.0
    relevant = sum(1 for r in retrieved if judge.is_relevant(question, r.chunk.text))
    return relevant / len(retrieved)


def faithfulness(answer: Answer, retrieved: list[RetrievedChunk], judge: LLMJudge) -> float:
    context = "\n\n".join(r.chunk.text for r in retrieved)
    return judge.faithfulness(answer.text, context)


def answer_relevancy(question: str, answer: Answer, judge: LLMJudge) -> float:
    return judge.answer_relevancy(question, answer.text)


def chunk_covers_point(cite_file: str, cite_line: int, gold_file: str, gold_range: tuple[int, int]) -> bool:
    gs, ge = gold_range
    return cite_file == gold_file and gs <= cite_line <= ge


def citation_accuracy(answer: Answer, record: EvalRecord) -> float:
    if not answer.citations:
        return 0.0
    golds = list(zip(record.gold_files, record.gold_line_ranges))
    correct = 0
    for cite in answer.citations:
        if any(chunk_covers_point(cite.file, cite.line, gf, gr) for gf, gr in golds):
            correct += 1
    return correct / len(answer.citations)
