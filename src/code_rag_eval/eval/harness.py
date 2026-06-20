from __future__ import annotations
import json
from pathlib import Path
from statistics import mean
from typing import Callable
from code_rag_eval.eval.dataset import EvalRecord
from code_rag_eval.generate.answer import Answer
from code_rag_eval.eval.judge import LLMJudge
from code_rag_eval.eval import retrieval_metrics as rm
from code_rag_eval.eval import generation_metrics as gm

AnswerFn = Callable[[str, list], Answer]


def evaluate(records: list[EvalRecord], retriever, answer_fn: AnswerFn,
             judge: LLMJudge, ks: tuple[int, ...] = (1, 3, 5, 10), top_k: int = 10) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        retrieved = retriever.retrieve(rec.question, top_k)
        answer = answer_fn(rec.question, retrieved)
        row: dict = {"id": rec.id, "category": rec.category}
        for k in ks:
            row[f"hit_at_{k}"] = rm.hit_at_k(retrieved, rec, k)
            row[f"recall_at_{k}"] = rm.recall_at_k(retrieved, rec, k)
        row["mrr"] = rm.reciprocal_rank(retrieved, rec)
        row["context_precision"] = gm.context_precision(rec.question, retrieved, judge)
        row["faithfulness"] = gm.faithfulness(answer, retrieved, judge)
        row["answer_relevancy"] = gm.answer_relevancy(rec.question, answer, judge)
        row["citation_accuracy"] = gm.citation_accuracy(answer, rec)
        rows.append(row)
    return rows


def aggregate(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    metric_keys = ["mrr", "context_precision", "faithfulness", "answer_relevancy", "citation_accuracy"]
    for k in ks:
        metric_keys += [f"hit_at_{k}", f"recall_at_{k}"]
    return {key: mean(r[key] for r in rows) for key in metric_keys if rows}


def write_report(config_name: str, rows: list[dict], agg: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": config_name, "aggregate": agg, "per_record": rows}
    out = out_dir / f"{config_name}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
