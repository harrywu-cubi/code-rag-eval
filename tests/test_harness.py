from code_rag_eval.types import Chunk, RetrievedChunk
from code_rag_eval.eval.dataset import EvalRecord
from code_rag_eval.generate.answer import Answer, Citation
from code_rag_eval.eval.harness import evaluate, aggregate
from tests.fakes import FakeJudge


class _StubRetriever:
    def retrieve(self, query, k):
        c = Chunk(text="def a(): ...", file="a.py", start_line=10, end_line=20)
        return [RetrievedChunk(chunk=c, score=1.0, rank=1)]


def _answer_fn(question, retrieved):
    return Answer(text="see a.py:12", citations=[Citation("a.py", 12)])


def test_evaluate_produces_per_record_and_aggregate_metrics():
    records = [
        EvalRecord(id="q1", category="locate", question="where is a?",
                   gold_symbols=["m.a"], gold_files=["a.py"], gold_line_ranges=[(10, 20)],
                   reference_answer="in a.py"),
    ]
    per_record = evaluate(records, _StubRetriever(), _answer_fn, FakeJudge(), ks=(1, 3))
    assert per_record[0]["hit_at_1"] == 1
    assert per_record[0]["recall_at_1"] == 1.0
    assert per_record[0]["citation_accuracy"] == 1.0

    agg = aggregate(per_record, ks=(1, 3))
    assert agg["hit_at_1"] == 1.0
    assert agg["mrr"] == 1.0
    assert "faithfulness" in agg and "context_precision" in agg
