from code_rag_eval.types import Chunk, RetrievedChunk
from code_rag_eval.eval.dataset import EvalRecord
from code_rag_eval.generate.answer import Answer, Citation
from code_rag_eval.eval.generation_metrics import (
    context_precision, faithfulness, answer_relevancy, citation_accuracy,
)
from tests.fakes import FakeJudge


def _retrieved():
    return [
        RetrievedChunk(chunk=Chunk(text="def a(): ...", file="a.py", start_line=10, end_line=20), score=1.0, rank=1),
        RetrievedChunk(chunk=Chunk(text="def b(): ...", file="b.py", start_line=1, end_line=5), score=0.5, rank=2),
    ]


def _record():
    return EvalRecord(id="q", category="locate", question="where is a?",
                      gold_symbols=["m.a"], gold_files=["a.py"], gold_line_ranges=[(10, 20)],
                      reference_answer="in a.py")


def test_context_precision_uses_judge():
    # judge says everything relevant -> precision 1.0
    assert context_precision("q", _retrieved(), FakeJudge(relevant=True)) == 1.0
    assert context_precision("q", _retrieved(), FakeJudge(relevant=False)) == 0.0


def test_faithfulness_and_relevancy_passthrough_judge():
    ans = Answer(text="see a.py:12", citations=[Citation("a.py", 12)])
    assert faithfulness(ans, _retrieved(), FakeJudge(faithful=0.8)) == 0.8
    assert answer_relevancy("q", ans, FakeJudge(relevancy=0.7)) == 0.7


def test_citation_accuracy_fraction_within_gold_ranges():
    ans = Answer(text="a.py:12 and b.py:99", citations=[Citation("a.py", 12), Citation("b.py", 99)])
    # a.py:12 is inside gold (10-20); b.py:99 is not a gold file -> 1/2
    assert citation_accuracy(ans, _record()) == 0.5
    empty = Answer(text="no cites", citations=[])
    assert citation_accuracy(empty, _record()) == 0.0
