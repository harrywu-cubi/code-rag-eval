from pathlib import Path
from code_rag_eval.eval.dataset import EvalRecord, load_eval_set, save_eval_set


def test_eval_record_round_trips_jsonl(tmp_path: Path):
    recs = [
        EvalRecord(
            id="q1",
            category="locate",
            question="Where is the route handler built?",
            gold_symbols=["fastapi.routing.APIRoute.get_route_handler"],
            gold_files=["fastapi/routing.py"],
            gold_line_ranges=[(120, 180)],
            reference_answer="In APIRoute.get_route_handler.",
        )
    ]
    p = tmp_path / "q.jsonl"
    save_eval_set(recs, p)
    loaded = load_eval_set(p)
    assert loaded == recs
    assert loaded[0].gold_line_ranges == [(120, 180)]


def test_invalid_category_rejected():
    import pytest
    with pytest.raises(Exception):
        EvalRecord(id="x", category="nope", question="q",
                   gold_symbols=[], gold_files=[], gold_line_ranges=[], reference_answer="a")
