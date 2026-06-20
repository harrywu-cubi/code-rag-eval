import json
from code_rag_eval.eval.symbols import Symbol
from code_rag_eval.eval.generate_questions import draft_record


class _ScriptedLLM:
    def complete(self, system: str, user: str) -> str:
        return json.dumps({
            "question": "Where is get_route_handler defined?",
            "reference_answer": "In APIRoute.get_route_handler in fastapi/routing.py.",
        })


def test_draft_record_builds_eval_record_from_symbol():
    sym = Symbol(
        qualified_name="fastapi.routing.APIRoute.get_route_handler",
        file="fastapi/routing.py", start_line=120, end_line=180,
        kind="function", signature="def get_route_handler(self)", docstring="builds handler",
    )
    rec = draft_record(sym, "locate", _ScriptedLLM(), idx=1)
    assert rec.id == "locate-0001"
    assert rec.category == "locate"
    assert rec.gold_symbols == ["fastapi.routing.APIRoute.get_route_handler"]
    assert rec.gold_files == ["fastapi/routing.py"]
    assert rec.gold_line_ranges == [(120, 180)]
    assert "get_route_handler" in rec.question
