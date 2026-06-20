from pathlib import Path
import json
from code_rag_eval.eval.report import comparison_table, pick_winner, load_results, write_decisions

_RESULTS = [
    {"config": "fixed_openai_vector", "aggregate": {"hit_at_5": 0.60, "mrr": 0.50, "faithfulness": 0.70, "citation_accuracy": 0.6}},
    {"config": "ast_voyage_hybrid", "aggregate": {"hit_at_5": 0.84, "mrr": 0.71, "faithfulness": 0.88, "citation_accuracy": 0.76}},
]


def test_pick_winner_uses_primary_then_tiebreak():
    assert pick_winner(_RESULTS) == "ast_voyage_hybrid"


def test_comparison_table_lists_all_configs():
    table = comparison_table(_RESULTS)
    assert "fixed_openai_vector" in table and "ast_voyage_hybrid" in table
    assert "hit_at_5" in table


def test_load_results_reads_json_dir(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps(
        {"config": "a", "aggregate": {"hit_at_5": 0.5}, "per_record": []}), encoding="utf-8")
    got = load_results(tmp_path)
    assert got and got[0]["config"] == "a"


def test_write_decisions_handles_empty(tmp_path: Path):
    out = tmp_path / "DECISIONS.md"
    write_decisions([], out)
    text = out.read_text(encoding="utf-8")
    assert "DECISIONS" in text
    assert "no experiment results" in text.lower()


def test_write_decisions_with_results(tmp_path: Path):
    out = tmp_path / "DECISIONS.md"
    write_decisions(_RESULTS, out)
    text = out.read_text(encoding="utf-8")
    assert "ast_voyage_hybrid" in text
    assert "| config |" in text.lower()
