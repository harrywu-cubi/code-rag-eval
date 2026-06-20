from typer.testing import CliRunner
from code_rag_eval.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "ask", "eval", "experiment"):
        assert cmd in result.output


def test_eval_errors_when_questions_missing(tmp_path):
    # Missing eval set must fail fast with a clear message, before any API client is built.
    result = runner.invoke(app, ["eval", "--questions", str(tmp_path / "nope.jsonl")])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_eval_exits_when_questions_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["eval", "--questions", str(p)])
    assert result.exit_code != 0
    assert "nothing to evaluate" in result.output.lower()


def test_experiment_dry_run_lists_matrix():
    result = runner.invoke(app, ["experiment", "--dry-run"])
    assert result.exit_code == 0
    assert "fixed_openai_vector" in result.output
    assert "ast_voyage_hybrid" in result.output
