from typer.testing import CliRunner
from code_rag_eval.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "ask", "eval", "experiment"):
        assert cmd in result.output
