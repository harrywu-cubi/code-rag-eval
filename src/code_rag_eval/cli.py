from __future__ import annotations
import typer

app = typer.Typer(help="code-rag-eval: code Q&A RAG with an evaluation harness")


@app.command()
def ingest(config: str = "configs/baseline.yaml") -> None:
    """Chunk + embed the corpus into a vector store. (wired in Task 14)"""
    typer.echo(f"ingest stub: {config}")


@app.command()
def ask(question: str, config: str = "configs/baseline.yaml") -> None:
    """Answer a question against the ingested corpus. (wired in Task 14)"""
    typer.echo(f"ask stub: {question}")


@app.command()
def eval(config: str = "configs/baseline.yaml") -> None:
    """Run the evaluation harness for a config. (wired in Task 20)"""
    typer.echo(f"eval stub: {config}")


@app.command()
def experiment() -> None:
    """Sweep the full config matrix. (Phase 5 plan)"""
    typer.echo("experiment stub")


if __name__ == "__main__":
    app()
