from __future__ import annotations
from pathlib import Path
import typer
from dotenv import load_dotenv
from code_rag_eval.config import load_config
from code_rag_eval.factories import make_embedding_client, make_llm_client
from code_rag_eval.ingest.store import ChromaStore
from code_rag_eval.ingest.pipeline import ingest as run_ingest
from code_rag_eval.retrieve.vector import VectorRetriever
from code_rag_eval.generate.answer import generate_answer

app = typer.Typer(help="code-rag-eval: code Q&A RAG with an evaluation harness")

CORPUS_ROOT = Path("data/corpus/fastapi")
SOURCE_DIR = CORPUS_ROOT / "fastapi"
CHROMA_DIR = ".chroma"


@app.command()
def ingest(config: str = "configs/baseline.yaml") -> None:
    """Chunk + embed the corpus into a vector store."""
    load_dotenv()
    cfg = load_config(config)
    embed = make_embedding_client(cfg.embedding)
    store = ChromaStore(collection_name=cfg.name, persist_dir=CHROMA_DIR)
    n = run_ingest(SOURCE_DIR, CORPUS_ROOT, cfg.chunking, embed, store)
    typer.echo(f"ingested {n} chunks into collection '{cfg.name}'")


@app.command()
def ask(question: str, config: str = "configs/baseline.yaml") -> None:
    """Answer a question against the ingested corpus."""
    load_dotenv()
    cfg = load_config(config)
    embed = make_embedding_client(cfg.embedding)
    llm = make_llm_client(cfg.generation)
    store = ChromaStore(collection_name=cfg.name, persist_dir=CHROMA_DIR)
    retriever = VectorRetriever(store, embed)
    retrieved = retriever.retrieve(question, cfg.retrieval.top_k)
    answer = generate_answer(question, retrieved, llm)
    typer.echo(answer.text)
    typer.echo("\n--- sources ---")
    for r in retrieved:
        typer.echo(f"  {r.chunk.file}:{r.chunk.start_line}-{r.chunk.end_line} (score {r.score:.3f})")


@app.command()
def eval(config: str = "configs/baseline.yaml") -> None:
    """Run the evaluation harness for a config. (wired in Task 21)"""
    typer.echo(f"eval stub: {config}")


@app.command()
def experiment() -> None:
    """Sweep the full config matrix. (Phase 5 plan)"""
    typer.echo("experiment stub")


if __name__ == "__main__":
    app()
