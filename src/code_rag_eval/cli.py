from __future__ import annotations
from functools import partial
from pathlib import Path
import typer
from dotenv import load_dotenv
from code_rag_eval.config import load_config
from code_rag_eval.factories import make_embedding_client, make_llm_client
from code_rag_eval.ingest.store import ChromaStore
from code_rag_eval.ingest.cache import EmbeddingCache, CachedEmbeddingClient
from code_rag_eval.ingest.pipeline import ingest as run_ingest
from code_rag_eval.retrieve.vector import VectorRetriever
from code_rag_eval.generate.answer import generate_answer
from code_rag_eval.eval.generate_questions import collect_symbols, draft_candidates
from code_rag_eval.eval.dataset import load_eval_set
from code_rag_eval.eval.judge import AnthropicJudge
from code_rag_eval.eval.harness import evaluate, aggregate, write_report

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


@app.command("draft-questions")
def draft_questions(config: str = "configs/baseline.yaml",
                    per_category: int = 15,
                    out: str = "data/eval/candidates.jsonl") -> None:
    """Draft candidate eval questions for HUMAN verification (writes candidates.jsonl)."""
    load_dotenv()
    cfg = load_config(config)
    llm = make_llm_client(cfg.generation)
    symbols = collect_symbols(SOURCE_DIR, CORPUS_ROOT)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    recs = draft_candidates(symbols, ["locate", "explain", "trace", "behavior"], llm, per_category, Path(out))
    typer.echo(f"drafted {len(recs)} candidates -> {out} (REVIEW + verify before use)")


@app.command()
def eval(config: str = "configs/baseline.yaml",
         questions: str = "data/eval/questions.jsonl",
         results_dir: str = "results") -> None:
    """Run the evaluation harness for a config over the verified question set."""
    load_dotenv()
    cfg = load_config(config)

    # Validate the question set before constructing any API clients, so missing/empty
    # eval sets fail fast with a clear message instead of a key error or a crash.
    qpath = Path(questions)
    if not qpath.exists():
        typer.echo(
            f"eval set not found: {questions}. Build it with `draft-questions`, "
            f"hand-verify, then save it to {questions} (see README)."
        )
        raise typer.Exit(code=1)
    records = load_eval_set(qpath)
    if not records:
        typer.echo(f"no questions in {questions}; nothing to evaluate.")
        raise typer.Exit(code=1)

    embed = CachedEmbeddingClient(make_embedding_client(cfg.embedding),
                                  EmbeddingCache(".emb_cache.sqlite"))
    llm = make_llm_client(cfg.generation)
    store = ChromaStore(collection_name=cfg.name, persist_dir=CHROMA_DIR)
    retriever = VectorRetriever(store, embed)
    judge = AnthropicJudge(model=cfg.generation.model)
    answer_fn = partial(generate_answer, llm=llm)

    rows = evaluate(records, retriever, answer_fn, judge)
    agg = aggregate(rows)
    out = write_report(cfg.name, rows, agg, Path(results_dir))
    typer.echo(f"evaluated {len(records)} questions -> {out}")
    for key in ("hit_at_1", "hit_at_5", "mrr", "faithfulness", "citation_accuracy"):
        typer.echo(f"  {key}: {agg[key]:.3f}")


@app.command()
def experiment() -> None:
    """Sweep the full config matrix. (Phase 5 plan)"""
    typer.echo("experiment stub")


if __name__ == "__main__":
    app()
