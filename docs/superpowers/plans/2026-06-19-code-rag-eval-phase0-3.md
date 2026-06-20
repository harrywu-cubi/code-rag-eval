# code-rag-eval — Phases 0–3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working code-Q&A RAG over the FastAPI corpus (baseline pipeline) plus an objective evaluation harness, so retrieval and generation can be measured before any experiment sweep.

**Architecture:** Offline ingest (walk → chunk → embed → Chroma). Online query (embed → vector top-k → prompt → LLM → answer + `file:line` citations). An eval layer scores retrieval (hit-rate@k, recall@k, MRR) and generation (LLM-judged faithfulness/relevancy/context-precision + custom citation accuracy) against a hand-verified question set. Every component sits behind a small interface so tests run offline with fakes and Phase 4 can add AST chunking, a code embedding, and hybrid retrieval by adding implementations — not rewiring.

**Tech Stack:** Python 3.11+, `uv`, `typer` (CLI), `pydantic` v2 (config/records), `chromadb` (vector store), `openai` (embeddings), `anthropic` (LLM/judge), `pyyaml`, `python-dotenv`, `pytest`.

**Scope:** Phases 0–3 only. Phase 4 (AST chunker via tree-sitter, Voyage `voyage-code-3`, BM25+RRF hybrid), Phase 5 (8-config sweep), and Phases 6–7 (DECISIONS.md, UI/polish) are a separate plan written after this baseline is validated.

**Spec:** `docs/superpowers/specs/2026-06-19-code-rag-eval-design.md`

---

## File Structure

```
code-rag-eval/
  pyproject.toml                       # uv project + deps + pytest + console script
  .gitignore
  .env.example
  configs/baseline.yaml                # the Phase-1 experiment config
  scripts/fetch_corpus.sh              # clone + pin FastAPI
  data/
    corpus/                            # FastAPI source (gitignored)
    eval/questions.jsonl               # ~50 hand-verified triples (committed)
  src/code_rag_eval/
    __init__.py
    types.py                           # Chunk, RetrievedChunk dataclasses
    config.py                          # pydantic experiment-config models + loader
    paths.py                           # corpus path helpers + module-prefix mapping
    ingest/
      walk.py                          # find .py files, relative paths
      chunkers.py                      # chunk_fixed (AST added Phase 4)
      embed.py                         # EmbeddingClient protocol + OpenAI impl
      cache.py                         # sqlite embedding cache + cached wrapper
      store.py                         # ChromaStore + Chunk<->metadata mapping
      pipeline.py                      # build_chunks + ingest orchestration
    retrieve/
      vector.py                        # VectorRetriever
    generate/
      prompt.py                        # SYSTEM + build_user_prompt
      answer.py                        # LLMClient protocol, Anthropic impl, citations
    eval/
      symbols.py                       # stdlib-ast symbol enumerator (ground truth)
      dataset.py                       # EvalRecord schema + jsonl load/save
      generate_questions.py            # LLM draft tool (human verifies output)
      coverage.py                      # chunk-covers-gold-symbol (line overlap)
      retrieval_metrics.py             # hit-rate@k, recall@k, MRR
      judge.py                         # LLMJudge protocol + Anthropic judge
      generation_metrics.py            # faithfulness, answer relevancy, context precision, citation accuracy
      harness.py                       # evaluate_config -> aggregated metrics + report
    cli.py                             # typer app: ingest | ask | eval | experiment
  tests/
    fakes.py                           # FakeEmbeddingClient, FakeLLMClient, FakeJudge
    test_*.py                          # one per module
```

**Conventions used throughout:**
- File paths in `Chunk`/`Symbol`/`EvalRecord` are **POSIX-relative to the corpus root** (`data/corpus/fastapi`), e.g. `fastapi/routing.py`.
- Line numbers are **1-based, inclusive** on both ends.
- All tests use fakes from `tests/fakes.py`; no test hits a network or needs an API key.
- Run tests with `uv run pytest`; run the CLI with `uv run code-rag-eval ...`.

---

## Phase 0 — Scaffold

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `src/code_rag_eval/__init__.py`, `tests/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Initialize the uv project and add dependencies**

Run:
```bash
uv init --package --name code-rag-eval .
uv add pydantic pyyaml typer chromadb openai anthropic python-dotenv
uv add --dev pytest
```

- [ ] **Step 2: Overwrite `pyproject.toml` to pin Python, add pytest config, and the console script**

```toml
[project]
name = "code-rag-eval"
version = "0.1.0"
description = "Code Q&A RAG with a rigorous evaluation harness over the FastAPI corpus"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2",
    "pyyaml>=6",
    "typer>=0.12",
    "chromadb>=0.5",
    "openai>=1.40",
    "anthropic>=0.34",
    "python-dotenv>=1",
]

[project.scripts]
code-rag-eval = "code_rag_eval.cli:app"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Create `.gitignore` and `.env.example`**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
data/corpus/
.chroma/
*.sqlite
.pytest_cache/
results/*.json
```

`.env.example`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
# VOYAGE_API_KEY=...   # added in Phase 4
```

- [ ] **Step 4: Write a smoke test**

`tests/test_smoke.py`:
```python
import code_rag_eval


def test_package_imports():
    assert code_rag_eval is not None
```

- [ ] **Step 5: Run the smoke test**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/code_rag_eval/__init__.py tests/__init__.py tests/test_smoke.py uv.lock
git commit -m "chore: scaffold uv project, deps, pytest"
```

---

### Task 2: Config models + loader

**Files:**
- Create: `src/code_rag_eval/config.py`, `configs/baseline.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path
from code_rag_eval.config import ExperimentConfig, load_config


def test_load_baseline_config(tmp_path: Path):
    yaml_text = """
name: baseline
chunking:
  strategy: fixed
  window_lines: 40
  overlap_lines: 10
embedding:
  provider: openai
  model: text-embedding-3-large
retrieval:
  method: vector
  top_k: 5
generation:
  provider: anthropic
  model: claude-sonnet-4-6
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "baseline"
    assert cfg.chunking.strategy == "fixed"
    assert cfg.chunking.window_lines == 40
    assert cfg.embedding.model == "text-embedding-3-large"
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.rrf_k == 60  # default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: code_rag_eval.config)

- [ ] **Step 3: Implement `config.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "ast"] = "fixed"
    window_lines: int = 40       # used by fixed
    overlap_lines: int = 10      # used by fixed


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "voyage"] = "openai"
    model: str = "text-embedding-3-large"


class RetrievalConfig(BaseModel):
    method: Literal["vector", "bm25", "hybrid"] = "vector"
    top_k: int = 5
    rrf_k: int = 60


class GenerationConfig(BaseModel):
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-sonnet-4-6"


class ExperimentConfig(BaseModel):
    name: str
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(**data)
```

- [ ] **Step 4: Create `configs/baseline.yaml`**

```yaml
name: baseline
chunking:
  strategy: fixed
  window_lines: 40
  overlap_lines: 10
embedding:
  provider: openai
  model: text-embedding-3-large
retrieval:
  method: vector
  top_k: 5
generation:
  provider: anthropic
  model: claude-sonnet-4-6
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/code_rag_eval/config.py configs/baseline.yaml tests/test_config.py
git commit -m "feat: experiment config models + yaml loader"
```

---

### Task 3: Corpus paths + module prefix helper

**Files:**
- Create: `src/code_rag_eval/paths.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Write the failing test**

`tests/test_paths.py`:
```python
from pathlib import Path
from code_rag_eval.paths import relpath, module_prefix


def test_relpath_is_posix(tmp_path: Path):
    root = tmp_path / "fastapi"
    f = root / "fastapi" / "routing.py"
    f.parent.mkdir(parents=True)
    f.write_text("x = 1", encoding="utf-8")
    assert relpath(f, root) == "fastapi/routing.py"


def test_module_prefix():
    assert module_prefix("fastapi/routing.py") == "fastapi.routing"
    assert module_prefix("fastapi/__init__.py") == "fastapi"
    assert module_prefix("fastapi/security/oauth2.py") == "fastapi.security.oauth2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_paths.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `paths.py`**

```python
from __future__ import annotations
from pathlib import Path


def relpath(path: Path, corpus_root: Path) -> str:
    return path.resolve().relative_to(corpus_root.resolve()).as_posix()


def module_prefix(rel_path: str) -> str:
    parts = rel_path[:-3].split("/")  # strip ".py"
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_paths.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/paths.py tests/test_paths.py
git commit -m "feat: corpus path + module-prefix helpers"
```

---

### Task 4: CLI skeleton

**Files:**
- Create: `src/code_rag_eval/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from typer.testing import CliRunner
from code_rag_eval.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "ask", "eval", "experiment"):
        assert cmd in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `cli.py` with stub commands**

```python
from __future__ import annotations
import typer

app = typer.Typer(help="code-rag-eval: code Q&A RAG with an evaluation harness")


@app.command()
def ingest(config: str = "configs/baseline.yaml") -> None:
    """Chunk + embed the corpus into a vector store. (wired in Task 13)"""
    typer.echo(f"ingest stub: {config}")


@app.command()
def ask(question: str, config: str = "configs/baseline.yaml") -> None:
    """Answer a question against the ingested corpus. (wired in Task 13)"""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/cli.py tests/test_cli.py
git commit -m "feat: CLI skeleton with stub commands"
```

---

### Task 5: Fetch + pin the FastAPI corpus

**Files:**
- Create: `scripts/fetch_corpus.sh`, `data/corpus/COMMIT.txt` (generated)

- [ ] **Step 1: Write the fetch script**

`scripts/fetch_corpus.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
TAG="${1:-0.115.0}"
DEST="data/corpus/fastapi"
rm -rf "$DEST"
git clone --depth 1 --branch "$TAG" https://github.com/fastapi/fastapi.git "$DEST"
SHA="$(git -C "$DEST" rev-parse HEAD)"
echo "tag=$TAG sha=$SHA" > data/corpus/COMMIT.txt
echo "Pinned FastAPI $TAG @ $SHA"
echo "Source package: $DEST/fastapi"
```

- [ ] **Step 2: Run it**

Run:
```bash
bash scripts/fetch_corpus.sh 0.115.0
```
Expected: prints `Pinned FastAPI 0.115.0 @ <sha>`. If tag `0.115.0` does not exist, re-run with any recent stable tag (e.g. `bash scripts/fetch_corpus.sh 0.115.2`); the exact tag does not matter as long as it is recorded in `COMMIT.txt`.

- [ ] **Step 3: Verify the corpus is present**

Run:
```bash
ls data/corpus/fastapi/fastapi/routing.py && cat data/corpus/COMMIT.txt
```
Expected: the file exists and `COMMIT.txt` shows the pinned tag + SHA.

- [ ] **Step 4: Commit the script + pin record (corpus itself is gitignored)**

```bash
git add scripts/fetch_corpus.sh data/corpus/COMMIT.txt
git commit -m "chore: fetch + pin FastAPI corpus"
```

---

## Phase 1 — Baseline pipeline

### Task 6: Core types

**Files:**
- Create: `src/code_rag_eval/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

`tests/test_types.py`:
```python
from code_rag_eval.types import Chunk, RetrievedChunk


def test_chunk_id_is_stable_and_unique():
    c = Chunk(text="x", file="fastapi/routing.py", start_line=10, end_line=20)
    assert c.chunk_id == "fastapi/routing.py:10-20"
    assert c.kind == "fixed"


def test_retrieved_chunk_holds_score_and_rank():
    c = Chunk(text="x", file="a.py", start_line=1, end_line=2)
    rc = RetrievedChunk(chunk=c, score=0.9, rank=1)
    assert rc.score == 0.9 and rc.rank == 1 and rc.chunk is c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `types.py`**

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    file: str
    start_line: int
    end_line: int
    kind: str = "fixed"               # "fixed" | "function" | "class"
    symbol: str | None = None
    signature: str | None = None
    docstring: str | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.file}:{self.start_line}-{self.end_line}"


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/types.py tests/test_types.py
git commit -m "feat: Chunk and RetrievedChunk types"
```

---

### Task 7: File walker

**Files:**
- Create: `src/code_rag_eval/ingest/__init__.py`, `src/code_rag_eval/ingest/walk.py`
- Test: `tests/test_walk.py`

- [ ] **Step 1: Write the failing test**

`tests/test_walk.py`:
```python
from pathlib import Path
from code_rag_eval.ingest.walk import iter_python_files


def test_iter_python_files_skips_pycache(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("a = 1", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("b = 2", encoding="utf-8")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "a.cpython.pyc").write_text("x", encoding="utf-8")
    (tmp_path / "pkg" / "notes.md").write_text("hi", encoding="utf-8")
    files = iter_python_files(tmp_path)
    names = [f.name for f in files]
    assert names == ["a.py", "b.py"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_walk.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `walk.py` (and empty `ingest/__init__.py`)**

`src/code_rag_eval/ingest/__init__.py`: (empty file)

`src/code_rag_eval/ingest/walk.py`:
```python
from __future__ import annotations
from pathlib import Path


def iter_python_files(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.rglob("*.py")
        if "__pycache__" not in p.parts
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_walk.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/ingest/__init__.py src/code_rag_eval/ingest/walk.py tests/test_walk.py
git commit -m "feat: python file walker"
```

---

### Task 8: Fixed-size chunker

**Files:**
- Create: `src/code_rag_eval/ingest/chunkers.py`
- Test: `tests/test_chunkers.py`

- [ ] **Step 1: Write the failing test**

`tests/test_chunkers.py`:
```python
from code_rag_eval.ingest.chunkers import chunk_fixed


def test_fixed_windows_with_overlap():
    text = "\n".join(f"line{i}" for i in range(1, 11))  # 10 lines
    chunks = chunk_fixed(text, "a.py", window_lines=4, overlap_lines=1)
    # step = 3 -> windows starting at lines 1, 4, 7
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 4), (4, 7), (7, 10)]
    assert chunks[0].text.splitlines()[0] == "line1"
    assert all(c.kind == "fixed" and c.file == "a.py" for c in chunks)


def test_fixed_handles_short_and_empty():
    assert chunk_fixed("", "a.py", 40, 10) == []
    short = chunk_fixed("only one line", "a.py", 40, 10)
    assert len(short) == 1 and short[0].start_line == 1 and short[0].end_line == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chunkers.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `chunkers.py`**

```python
from __future__ import annotations
from code_rag_eval.types import Chunk


def chunk_fixed(text: str, file: str, window_lines: int, overlap_lines: int) -> list[Chunk]:
    """Deliberately naive: fixed line windows that ignore code structure.

    This is the baseline the AST chunker (Phase 4) is measured against.
    """
    lines = text.splitlines()
    if not lines:
        return []
    step = max(1, window_lines - overlap_lines)
    chunks: list[Chunk] = []
    i = 0
    n = len(lines)
    while i < n:
        window = lines[i:i + window_lines]
        chunks.append(Chunk(
            text="\n".join(window),
            file=file,
            start_line=i + 1,
            end_line=i + len(window),
            kind="fixed",
        ))
        if i + window_lines >= n:
            break
        i += step
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chunkers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/ingest/chunkers.py tests/test_chunkers.py
git commit -m "feat: naive fixed-size line-window chunker"
```

---

### Task 9: Embedding client + test fakes

**Files:**
- Create: `src/code_rag_eval/ingest/embed.py`, `tests/fakes.py`
- Test: `tests/test_embed.py`

- [ ] **Step 1: Write `tests/fakes.py` and the failing test**

`tests/fakes.py`:
```python
from __future__ import annotations
import hashlib


class FakeEmbeddingClient:
    """Deterministic small-dim embeddings derived from text bytes. No network."""

    def __init__(self, model: str = "fake-embed", dim: int = 16):
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [h[i % len(h)] / 255.0 for i in range(self.dim)]
            out.append(vec)
        return out


class FakeLLMClient:
    """Returns a canned answer; records the last prompt it received."""

    def __init__(self, answer: str = "See fastapi/routing.py:1 for details."):
        self.answer = answer
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self.answer


class FakeJudge:
    """Returns fixed verdicts for judge-based metrics."""

    def __init__(self, relevant: bool = True, faithful: float = 1.0, relevancy: float = 1.0):
        self._relevant = relevant
        self._faithful = faithful
        self._relevancy = relevancy

    def is_relevant(self, question: str, chunk_text: str) -> bool:
        return self._relevant

    def faithfulness(self, answer: str, context: str) -> float:
        return self._faithful

    def answer_relevancy(self, question: str, answer: str) -> float:
        return self._relevancy
```

`tests/test_embed.py`:
```python
from code_rag_eval.ingest.embed import EmbeddingClient
from tests.fakes import FakeEmbeddingClient


def test_fake_embed_is_deterministic_and_right_shape():
    client: EmbeddingClient = FakeEmbeddingClient(dim=16)
    a1 = client.embed(["hello"])[0]
    a2 = client.embed(["hello"])[0]
    b = client.embed(["world"])[0]
    assert len(a1) == 16
    assert a1 == a2          # deterministic
    assert a1 != b           # content-sensitive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embed.py -v`
Expected: FAIL (ImportError: EmbeddingClient)

- [ ] **Step 3: Implement `embed.py`**

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingClient:
    """Thin wrapper over the OpenAI embeddings API. Exercised manually / via CLI."""

    def __init__(self, model: str = "text-embedding-3-large"):
        from openai import OpenAI
        self._client = OpenAI()
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/ingest/embed.py tests/fakes.py tests/test_embed.py
git commit -m "feat: embedding client protocol + OpenAI impl + test fakes"
```

---

### Task 10: Chroma vector store

**Files:**
- Create: `src/code_rag_eval/ingest/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

`tests/test_store.py`:
```python
from code_rag_eval.types import Chunk
from code_rag_eval.ingest.store import ChromaStore


def test_add_then_query_round_trips_chunk():
    store = ChromaStore(collection_name="t")  # ephemeral, in-memory
    chunks = [
        Chunk(text="def login(): ...", file="auth.py", start_line=1, end_line=3, symbol="auth.login"),
        Chunk(text="def logout(): ...", file="auth.py", start_line=5, end_line=7, symbol="auth.logout"),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    store.add(chunks, vectors)
    hits = store.query([1.0, 0.0], n=1)
    assert len(hits) == 1
    chunk, score = hits[0]
    assert chunk.file == "auth.py"
    assert chunk.symbol == "auth.login"
    assert chunk.start_line == 1 and chunk.end_line == 3
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `store.py`**

```python
from __future__ import annotations
from typing import Protocol
import chromadb
from code_rag_eval.types import Chunk


def chunk_to_metadata(c: Chunk) -> dict:
    return {
        "file": c.file,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "kind": c.kind,
        "symbol": c.symbol or "",
        "signature": c.signature or "",
        "docstring": c.docstring or "",
    }


def metadata_to_chunk(text: str, m: dict) -> Chunk:
    return Chunk(
        text=text,
        file=str(m["file"]),
        start_line=int(m["start_line"]),
        end_line=int(m["end_line"]),
        kind=str(m.get("kind", "fixed")),
        symbol=(m.get("symbol") or None),
        signature=(m.get("signature") or None),
        docstring=(m.get("docstring") or None),
    )


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def query(self, vector: list[float], n: int) -> list[tuple[Chunk, float]]: ...


class ChromaStore:
    def __init__(self, collection_name: str, persist_dir: str | None = None):
        self._client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir else chromadb.EphemeralClient()
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[chunk_to_metadata(c) for c in chunks],
        )

    def query(self, vector: list[float], n: int) -> list[tuple[Chunk, float]]:
        res = self._col.query(query_embeddings=[vector], n_results=n)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            (metadata_to_chunk(doc, meta), 1.0 - float(dist))
            for doc, meta, dist in zip(docs, metas, dists)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/ingest/store.py tests/test_store.py
git commit -m "feat: Chroma vector store with chunk<->metadata mapping"
```

---

### Task 11: Ingest pipeline

**Files:**
- Create: `src/code_rag_eval/ingest/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
from pathlib import Path
from code_rag_eval.config import ChunkingConfig
from code_rag_eval.ingest.pipeline import build_chunks, ingest
from code_rag_eval.ingest.store import ChromaStore
from tests.fakes import FakeEmbeddingClient


def _make_corpus(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "fastapi"
    src = root / "fastapi"
    src.mkdir(parents=True)
    (src / "routing.py").write_text("\n".join(f"line{i}" for i in range(1, 51)), encoding="utf-8")
    return root, src


def test_build_chunks_uses_posix_relative_paths(tmp_path: Path):
    root, src = _make_corpus(tmp_path)
    chunks = build_chunks(src, root, ChunkingConfig(strategy="fixed", window_lines=40, overlap_lines=10))
    assert chunks
    assert all(c.file == "fastapi/routing.py" for c in chunks)


def test_ingest_embeds_and_stores(tmp_path: Path):
    root, src = _make_corpus(tmp_path)
    store = ChromaStore(collection_name="ingest-test")
    n = ingest(src, root, ChunkingConfig(strategy="fixed", window_lines=40, overlap_lines=10),
               FakeEmbeddingClient(), store, batch_size=2)
    assert n >= 1
    hits = store.query(FakeEmbeddingClient().embed(["line1"])[0], n=1)
    assert hits and hits[0][0].file == "fastapi/routing.py"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `pipeline.py`**

```python
from __future__ import annotations
from pathlib import Path
from code_rag_eval.types import Chunk
from code_rag_eval.config import ChunkingConfig
from code_rag_eval.ingest.walk import iter_python_files
from code_rag_eval.ingest.chunkers import chunk_fixed
from code_rag_eval.paths import relpath


def build_chunks(source_dir: Path, corpus_root: Path, chunking: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_python_files(source_dir):
        text = path.read_text(encoding="utf-8")
        rel = relpath(path, corpus_root)
        if chunking.strategy == "fixed":
            chunks.extend(chunk_fixed(text, rel, chunking.window_lines, chunking.overlap_lines))
        else:
            raise ValueError(f"unknown chunking strategy: {chunking.strategy} (ast lands in Phase 4)")
    return chunks


def ingest(source_dir: Path, corpus_root: Path, chunking: ChunkingConfig,
           embed_client, store, batch_size: int = 100) -> int:
    chunks = build_chunks(source_dir, corpus_root, chunking)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = embed_client.embed([c.text for c in batch])
        store.add(batch, vectors)
    return len(chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/ingest/pipeline.py tests/test_pipeline.py
git commit -m "feat: ingest pipeline (walk -> chunk -> embed -> store)"
```

---

### Task 12: Vector retriever

**Files:**
- Create: `src/code_rag_eval/retrieve/__init__.py`, `src/code_rag_eval/retrieve/vector.py`
- Test: `tests/test_vector_retriever.py`

- [ ] **Step 1: Write the failing test**

`tests/test_vector_retriever.py`:
```python
from code_rag_eval.types import Chunk
from code_rag_eval.ingest.store import ChromaStore
from code_rag_eval.retrieve.vector import VectorRetriever
from tests.fakes import FakeEmbeddingClient


def test_retrieve_returns_ranked_chunks():
    embed = FakeEmbeddingClient()
    store = ChromaStore(collection_name="retr-test")
    chunks = [
        Chunk(text="def login(): pass", file="auth.py", start_line=1, end_line=1),
        Chunk(text="def logout(): pass", file="auth.py", start_line=2, end_line=2),
        Chunk(text="def render(): pass", file="view.py", start_line=1, end_line=1),
    ]
    store.add(chunks, embed.embed([c.text for c in chunks]))
    retr = VectorRetriever(store, embed)
    results = retr.retrieve("def login(): pass", k=2)
    assert len(results) == 2
    assert results[0].rank == 1 and results[1].rank == 2
    assert results[0].chunk.text == "def login(): pass"  # identical query -> top hit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vector_retriever.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `vector.py` (and empty `retrieve/__init__.py`)**

`src/code_rag_eval/retrieve/__init__.py`: (empty file)

`src/code_rag_eval/retrieve/vector.py`:
```python
from __future__ import annotations
from code_rag_eval.types import RetrievedChunk


class VectorRetriever:
    def __init__(self, store, embed_client):
        self._store = store
        self._embed = embed_client

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        qv = self._embed.embed([query])[0]
        hits = self._store.query(qv, k)
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=i + 1)
            for i, (chunk, score) in enumerate(hits)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vector_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/retrieve/__init__.py src/code_rag_eval/retrieve/vector.py tests/test_vector_retriever.py
git commit -m "feat: vector retriever"
```

---

### Task 13: Prompt assembly + answer/citations

**Files:**
- Create: `src/code_rag_eval/generate/__init__.py`, `src/code_rag_eval/generate/prompt.py`, `src/code_rag_eval/generate/answer.py`
- Test: `tests/test_generate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_generate.py`:
```python
from code_rag_eval.types import Chunk, RetrievedChunk
from code_rag_eval.generate.prompt import build_user_prompt, SYSTEM
from code_rag_eval.generate.answer import extract_citations, generate_answer, Citation
from tests.fakes import FakeLLMClient


def _retrieved():
    c = Chunk(text="def get_route_handler(): ...", file="fastapi/routing.py", start_line=120, end_line=180)
    return [RetrievedChunk(chunk=c, score=0.9, rank=1)]


def test_prompt_includes_file_line_header_and_question():
    prompt = build_user_prompt("Where is the route handler?", _retrieved())
    assert "# fastapi/routing.py:120-180" in prompt
    assert "Where is the route handler?" in prompt
    assert "def get_route_handler" in prompt


def test_extract_citations_dedupes():
    text = "See fastapi/routing.py:120 and again fastapi/routing.py:120 plus fastapi/params.py:5."
    cites = extract_citations(text)
    assert Citation("fastapi/routing.py", 120) in cites
    assert Citation("fastapi/params.py", 5) in cites
    assert len(cites) == 2


def test_generate_answer_calls_llm_and_parses_citations():
    llm = FakeLLMClient(answer="Defined at fastapi/routing.py:120.")
    ans = generate_answer("Where?", _retrieved(), llm)
    assert ans.text == "Defined at fastapi/routing.py:120."
    assert ans.citations == [Citation("fastapi/routing.py", 120)]
    assert llm.last_system == SYSTEM
    assert "fastapi/routing.py:120-180" in llm.last_user
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `prompt.py` and `answer.py` (and empty `generate/__init__.py`)**

`src/code_rag_eval/generate/__init__.py`: (empty file)

`src/code_rag_eval/generate/prompt.py`:
```python
from __future__ import annotations
from code_rag_eval.types import RetrievedChunk

SYSTEM = (
    "You are a precise code assistant. Answer the question using ONLY the code "
    "context provided. Cite the source of every claim inline as `file:line` (use a "
    "line number from inside the cited chunk's range). If the context does not "
    "contain the answer, say you cannot find it in the provided code."
)


def build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for r in retrieved:
        c = r.chunk
        blocks.append(f"# {c.file}:{c.start_line}-{c.end_line}\n{c.text}")
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (cite file:line):"
```

`src/code_rag_eval/generate/answer.py`:
```python
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Protocol
from code_rag_eval.types import RetrievedChunk
from code_rag_eval.generate.prompt import SYSTEM, build_user_prompt


@dataclass(frozen=True)
class Citation:
    file: str
    line: int


@dataclass
class Answer:
    text: str
    citations: list[Citation]


_CITE = re.compile(r"([\w./\\-]+\.py):(\d+)")


def extract_citations(text: str) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    out: list[Citation] = []
    for m in _CITE.finditer(text):
        f = m.group(1).replace("\\", "/")
        ln = int(m.group(2))
        if (f, ln) not in seen:
            seen.add((f, ln))
            out.append(Citation(file=f, line=ln))
    return out


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API. Exercised manually / via CLI."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        from anthropic import Anthropic
        self._client = Anthropic()
        self.model = model

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def generate_answer(question: str, retrieved: list[RetrievedChunk], llm: LLMClient) -> Answer:
    user = build_user_prompt(question, retrieved)
    text = llm.complete(SYSTEM, user)
    return Answer(text=text, citations=extract_citations(text))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/generate/__init__.py src/code_rag_eval/generate/prompt.py src/code_rag_eval/generate/answer.py tests/test_generate.py
git commit -m "feat: prompt assembly + LLM answer with file:line citations"
```

---

### Task 14: Wire CLI `ingest` and `ask` + client factories

**Files:**
- Create: `src/code_rag_eval/factories.py`
- Modify: `src/code_rag_eval/cli.py`
- Test: `tests/test_factories.py`

- [ ] **Step 1: Write the failing test (factory selects the right impl by config)**

`tests/test_factories.py`:
```python
import pytest
from code_rag_eval.config import EmbeddingConfig, GenerationConfig
from code_rag_eval.factories import make_embedding_client, make_llm_client


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        make_embedding_client(EmbeddingConfig(provider="voyage", model="voyage-code-3"))
    with pytest.raises(ValueError):
        make_llm_client(GenerationConfig(provider="openai", model="gpt-x"))


def test_openai_embedding_selected(monkeypatch):
    created = {}

    class _Stub:
        def __init__(self, model):
            created["model"] = model
            self.model = model

    monkeypatch.setattr("code_rag_eval.factories.OpenAIEmbeddingClient", _Stub)
    client = make_embedding_client(EmbeddingConfig(provider="openai", model="text-embedding-3-large"))
    assert created["model"] == "text-embedding-3-large"
    assert client.model == "text-embedding-3-large"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_factories.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `factories.py`**

```python
from __future__ import annotations
from code_rag_eval.config import EmbeddingConfig, GenerationConfig
from code_rag_eval.ingest.embed import OpenAIEmbeddingClient
from code_rag_eval.generate.answer import AnthropicClient


def make_embedding_client(cfg: EmbeddingConfig):
    if cfg.provider == "openai":
        return OpenAIEmbeddingClient(model=cfg.model)
    raise ValueError(f"embedding provider not available until Phase 4: {cfg.provider}")


def make_llm_client(cfg: GenerationConfig):
    if cfg.provider == "anthropic":
        return AnthropicClient(model=cfg.model)
    raise ValueError(f"llm provider not supported: {cfg.provider}")
```

- [ ] **Step 4: Run the factory test**

Run: `uv run pytest tests/test_factories.py -v`
Expected: PASS

- [ ] **Step 5: Wire the CLI `ingest` and `ask` commands**

Replace the **entire contents** of `src/code_rag_eval/cli.py` with the following (this keeps `app` defined exactly once, implements `ingest`/`ask`, and leaves `eval`/`experiment` as stubs to be wired in Tasks 21/Phase 5):
```python
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
```
Note: Tasks 17 and 21 add/replace commands in this file; keep `app` defined only here.

- [ ] **Step 6: Confirm the CLI still parses + run baseline end-to-end manually**

Run (offline test still passes):
```bash
uv run pytest tests/test_cli.py -v
```
Expected: PASS

Then, with `data/corpus/fastapi` present and a `.env` containing `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` (copy from `.env.example`):
```bash
uv run code-rag-eval ingest
uv run code-rag-eval ask "Where is the APIRoute request handler built?"
```
Expected: `ingest` prints a chunk count; `ask` prints an answer containing at least one `fastapi/....py:<line>` citation followed by a sources list. (This step costs real API tokens.)

- [ ] **Step 7: Commit**

```bash
git add src/code_rag_eval/factories.py src/code_rag_eval/cli.py tests/test_factories.py
git commit -m "feat: wire CLI ingest + ask with client factories"
```

**Phase 1 checkpoint:** the system answers FastAPI questions with `file:line` citations. Pause for review before Phase 2.

---

## Phase 2 — Evaluation set

### Task 15: Symbol enumerator (ground-truth source)

**Files:**
- Create: `src/code_rag_eval/eval/__init__.py`, `src/code_rag_eval/eval/symbols.py`
- Test: `tests/test_symbols.py`

- [ ] **Step 1: Write the failing test**

`tests/test_symbols.py`:
```python
from code_rag_eval.eval.symbols import enumerate_symbols

SRC = '''\
def top():
    """top docstring"""
    return 1


class Service:
    """svc"""
    def handle(self, x):
        return x
'''


def test_enumerate_symbols_qualified_names_and_ranges():
    syms = enumerate_symbols(SRC, "fastapi/routing.py", "fastapi.routing")
    by_name = {s.qualified_name: s for s in syms}
    assert "fastapi.routing.top" in by_name
    assert "fastapi.routing.Service" in by_name
    assert "fastapi.routing.Service.handle" in by_name
    top = by_name["fastapi.routing.top"]
    assert top.kind == "function"
    assert top.start_line == 1
    assert top.docstring == "top docstring"
    assert by_name["fastapi.routing.Service.handle"].signature == "def handle(self, x)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_symbols.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `symbols.py` (and empty `eval/__init__.py`)**

`src/code_rag_eval/eval/__init__.py`: (empty file)

`src/code_rag_eval/eval/symbols.py`:
```python
from __future__ import annotations
import ast
from dataclasses import dataclass


@dataclass
class Symbol:
    qualified_name: str
    file: str
    start_line: int
    end_line: int
    kind: str            # "function" | "class"
    signature: str
    docstring: str | None


def _signature(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    args = [a.arg for a in node.args.args]  # type: ignore[attr-defined]
    return f"def {node.name}({', '.join(args)})"  # type: ignore[attr-defined]


def enumerate_symbols(text: str, file: str, module_prefix: str) -> list[Symbol]:
    tree = ast.parse(text)
    out: list[Symbol] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qn = f"{prefix}.{child.name}"
                kind = "class" if isinstance(child, ast.ClassDef) else "function"
                out.append(Symbol(
                    qualified_name=qn,
                    file=file,
                    start_line=child.lineno,
                    end_line=child.end_lineno or child.lineno,
                    kind=kind,
                    signature=_signature(child),
                    docstring=ast.get_docstring(child),
                ))
                visit(child, qn)

    visit(tree, module_prefix)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_symbols.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/eval/__init__.py src/code_rag_eval/eval/symbols.py tests/test_symbols.py
git commit -m "feat: stdlib-ast symbol enumerator for ground truth"
```

---

### Task 16: Eval record schema + jsonl IO

**Files:**
- Create: `src/code_rag_eval/eval/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dataset.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `dataset.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel

Category = Literal["locate", "explain", "trace", "behavior"]


class EvalRecord(BaseModel):
    id: str
    category: Category
    question: str
    gold_symbols: list[str]
    gold_files: list[str]
    gold_line_ranges: list[tuple[int, int]]
    reference_answer: str


def load_eval_set(path: str | Path) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(EvalRecord(**json.loads(line)))
    return records


def save_eval_set(records: list[EvalRecord], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/eval/dataset.py tests/test_dataset.py
git commit -m "feat: eval record schema + jsonl load/save"
```

---

### Task 17: Question draft-generation tool

**Files:**
- Create: `src/code_rag_eval/eval/generate_questions.py`
- Modify: `src/code_rag_eval/cli.py` (add `draft-questions` command)
- Test: `tests/test_generate_questions.py`

- [ ] **Step 1: Write the failing test**

`tests/test_generate_questions.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_questions.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `generate_questions.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
from code_rag_eval.eval.symbols import Symbol, enumerate_symbols
from code_rag_eval.eval.dataset import EvalRecord, Category, save_eval_set
from code_rag_eval.ingest.walk import iter_python_files
from code_rag_eval.paths import relpath, module_prefix

_PROMPTS: dict[str, str] = {
    "locate": "Write a 'where is X defined' question whose answer is exactly this symbol.",
    "explain": "Write a 'how does X work' question answerable from this symbol's code.",
    "trace": "Write a 'what calls or uses X' question centered on this symbol.",
    "behavior": "Write a 'what does X return / do when ...' question about this symbol.",
}

_SYSTEM = (
    "You draft code-comprehension eval questions. Given one Python symbol, return a "
    "JSON object with keys 'question' and 'reference_answer'. The question must be "
    "answerable from the symbol's source. Mention the symbol's short name in the question. "
    "Return ONLY the JSON object."
)


def draft_record(symbol: Symbol, category: Category, llm, idx: int) -> EvalRecord:
    user = (
        f"{_PROMPTS[category]}\n\n"
        f"Symbol: {symbol.qualified_name}\n"
        f"Signature: {symbol.signature}\n"
        f"File: {symbol.file} (lines {symbol.start_line}-{symbol.end_line})\n"
        f"Docstring: {symbol.docstring or '(none)'}"
    )
    raw = llm.complete(_SYSTEM, user)
    data = json.loads(raw)
    return EvalRecord(
        id=f"{category}-{idx:04d}",
        category=category,
        question=data["question"],
        gold_symbols=[symbol.qualified_name],
        gold_files=[symbol.file],
        gold_line_ranges=[(symbol.start_line, symbol.end_line)],
        reference_answer=data["reference_answer"],
    )


def collect_symbols(source_dir: Path, corpus_root: Path, min_lines: int = 5) -> list[Symbol]:
    """Enumerate symbols worth asking about (skip trivial one-liners)."""
    syms: list[Symbol] = []
    for path in iter_python_files(source_dir):
        text = path.read_text(encoding="utf-8")
        rel = relpath(path, corpus_root)
        for s in enumerate_symbols(text, rel, module_prefix(rel)):
            if (s.end_line - s.start_line + 1) >= min_lines:
                syms.append(s)
    return syms


def draft_candidates(symbols: list[Symbol], categories: list[Category], llm,
                     per_category: int, out_path: Path) -> list[EvalRecord]:
    """Draft candidate records (NOT verified). Output is for HUMAN review/editing."""
    records: list[EvalRecord] = []
    idx = 1
    for cat in categories:
        chosen = symbols[:per_category]  # deterministic slice; reviewer curates
        for sym in chosen:
            records.append(draft_record(sym, cat, llm, idx))
            idx += 1
    save_eval_set(records, out_path)
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate_questions.py -v`
Expected: PASS

- [ ] **Step 5: Add a `draft-questions` CLI command**

Add to `src/code_rag_eval/cli.py` (imports at top, command below the others):
```python
from code_rag_eval.eval.generate_questions import collect_symbols, draft_candidates


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
```

- [ ] **Step 6: Run the offline tests, then build the verified eval set (manual)**

Run: `uv run pytest -q`
Expected: PASS (whole suite)

Then, with corpus + `.env` present:
```bash
uv run code-rag-eval draft-questions --per-category 15
```
This produces `data/eval/candidates.jsonl`. **Manually review every record**: fix wording, confirm the gold symbol/file/line range is correct, drop bad ones, and balance categories. Save the curated, verified set to `data/eval/questions.jsonl` (target ~50 records, hand-verified). The candidates file stays gitignored; `questions.jsonl` is committed.

- [ ] **Step 7: Commit the tool + the verified eval set**

```bash
git add src/code_rag_eval/eval/generate_questions.py src/code_rag_eval/cli.py tests/test_generate_questions.py data/eval/questions.jsonl
git commit -m "feat: question draft tool + verified 50-question eval set"
```

**Phase 2 checkpoint:** `data/eval/questions.jsonl` holds ~50 hand-verified triples. Pause for review.

---

## Phase 3 — Evaluation harness

### Task 18: Embedding cache + cached wrapper

**Files:**
- Create: `src/code_rag_eval/ingest/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cache.py`:
```python
from pathlib import Path
from code_rag_eval.ingest.cache import EmbeddingCache, CachedEmbeddingClient
from tests.fakes import FakeEmbeddingClient


class _CountingClient(FakeEmbeddingClient):
    def __init__(self):
        super().__init__(model="counting")
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return super().embed(texts)


def test_cache_avoids_recompute(tmp_path: Path):
    inner = _CountingClient()
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    client = CachedEmbeddingClient(inner, cache)

    v1 = client.embed(["alpha", "beta"])
    assert inner.calls == 1                # one batch for two misses
    v2 = client.embed(["alpha", "beta"])   # both hits
    assert inner.calls == 1                # no new inner call
    assert v1 == v2

    mixed = client.embed(["alpha", "gamma"])  # one hit, one miss
    assert inner.calls == 2                    # only the miss triggered a call
    assert mixed[0] == v1[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `cache.py`**

```python
from __future__ import annotations
import hashlib
import json
import sqlite3
from pathlib import Path


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: str | Path):
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v TEXT)")
        self._conn.commit()

    def get(self, model: str, text: str) -> list[float] | None:
        row = self._conn.execute("SELECT v FROM emb WHERE k=?", (_key(model, text),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, text: str, vector: list[float]) -> None:
        self._conn.execute("INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)",
                           (_key(model, text), json.dumps(vector)))
        self._conn.commit()


class CachedEmbeddingClient:
    """Wraps any EmbeddingClient; only misses hit the inner client."""

    def __init__(self, inner, cache: EmbeddingCache):
        self._inner = inner
        self._cache = cache
        self.model = inner.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [self._cache.get(self.model, t) for t in texts]
        misses = [t for t, r in zip(texts, results) if r is None]
        if misses:
            computed = self._inner.embed(misses)
            it = iter(computed)
            for i, r in enumerate(results):
                if r is None:
                    vec = next(it)
                    self._cache.put(self.model, texts[i], vec)
                    results[i] = vec
        return [r for r in results if r is not None]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/ingest/cache.py tests/test_cache.py
git commit -m "feat: sqlite embedding cache + cached client wrapper"
```

---

### Task 19: Coverage mapping + retrieval metrics

**Files:**
- Create: `src/code_rag_eval/eval/coverage.py`, `src/code_rag_eval/eval/retrieval_metrics.py`
- Test: `tests/test_retrieval_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_retrieval_metrics.py`:
```python
from code_rag_eval.types import Chunk, RetrievedChunk
from code_rag_eval.eval.dataset import EvalRecord
from code_rag_eval.eval.coverage import chunk_covers
from code_rag_eval.eval.retrieval_metrics import hit_at_k, recall_at_k, reciprocal_rank


def _r(file, s, e, rank):
    return RetrievedChunk(chunk=Chunk(text="x", file=file, start_line=s, end_line=e), score=1.0, rank=rank)


def test_chunk_covers_requires_same_file_and_overlap():
    c = Chunk(text="x", file="a.py", start_line=10, end_line=20)
    assert chunk_covers(c, "a.py", (15, 16)) is True
    assert chunk_covers(c, "a.py", (20, 25)) is True   # touching boundary overlaps
    assert chunk_covers(c, "a.py", (21, 30)) is False
    assert chunk_covers(c, "b.py", (15, 16)) is False


def _record(symbols, files, ranges):
    return EvalRecord(id="q", category="locate", question="q",
                      gold_symbols=symbols, gold_files=files, gold_line_ranges=ranges,
                      reference_answer="a")


def test_hit_recall_mrr():
    rec = _record(["s1", "s2"], ["a.py", "a.py"], [(10, 20), (100, 110)])
    retrieved = [_r("a.py", 12, 14, 1), _r("a.py", 200, 210, 2)]  # covers gold #1 only, at rank 1
    assert hit_at_k(retrieved, rec, k=2) == 1
    assert recall_at_k(retrieved, rec, k=2) == 0.5
    assert reciprocal_rank(retrieved, rec) == 1.0

    none = [_r("a.py", 200, 210, 1)]
    assert hit_at_k(none, rec, k=1) == 0
    assert recall_at_k(none, rec, k=1) == 0.0
    assert reciprocal_rank(none, rec) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval_metrics.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `coverage.py` and `retrieval_metrics.py`**

`src/code_rag_eval/eval/coverage.py`:
```python
from __future__ import annotations
from code_rag_eval.types import Chunk


def chunk_covers(chunk: Chunk, gold_file: str, gold_range: tuple[int, int]) -> bool:
    """A chunk covers a gold symbol if same file and line ranges overlap (inclusive)."""
    if chunk.file != gold_file:
        return False
    gs, ge = gold_range
    return chunk.start_line <= ge and gs <= chunk.end_line
```

`src/code_rag_eval/eval/retrieval_metrics.py`:
```python
from __future__ import annotations
from code_rag_eval.types import RetrievedChunk
from code_rag_eval.eval.dataset import EvalRecord
from code_rag_eval.eval.coverage import chunk_covers


def _gold_pairs(record: EvalRecord) -> list[tuple[str, tuple[int, int]]]:
    return list(zip(record.gold_files, record.gold_line_ranges))


def _covered_gold_indices(retrieved: list[RetrievedChunk], record: EvalRecord, k: int) -> set[int]:
    golds = _gold_pairs(record)
    covered: set[int] = set()
    for r in retrieved[:k]:
        for gi, (gf, gr) in enumerate(golds):
            if chunk_covers(r.chunk, gf, gr):
                covered.add(gi)
    return covered


def hit_at_k(retrieved: list[RetrievedChunk], record: EvalRecord, k: int) -> int:
    return 1 if _covered_gold_indices(retrieved, record, k) else 0


def recall_at_k(retrieved: list[RetrievedChunk], record: EvalRecord, k: int) -> float:
    golds = _gold_pairs(record)
    if not golds:
        return 0.0
    return len(_covered_gold_indices(retrieved, record, k)) / len(golds)


def reciprocal_rank(retrieved: list[RetrievedChunk], record: EvalRecord) -> float:
    golds = _gold_pairs(record)
    for r in retrieved:
        for gf, gr in golds:
            if chunk_covers(r.chunk, gf, gr):
                return 1.0 / r.rank
    return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieval_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/eval/coverage.py src/code_rag_eval/eval/retrieval_metrics.py tests/test_retrieval_metrics.py
git commit -m "feat: chunk coverage + retrieval metrics (hit/recall/MRR)"
```

---

### Task 20: LLM judge + generation metrics

**Files:**
- Create: `src/code_rag_eval/eval/judge.py`, `src/code_rag_eval/eval/generation_metrics.py`
- Test: `tests/test_generation_metrics.py`

> **Spec note:** The design named RAGAS for faithfulness/answer-relevancy/context-precision. This plan implements those three as a small, self-contained **LLM-judge** module (own prompts, behind a `LLMJudge` protocol) plus a deterministic **citation accuracy** metric. Rationale: full offline TDD coverage and no coupling to RAGAS's shifting API inside step-by-step tasks. RAGAS can be added later as an industry-standard cross-check behind the same `generation_metrics` interface — tracked for the Phase 4–7 plan.

- [ ] **Step 1: Write the failing test**

`tests/test_generation_metrics.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generation_metrics.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `judge.py` and `generation_metrics.py`**

`src/code_rag_eval/eval/judge.py`:
```python
from __future__ import annotations
import re
from typing import Protocol


class LLMJudge(Protocol):
    def is_relevant(self, question: str, chunk_text: str) -> bool: ...
    def faithfulness(self, answer: str, context: str) -> float: ...
    def answer_relevancy(self, question: str, answer: str) -> float: ...


def _first_float(text: str, default: float = 0.0) -> float:
    m = re.search(r"[0-1](?:\.\d+)?", text)
    return min(1.0, max(0.0, float(m.group(0)))) if m else default


class AnthropicJudge:
    """LLM-as-judge backed by Anthropic. Used for real eval runs (needs API key)."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        from anthropic import Anthropic
        self._client = Anthropic()
        self.model = model

    def _ask(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    def is_relevant(self, question: str, chunk_text: str) -> bool:
        out = self._ask(
            f"Is this code relevant to answering the question? Reply yes or no only.\n"
            f"Question: {question}\nCode:\n{chunk_text}"
        )
        return out.strip().lower().startswith("y")

    def faithfulness(self, answer: str, context: str) -> float:
        return _first_float(self._ask(
            "Rate 0.0-1.0 how fully the answer is supported by the context (no unsupported "
            f"claims). Reply with only the number.\nContext:\n{context}\nAnswer:\n{answer}"
        ))

    def answer_relevancy(self, question: str, answer: str) -> float:
        return _first_float(self._ask(
            "Rate 0.0-1.0 how directly the answer addresses the question. Reply with only "
            f"the number.\nQuestion: {question}\nAnswer:\n{answer}"
        ))
```

`src/code_rag_eval/eval/generation_metrics.py`:
```python
from __future__ import annotations
from code_rag_eval.types import RetrievedChunk
from code_rag_eval.eval.dataset import EvalRecord
from code_rag_eval.eval.coverage import chunk_covers
from code_rag_eval.generate.answer import Answer
from code_rag_eval.eval.judge import LLMJudge


def context_precision(question: str, retrieved: list[RetrievedChunk], judge: LLMJudge) -> float:
    if not retrieved:
        return 0.0
    relevant = sum(1 for r in retrieved if judge.is_relevant(question, r.chunk.text))
    return relevant / len(retrieved)


def faithfulness(answer: Answer, retrieved: list[RetrievedChunk], judge: LLMJudge) -> float:
    context = "\n\n".join(r.chunk.text for r in retrieved)
    return judge.faithfulness(answer.text, context)


def answer_relevancy(question: str, answer: Answer, judge: LLMJudge) -> float:
    return judge.answer_relevancy(question, answer.text)


def citation_accuracy(answer: Answer, record: EvalRecord) -> float:
    if not answer.citations:
        return 0.0
    golds = list(zip(record.gold_files, record.gold_line_ranges))
    correct = 0
    for cite in answer.citations:
        if any(chunk_covers_point(cite.file, cite.line, gf, gr) for gf, gr in golds):
            correct += 1
    return correct / len(answer.citations)


def chunk_covers_point(cite_file: str, cite_line: int, gold_file: str, gold_range: tuple[int, int]) -> bool:
    gs, ge = gold_range
    return cite_file == gold_file and gs <= cite_line <= ge
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generation_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/code_rag_eval/eval/judge.py src/code_rag_eval/eval/generation_metrics.py tests/test_generation_metrics.py
git commit -m "feat: LLM-judge + generation metrics (faithfulness/relevancy/precision/citation)"
```

---

### Task 21: Harness + report + wire CLI `eval`

**Files:**
- Create: `src/code_rag_eval/eval/harness.py`
- Modify: `src/code_rag_eval/cli.py` (replace `eval` stub)
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write the failing test**

`tests/test_harness.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_harness.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `harness.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean
from typing import Callable
from code_rag_eval.eval.dataset import EvalRecord
from code_rag_eval.generate.answer import Answer
from code_rag_eval.eval.judge import LLMJudge
from code_rag_eval.eval import retrieval_metrics as rm
from code_rag_eval.eval import generation_metrics as gm

AnswerFn = Callable[[str, list], Answer]


def evaluate(records: list[EvalRecord], retriever, answer_fn: AnswerFn,
             judge: LLMJudge, ks: tuple[int, ...] = (1, 3, 5, 10), top_k: int = 10) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        retrieved = retriever.retrieve(rec.question, top_k)
        answer = answer_fn(rec.question, retrieved)
        row: dict = {"id": rec.id, "category": rec.category}
        for k in ks:
            row[f"hit_at_{k}"] = rm.hit_at_k(retrieved, rec, k)
            row[f"recall_at_{k}"] = rm.recall_at_k(retrieved, rec, k)
        row["mrr"] = rm.reciprocal_rank(retrieved, rec)
        row["context_precision"] = gm.context_precision(rec.question, retrieved, judge)
        row["faithfulness"] = gm.faithfulness(answer, retrieved, judge)
        row["answer_relevancy"] = gm.answer_relevancy(rec.question, answer, judge)
        row["citation_accuracy"] = gm.citation_accuracy(answer, rec)
        rows.append(row)
    return rows


def aggregate(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    metric_keys = ["mrr", "context_precision", "faithfulness", "answer_relevancy", "citation_accuracy"]
    for k in ks:
        metric_keys += [f"hit_at_{k}", f"recall_at_{k}"]
    return {key: mean(r[key] for r in rows) for key in metric_keys if rows}


def write_report(config_name: str, rows: list[dict], agg: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": config_name, "aggregate": agg, "per_record": rows}
    out = out_dir / f"{config_name}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_harness.py -v`
Expected: PASS

- [ ] **Step 5: Wire the `eval` CLI command**

Replace the `eval` stub in `src/code_rag_eval/cli.py` with (add imports at top as needed):
```python
from functools import partial
from code_rag_eval.ingest.cache import EmbeddingCache, CachedEmbeddingClient
from code_rag_eval.eval.dataset import load_eval_set
from code_rag_eval.eval.judge import AnthropicJudge
from code_rag_eval.eval.harness import evaluate, aggregate, write_report
from code_rag_eval.generate.answer import generate_answer


@app.command()
def eval(config: str = "configs/baseline.yaml",
         questions: str = "data/eval/questions.jsonl",
         results_dir: str = "results") -> None:
    """Run the evaluation harness for a config over the verified question set."""
    load_dotenv()
    cfg = load_config(config)
    embed = CachedEmbeddingClient(make_embedding_client(cfg.embedding),
                                  EmbeddingCache(".emb_cache.sqlite"))
    llm = make_llm_client(cfg.generation)
    store = ChromaStore(collection_name=cfg.name, persist_dir=CHROMA_DIR)
    retriever = VectorRetriever(store, embed)
    judge = AnthropicJudge(model=cfg.generation.model)
    answer_fn = partial(generate_answer, llm=llm)

    records = load_eval_set(questions)
    rows = evaluate(records, retriever, lambda q, r: answer_fn(q, r), judge)
    agg = aggregate(rows)
    out = write_report(cfg.name, rows, agg, Path(results_dir))
    typer.echo(f"evaluated {len(records)} questions -> {out}")
    for key in ("hit_at_1", "hit_at_5", "mrr", "faithfulness", "citation_accuracy"):
        typer.echo(f"  {key}: {agg[key]:.3f}")
```

- [ ] **Step 6: Run the full offline suite, then a real eval pass (manual)**

Run: `uv run pytest -q`
Expected: PASS (entire suite)

Then, with corpus ingested + `.env` present:
```bash
uv run code-rag-eval ingest
uv run code-rag-eval eval
```
Expected: writes `results/baseline.json` and prints aggregate `hit_at_1`, `hit_at_5`, `mrr`, `faithfulness`, `citation_accuracy`. (Costs API tokens: embeds queries + runs the LLM judge per question.)

- [ ] **Step 7: Commit**

```bash
git add src/code_rag_eval/eval/harness.py src/code_rag_eval/cli.py tests/test_harness.py
git commit -m "feat: eval harness + report writer + wire CLI eval"
```

**Phase 3 checkpoint:** `uv run code-rag-eval eval` produces a metrics report for the baseline config. The system can now be compared against future configs (Phase 4–5).

---

## Self-Review Notes

- **Spec coverage:** baseline pipeline (Tasks 6–14) ✓; eval set of ~50 verified triples (Tasks 15–17) ✓; retrieval metrics hit-rate@k/recall@k/MRR (Task 19) ✓; generation metrics faithfulness/answer-relevancy/context-precision + citation accuracy (Task 20) ✓; embedding cache (Task 18) ✓; FastAPI corpus pinned (Task 5) ✓; `file:line` citations (Task 13) ✓; config-driven so Phase 4 adds impls (factories in Task 14) ✓.
- **Deferred to the Phase 4–7 plan (out of scope here, by design):** AST/tree-sitter chunker, Voyage `voyage-code-3`, BM25 + RRF hybrid, the 8-config sweep, RAGAS cross-check, DECISIONS.md, Streamlit UI, README diagram. The `experiment` CLI command and the `"ast"`/`"voyage"`/`"hybrid"` config values are intentional stubs that raise until then.
- **Type consistency:** `Chunk`, `RetrievedChunk`, `Answer`, `Citation`, `EvalRecord`, `Symbol` are defined once and reused with identical fields across tasks; metric helpers all key off `chunk_covers` / `chunk_covers_point`.
- **`context_precision` (RAGAS in the spec)** is realized via the custom `LLMJudge` here — flagged in Task 20's spec note and listed for a later RAGAS cross-check.
