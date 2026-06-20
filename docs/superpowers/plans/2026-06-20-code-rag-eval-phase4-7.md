# code-rag-eval — Phases 4–7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the experiment variables (AST chunking, code-specialized embeddings, hybrid retrieval), the config-matrix sweep, the data-backed decision report, and a thin demo UI — turning the Phase 0–3 baseline into the full `code-rag-eval` project.

**Architecture:** All new pieces slot behind the existing seams from Phases 0–3. A tree-sitter `chunk_ast` joins `chunk_fixed` behind `build_chunks`; a `VoyageEmbeddingClient` joins `OpenAIEmbeddingClient` behind the factory; `BM25Retriever` + `HybridRetriever` join `VectorRetriever` behind a new `make_retriever` factory; an `experiment` command sweeps the 8-config matrix; `report.py` turns results into `DECISIONS.md`; a Streamlit app reuses the retriever + generate path.

**Tech Stack (added in the deps commit `3a209fe`):** `tree-sitter` 0.25 + `tree-sitter-python` (AST chunking), `voyageai` (voyage-code-3), `rank-bm25` (lexical), `streamlit` (added in Task 11). Everything else as Phases 0–3 (uv, Python 3.12, Typer, Pydantic, Chroma).

**Spec:** `docs/superpowers/specs/2026-06-19-code-rag-eval-design.md`. Prior plan: `docs/superpowers/plans/2026-06-19-code-rag-eval-phase0-3.md`.

## ⚠️ No-API-keys constraint (read first)

There are **no API keys available**, so nothing that calls OpenAI / Anthropic / Voyage may be run. Every task below is **fully offline-testable with the existing fakes** (`tests/fakes.py`) — real clients (`VoyageEmbeddingClient`, etc.) are thin wrappers with lazy imports, exercised only through fakes/stubs. Three deliverables intrinsically need a live run and are therefore built as **runnable code + scaffolds, not executed**:

- **Phase 5 sweep** (`experiment` command): code complete + a `--dry-run` path that's offline-testable; the real 8-config run (embeddings + judge) is a documented manual step.
- **Phase 6 `DECISIONS.md`**: the generator is offline-tested with synthetic results; a scaffold `DECISIONS.md` is committed and regenerated after a real sweep.
- **Phase 7 Streamlit app**: built + import-smoke-tested; live answering needs keys.

The implementer must **NOT** run any live API command. Run only `uv run pytest` and offline `--help`/`--dry-run` checks.

## File structure (new/changed)

```
src/code_rag_eval/
  ingest/
    chunkers.py        # + chunk_ast (tree-sitter)         [Task 2]
    embed.py           # + VoyageEmbeddingClient            [Task 3]
    pipeline.py        # build_chunks: add "ast" branch     [Task 2]
    store.py           # + ChromaStore.all_chunks()         [Task 4]
  retrieve/
    tokenize.py        # code-aware tokenizer (NEW)         [Task 1]
    bm25.py            # BM25Retriever (NEW)                [Task 5]
    hybrid.py          # HybridRetriever (RRF) (NEW)        [Task 6]
    factory.py         # make_retriever (NEW)              [Task 7]
  factories.py         # wire voyage provider               [Task 3]
  eval/
    experiment.py      # matrix_configs (NEW)               [Task 8]
    report.py          # comparison/winner/DECISIONS (NEW)  [Task 9]
  cli.py               # ask/eval use make_retriever [T7]; experiment command [T10]
app/
  streamlit_app.py     # demo UI (NEW)                      [Task 11]
DECISIONS.md           # scaffold, generated                [Task 9]
README.md              # status + new commands              [Task 12]
tests/                 # one test module per new unit
```

Conventions unchanged from Phase 0–3 (POSIX-relative file paths, 1-based inclusive lines, offline tests via `tests/fakes.py`, `uv run pytest` / `uv run code-rag-eval`).

---

## Phase 4 — Experiment variables

### Task 1: Code-aware tokenizer

**Files:** Create `src/code_rag_eval/retrieve/tokenize.py`. Test: `tests/test_tokenize.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_tokenize.py`:
```python
from code_rag_eval.retrieve.tokenize import tokenize_code


def test_tokenize_keeps_whole_identifier_and_subtokens():
    toks = tokenize_code("def get_route_handler(): camelCaseName = 1")
    assert "get_route_handler" in toks                 # whole snake identifier
    assert {"get", "route", "handler"} <= set(toks)    # snake subtokens
    assert "camelcasename" in toks                     # whole identifier, lowercased
    assert {"camel", "case", "name"} <= set(toks)      # camelCase subtokens
    assert "def" in toks


def test_tokenize_empty_is_empty():
    assert tokenize_code("") == []
```

- [ ] **Step 2: Run** `uv run pytest tests/test_tokenize.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/code_rag_eval/retrieve/tokenize.py`:
```python
from __future__ import annotations
import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[a-z]+|[A-Z]+|[0-9]+")


def tokenize_code(text: str) -> list[str]:
    """Tokenize code for BM25: keep each whole identifier (lowercased) plus its
    snake_case / camelCase subtokens, so both exact symbol names and their parts match."""
    out: list[str] = []
    for ident in _IDENT.findall(text):
        low = ident.lower()
        out.append(low)
        for part in ident.split("_"):
            for sub in _CAMEL.findall(part):
                s = sub.lower()
                if s and s != low:
                    out.append(s)
    return out
```

- [ ] **Step 4: Run** `uv run pytest tests/test_tokenize.py -v` → PASS.

- [ ] **Step 5: Commit**
```
git add src/code_rag_eval/retrieve/tokenize.py tests/test_tokenize.py
git commit -m "feat: code-aware tokenizer for BM25"
```
Add trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 2: AST-aware chunker (tree-sitter)

**Files:** Modify `src/code_rag_eval/ingest/chunkers.py`, `src/code_rag_eval/ingest/pipeline.py`. Test: `tests/test_chunkers.py` (add), `tests/test_pipeline.py` (add).

> The tree-sitter 0.25 API used here is verified working in this environment: `Parser(Language(tree_sitter_python.language()))`, `node.start_point[0]`/`end_point[0]` are 0-based rows, decorated defs appear as `decorated_definition` whose inner child is the `function_definition`/`class_definition`.

- [ ] **Step 1: Add the failing tests** to `tests/test_chunkers.py` (keep the existing `chunk_fixed` tests; append):
```python
from code_rag_eval.ingest.chunkers import chunk_ast

_AST_SRC = '''\
import os


def top():
    """doc"""
    return 1


@decorator
class Service:
    """svc"""

    def handle(self, x):
        return x

    def other(self):
        return 2
'''


def test_chunk_ast_splits_by_definition():
    chunks = chunk_ast(_AST_SRC, "m.py")
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["top"].kind == "function"
    assert by_symbol["top"].start_line == 4
    assert "return 1" in by_symbol["top"].text
    assert by_symbol["Service"].kind == "class"
    assert "class Service" in by_symbol["Service"].text
    assert "return x" not in by_symbol["Service"].text          # method bodies are separate
    assert by_symbol["Service.handle"].kind == "method"
    assert "return x" in by_symbol["Service.handle"].text       # never split a function
    assert "Service.other" in by_symbol


def test_chunk_ast_windows_long_units():
    long_src = "def big():\n" + "\n".join(f"    x{i} = {i}" for i in range(300))
    chunks = chunk_ast(long_src, "b.py", max_lines=100, overlap_lines=10)
    assert len(chunks) > 1
    assert all(c.symbol == "big" and c.kind == "function" for c in chunks)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_chunkers.py -v` → the two new tests FAIL (ImportError: chunk_ast).

- [ ] **Step 3: Implement** — add to `src/code_rag_eval/ingest/chunkers.py` (keep `chunk_fixed`; add imports at top and the functions below):
```python
from tree_sitter import Language, Parser
import tree_sitter_python

_PARSER = Parser(Language(tree_sitter_python.language()))


def _name_of(def_node) -> str:
    name = def_node.child_by_field_name("name")
    return name.text.decode() if name is not None else "?"


def _inner_def(node):
    """Unwrap a decorated_definition to its function_definition/class_definition."""
    if node.type == "decorated_definition":
        for c in node.children:
            if c.type in ("function_definition", "class_definition"):
                return c
    return node


def _split_unit(unit_lines, start_line, file, symbol, kind, signature, max_lines, overlap_lines):
    n = len(unit_lines)
    if n <= max_lines:
        return [Chunk(text="\n".join(unit_lines), file=file, start_line=start_line,
                      end_line=start_line + n - 1, kind=kind, symbol=symbol, signature=signature)]
    out = []
    step = max(1, max_lines - overlap_lines)
    i = 0
    while i < n:
        window = unit_lines[i:i + max_lines]
        out.append(Chunk(text="\n".join(window), file=file, start_line=start_line + i,
                         end_line=start_line + i + len(window) - 1, kind=kind,
                         symbol=symbol, signature=signature))
        if i + max_lines >= n:
            break
        i += step
    return out


def chunk_ast(text: str, file: str, max_lines: int = 120, overlap_lines: int = 20) -> list[Chunk]:
    """AST-aware chunking via tree-sitter: one chunk per top-level function, per class
    header, and per method. Never splits a definition mid-body unless it exceeds
    max_lines (then it is windowed). Module-level statements between definitions are not
    separately indexed — eval gold symbols are always definitions, so retrieval metrics
    are unaffected; this is a deliberate scope choice for the AST strategy.
    """
    tree = _PARSER.parse(text.encode("utf-8"))
    root = tree.root_node
    lines = text.splitlines()
    units: list[tuple[int, int, str, str]] = []  # (start_line, end_line, kind, symbol) 1-based

    for node in root.children:
        inner = _inner_def(node)
        if inner.type == "function_definition":
            units.append((node.start_point[0] + 1, node.end_point[0] + 1, "function", _name_of(inner)))
        elif inner.type == "class_definition":
            cname = _name_of(inner)
            body = inner.child_by_field_name("body")
            methods = []
            if body is not None:
                for ch in body.children:
                    if _inner_def(ch).type == "function_definition":
                        methods.append(ch)
            if methods:
                header_start = node.start_point[0] + 1
                header_end = max(header_start, methods[0].start_point[0])  # line before first method
                units.append((header_start, header_end, "class", cname))
                for ch in methods:
                    m_inner = _inner_def(ch)
                    units.append((ch.start_point[0] + 1, ch.end_point[0] + 1, "method",
                                  f"{cname}.{_name_of(m_inner)}"))
            else:
                units.append((node.start_point[0] + 1, node.end_point[0] + 1, "class", cname))

    units.sort(key=lambda u: u[0])
    chunks: list[Chunk] = []
    for (s, e, kind, symbol) in units:
        unit_lines = lines[s - 1:e]
        if not unit_lines:
            continue
        signature = unit_lines[0].strip()
        chunks.extend(_split_unit(unit_lines, s, file, symbol, kind, signature, max_lines, overlap_lines))
    return chunks
```

- [ ] **Step 4: Run** `uv run pytest tests/test_chunkers.py -v` → PASS.

- [ ] **Step 5: Wire the pipeline** — in `src/code_rag_eval/ingest/pipeline.py`, change the strategy branch in `build_chunks` to add the `ast` case (import `chunk_ast`):
```python
from code_rag_eval.ingest.chunkers import chunk_fixed, chunk_ast
```
```python
        if chunking.strategy == "fixed":
            chunks.extend(chunk_fixed(text, rel, chunking.window_lines, chunking.overlap_lines))
        elif chunking.strategy == "ast":
            chunks.extend(chunk_ast(text, rel))
        else:
            raise ValueError(f"unknown chunking strategy: {chunking.strategy}")
```

- [ ] **Step 6: Add a pipeline test** to `tests/test_pipeline.py` (append):
```python
def test_build_chunks_ast_strategy(tmp_path):
    from code_rag_eval.config import ChunkingConfig
    from code_rag_eval.ingest.pipeline import build_chunks
    root = tmp_path / "fastapi"
    src = root / "fastapi"
    src.mkdir(parents=True)
    (src / "m.py").write_text("def alpha():\n    return 1\n\n\nclass C:\n    def beta(self):\n        return 2\n", encoding="utf-8")
    chunks = build_chunks(src, root, ChunkingConfig(strategy="ast"))
    syms = {c.symbol for c in chunks}
    assert "alpha" in syms and "C.beta" in syms
    assert all(c.file == "fastapi/m.py" for c in chunks)
```

- [ ] **Step 7: Run** `uv run pytest tests/test_chunkers.py tests/test_pipeline.py -v` → PASS.

- [ ] **Step 8: Commit**
```
git add src/code_rag_eval/ingest/chunkers.py src/code_rag_eval/ingest/pipeline.py tests/test_chunkers.py tests/test_pipeline.py
git commit -m "feat: tree-sitter AST-aware chunker + pipeline wiring"
```
Add the Co-Authored-By trailer.

---

### Task 3: Voyage embedding client + factory wiring

**Files:** Modify `src/code_rag_eval/ingest/embed.py`, `src/code_rag_eval/factories.py`. Test: `tests/test_factories.py` (rewrite).

- [ ] **Step 1: Rewrite** `tests/test_factories.py` to the failing target:
```python
import pytest
from code_rag_eval.config import EmbeddingConfig, GenerationConfig
from code_rag_eval.factories import make_embedding_client, make_llm_client


def _stub_factory():
    created = {}

    class _Stub:
        def __init__(self, model):
            created["model"] = model
            self.model = model

    return created, _Stub


def test_openai_embedding_selected(monkeypatch):
    created, stub = _stub_factory()
    monkeypatch.setattr("code_rag_eval.factories.OpenAIEmbeddingClient", stub)
    client = make_embedding_client(EmbeddingConfig(provider="openai", model="text-embedding-3-large"))
    assert created["model"] == "text-embedding-3-large" and client.model == "text-embedding-3-large"


def test_voyage_embedding_selected(monkeypatch):
    created, stub = _stub_factory()
    monkeypatch.setattr("code_rag_eval.factories.VoyageEmbeddingClient", stub)
    client = make_embedding_client(EmbeddingConfig(provider="voyage", model="voyage-code-3"))
    assert created["model"] == "voyage-code-3" and client.model == "voyage-code-3"


def test_unknown_llm_provider_raises():
    with pytest.raises(ValueError):
        make_llm_client(GenerationConfig(provider="openai", model="gpt-x"))
```

- [ ] **Step 2: Run** `uv run pytest tests/test_factories.py -v` → FAIL (ImportError: VoyageEmbeddingClient / AttributeError).

- [ ] **Step 3: Add `VoyageEmbeddingClient`** to `src/code_rag_eval/ingest/embed.py` (after `OpenAIEmbeddingClient`):
```python
class VoyageEmbeddingClient:
    """Thin wrapper over the Voyage AI embeddings API. Exercised manually / via CLI."""

    def __init__(self, model: str = "voyage-code-3"):
        import voyageai
        self._client = voyageai.Client()
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(texts, model=self.model).embeddings
```

- [ ] **Step 4: Wire the factory** — update `src/code_rag_eval/factories.py`:
```python
from __future__ import annotations
from code_rag_eval.config import EmbeddingConfig, GenerationConfig
from code_rag_eval.ingest.embed import OpenAIEmbeddingClient, VoyageEmbeddingClient
from code_rag_eval.generate.answer import AnthropicClient


def make_embedding_client(cfg: EmbeddingConfig):
    if cfg.provider == "openai":
        return OpenAIEmbeddingClient(model=cfg.model)
    if cfg.provider == "voyage":
        return VoyageEmbeddingClient(model=cfg.model)
    raise ValueError(f"unknown embedding provider: {cfg.provider}")


def make_llm_client(cfg: GenerationConfig):
    if cfg.provider == "anthropic":
        return AnthropicClient(model=cfg.model)
    raise ValueError(f"llm provider not supported: {cfg.provider}")
```

- [ ] **Step 5: Run** `uv run pytest tests/test_factories.py -v` → PASS. Confirm importing the module needs no Voyage key (the `import voyageai` is inside `__init__`).

- [ ] **Step 6: Commit**
```
git add src/code_rag_eval/ingest/embed.py src/code_rag_eval/factories.py tests/test_factories.py
git commit -m "feat: Voyage code embedding client + factory wiring"
```
Add the Co-Authored-By trailer.

---

### Task 4: `ChromaStore.all_chunks()`

**Files:** Modify `src/code_rag_eval/ingest/store.py`. Test: `tests/test_store.py` (add).

- [ ] **Step 1: Add the failing test** to `tests/test_store.py` (append):
```python
def test_all_chunks_round_trips_everything():
    store = ChromaStore(collection_name="all-chunks-test")
    chunks = [
        Chunk(text="def a(): ...", file="a.py", start_line=1, end_line=2, symbol="a"),
        Chunk(text="def b(): ...", file="b.py", start_line=3, end_line=4, symbol="b"),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    got = store.all_chunks()
    assert {c.symbol for c in got} == {"a", "b"}
    assert {c.file for c in got} == {"a.py", "b.py"}
```

- [ ] **Step 2: Run** `uv run pytest tests/test_store.py -v` → the new test FAILs (AttributeError: all_chunks).

- [ ] **Step 3: Implement** — add to `ChromaStore` in `src/code_rag_eval/ingest/store.py`:
```python
    def all_chunks(self) -> list[Chunk]:
        res = self._col.get(include=["documents", "metadatas"])
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        return [metadata_to_chunk(doc, meta) for doc, meta in zip(docs, metas)]
```

- [ ] **Step 4: Run** `uv run pytest tests/test_store.py -v` → PASS.

- [ ] **Step 5: Commit**
```
git add src/code_rag_eval/ingest/store.py tests/test_store.py
git commit -m "feat: ChromaStore.all_chunks for lexical index building"
```
Add the Co-Authored-By trailer.

---

### Task 5: BM25 retriever

**Files:** Create `src/code_rag_eval/retrieve/bm25.py`. Test: `tests/test_bm25.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_bm25.py`:
```python
from code_rag_eval.types import Chunk
from code_rag_eval.retrieve.bm25 import BM25Retriever


def _chunks():
    return [
        Chunk(text="def login(user): pass", file="auth.py", start_line=1, end_line=1),
        Chunk(text="def logout(user): pass", file="auth.py", start_line=2, end_line=2),
        Chunk(text="def render_template(name): pass", file="view.py", start_line=1, end_line=1),
    ]


def test_bm25_ranks_exact_symbol_first():
    retr = BM25Retriever(_chunks())
    results = retr.retrieve("login", k=2)
    assert len(results) == 2
    assert results[0].rank == 1 and results[1].rank == 2
    assert results[0].chunk.text.startswith("def login")


def test_bm25_empty_corpus_returns_empty():
    assert BM25Retriever([]).retrieve("login", k=5) == []
```

- [ ] **Step 2: Run** `uv run pytest tests/test_bm25.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/code_rag_eval/retrieve/bm25.py`:
```python
from __future__ import annotations
from rank_bm25 import BM25Okapi
from code_rag_eval.types import Chunk, RetrievedChunk
from code_rag_eval.retrieve.tokenize import tokenize_code


class BM25Retriever:
    """Lexical retriever over code-aware tokens. Builds its index from a chunk list."""

    def __init__(self, chunks: list[Chunk]):
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize_code(c.text) for c in self._chunks]) if self._chunks else None

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize_code(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]), rank=rank + 1)
                for rank, i in enumerate(order)]
```

- [ ] **Step 4: Run** `uv run pytest tests/test_bm25.py -v` → PASS.

- [ ] **Step 5: Commit**
```
git add src/code_rag_eval/retrieve/bm25.py tests/test_bm25.py
git commit -m "feat: BM25 lexical retriever"
```
Add the Co-Authored-By trailer.

---

### Task 6: Hybrid retriever (Reciprocal Rank Fusion)

**Files:** Create `src/code_rag_eval/retrieve/hybrid.py`. Test: `tests/test_hybrid.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_hybrid.py`:
```python
from code_rag_eval.types import Chunk, RetrievedChunk
from code_rag_eval.retrieve.hybrid import HybridRetriever


class _FixedRetriever:
    def __init__(self, ordered_chunks):
        self._ordered = ordered_chunks

    def retrieve(self, query, k):
        return [RetrievedChunk(chunk=c, score=1.0 / (i + 1), rank=i + 1)
                for i, c in enumerate(self._ordered[:k])]


def _c(name):
    return Chunk(text=name, file=f"{name}.py", start_line=1, end_line=1, symbol=name)


def test_hybrid_rrf_rewards_agreement():
    a, b, d = _c("a"), _c("b"), _c("d")
    # vector ranks a,b,d ; bm25 ranks b,a,d -> b and a both rank high in both arms
    vec = _FixedRetriever([a, b, d])
    bm25 = _FixedRetriever([b, a, d])
    results = HybridRetriever(vec, bm25, rrf_k=60).retrieve("q", k=3)
    assert {r.chunk.symbol for r in results} == {"a", "b", "d"}
    assert results[0].chunk.symbol in {"a", "b"}      # an agreed-upon top item wins
    assert results[-1].chunk.symbol == "d"            # ranked low in both -> last
    assert [r.rank for r in results] == [1, 2, 3]


def test_hybrid_dedupes_across_arms():
    a = _c("a")
    results = HybridRetriever(_FixedRetriever([a]), _FixedRetriever([a]), rrf_k=60).retrieve("q", k=5)
    assert len(results) == 1 and results[0].chunk.symbol == "a"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_hybrid.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/code_rag_eval/retrieve/hybrid.py`:
```python
from __future__ import annotations
from code_rag_eval.types import Chunk, RetrievedChunk


class HybridRetriever:
    """Fuses a vector retriever and a lexical retriever with Reciprocal Rank Fusion."""

    def __init__(self, vector_retriever, bm25_retriever, rrf_k: int = 60):
        self._vec = vector_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        pool = max(k, 20)  # fuse a wider pool from each arm than we finally return
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for results in (self._vec.retrieve(query, pool), self._bm25.retrieve(query, pool)):
            for r in results:
                cid = r.chunk.chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + r.rank)
                chunks[cid] = r.chunk
        order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [RetrievedChunk(chunk=chunks[cid], score=score, rank=i + 1)
                for i, (cid, score) in enumerate(order)]
```

- [ ] **Step 4: Run** `uv run pytest tests/test_hybrid.py -v` → PASS.

- [ ] **Step 5: Commit**
```
git add src/code_rag_eval/retrieve/hybrid.py tests/test_hybrid.py
git commit -m "feat: hybrid retriever (RRF fusion of vector + BM25)"
```
Add the Co-Authored-By trailer.

---

### Task 7: Retriever factory + wire CLI `ask`/`eval`

**Files:** Create `src/code_rag_eval/retrieve/factory.py`. Modify `src/code_rag_eval/cli.py`. Test: `tests/test_retriever_factory.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_retriever_factory.py`:
```python
from code_rag_eval.config import ExperimentConfig, RetrievalConfig
from code_rag_eval.types import Chunk
from code_rag_eval.ingest.store import ChromaStore
from code_rag_eval.retrieve.factory import make_retriever
from code_rag_eval.retrieve.vector import VectorRetriever
from code_rag_eval.retrieve.bm25 import BM25Retriever
from code_rag_eval.retrieve.hybrid import HybridRetriever
from tests.fakes import FakeEmbeddingClient


def _store():
    s = ChromaStore(collection_name="factory-test")
    embed = FakeEmbeddingClient()
    chunks = [Chunk(text="def login(): pass", file="a.py", start_line=1, end_line=1)]
    s.add(chunks, embed.embed([c.text for c in chunks]))
    return s


def _cfg(method):
    return ExperimentConfig(name="t", retrieval=RetrievalConfig(method=method))


def test_make_retriever_selects_impl():
    store, embed = _store(), FakeEmbeddingClient()
    assert isinstance(make_retriever(_cfg("vector"), store, embed), VectorRetriever)
    assert isinstance(make_retriever(_cfg("bm25"), store, embed), BM25Retriever)
    assert isinstance(make_retriever(_cfg("hybrid"), store, embed), HybridRetriever)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_retriever_factory.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/code_rag_eval/retrieve/factory.py`:
```python
from __future__ import annotations
from code_rag_eval.config import ExperimentConfig
from code_rag_eval.retrieve.vector import VectorRetriever
from code_rag_eval.retrieve.bm25 import BM25Retriever
from code_rag_eval.retrieve.hybrid import HybridRetriever


def make_retriever(cfg: ExperimentConfig, store, embed_client):
    method = cfg.retrieval.method
    vector = VectorRetriever(store, embed_client)
    if method == "vector":
        return vector
    bm25 = BM25Retriever(store.all_chunks())
    if method == "bm25":
        return bm25
    if method == "hybrid":
        return HybridRetriever(vector, bm25, rrf_k=cfg.retrieval.rrf_k)
    raise ValueError(f"unknown retrieval method: {method}")
```

- [ ] **Step 4: Run** `uv run pytest tests/test_retriever_factory.py -v` → PASS.

- [ ] **Step 5: Wire the CLI** — in `src/code_rag_eval/cli.py`, import the factory and replace the direct `VectorRetriever(...)` construction in BOTH `ask` and `eval` with `make_retriever(cfg, store, embed)`.
  - Add import: `from code_rag_eval.retrieve.factory import make_retriever`
  - In `ask`: replace `retriever = VectorRetriever(store, embed)` with `retriever = make_retriever(cfg, store, embed)`.
  - In `eval`: replace `retriever = VectorRetriever(store, embed)` with `retriever = make_retriever(cfg, store, embed)`.
  - The now-unused `from code_rag_eval.retrieve.vector import VectorRetriever` import in cli.py may be removed.

- [ ] **Step 6: Run** `uv run pytest -q` (whole suite) → PASS. Run `uv run code-rag-eval ask --help` to confirm the CLI still loads.

- [ ] **Step 7: Commit**
```
git add src/code_rag_eval/retrieve/factory.py src/code_rag_eval/cli.py tests/test_retriever_factory.py
git commit -m "feat: retriever factory + wire CLI ask/eval to it"
```
Add the Co-Authored-By trailer.

---

## Phase 5 — Experiment sweep

### Task 8: Config matrix generation

**Files:** Create `src/code_rag_eval/eval/experiment.py`. Test: `tests/test_experiment.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_experiment.py`:
```python
from code_rag_eval.config import ExperimentConfig
from code_rag_eval.eval.experiment import matrix_configs


def test_matrix_is_eight_named_configs():
    cfgs = matrix_configs(ExperimentConfig(name="base"))
    names = [c.name for c in cfgs]
    assert len(cfgs) == 8
    assert "fixed_openai_vector" in names
    assert "ast_voyage_hybrid" in names
    # each config carries the right axis values
    byname = {c.name: c for c in cfgs}
    c = byname["ast_voyage_hybrid"]
    assert c.chunking.strategy == "ast"
    assert c.embedding.provider == "voyage" and c.embedding.model == "voyage-code-3"
    assert c.retrieval.method == "hybrid"
    d = byname["fixed_openai_vector"]
    assert d.chunking.strategy == "fixed"
    assert d.embedding.provider == "openai" and d.embedding.model == "text-embedding-3-large"
    assert d.retrieval.method == "vector"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_experiment.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/code_rag_eval/eval/experiment.py`:
```python
from __future__ import annotations
from code_rag_eval.config import ExperimentConfig, EmbeddingConfig

_EMBEDDINGS = {
    "openai": "text-embedding-3-large",
    "voyage": "voyage-code-3",
}


def matrix_configs(base: ExperimentConfig) -> list[ExperimentConfig]:
    """The 8-config sweep: chunking{fixed,ast} x embedding{openai,voyage} x retrieval{vector,hybrid}."""
    configs: list[ExperimentConfig] = []
    for strategy in ("fixed", "ast"):
        for provider in ("openai", "voyage"):
            for method in ("vector", "hybrid"):
                configs.append(base.model_copy(update={
                    "name": f"{strategy}_{provider}_{method}",
                    "chunking": base.chunking.model_copy(update={"strategy": strategy}),
                    "embedding": EmbeddingConfig(provider=provider, model=_EMBEDDINGS[provider]),
                    "retrieval": base.retrieval.model_copy(update={"method": method}),
                }))
    return configs
```

- [ ] **Step 4: Run** `uv run pytest tests/test_experiment.py -v` → PASS.

- [ ] **Step 5: Commit**
```
git add src/code_rag_eval/eval/experiment.py tests/test_experiment.py
git commit -m "feat: 8-config experiment matrix generation"
```
Add the Co-Authored-By trailer.

---

## Phase 6 — Decision report

### Task 9: Report generator + `DECISIONS.md` scaffold

**Files:** Create `src/code_rag_eval/eval/report.py`, `DECISIONS.md` (generated). Test: `tests/test_report.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
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
    assert "ast_voyage_hybrid" in text          # winner named
    assert "| config |" in text.lower()         # comparison table present
```

- [ ] **Step 2: Run** `uv run pytest tests/test_report.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/code_rag_eval/eval/report.py`:
```python
from __future__ import annotations
import json
from pathlib import Path

_COLUMNS = ["hit_at_1", "hit_at_5", "recall_at_5", "mrr", "faithfulness", "answer_relevancy",
            "context_precision", "citation_accuracy"]


def load_results(results_dir: str | Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted(Path(results_dir).glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if "aggregate" in data:
            out.append(data)
    return out


def pick_winner(results: list[dict], primary: str = "hit_at_5", tiebreak: str = "faithfulness") -> str | None:
    if not results:
        return None
    best = max(results, key=lambda r: (r["aggregate"].get(primary, 0.0),
                                       r["aggregate"].get(tiebreak, 0.0)))
    return best["config"]


def comparison_table(results: list[dict]) -> str:
    cols = [c for c in _COLUMNS if any(c in r["aggregate"] for r in results)]
    header = "| config | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for r in sorted(results, key=lambda r: r["config"]):
        cells = [f"{r['aggregate'].get(c, float('nan')):.3f}" for c in cols]
        rows.append(f"| {r['config']} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_decisions(results: list[dict], out_path: str | Path) -> None:
    out = Path(out_path)
    if not results:
        out.write_text(
            "# DECISIONS\n\n"
            "_No experiment results yet._ Run the sweep with API keys and a verified eval set:\n\n"
            "```\nuv run code-rag-eval experiment\n```\n\n"
            "This regenerates the comparison table and names the winning config below.\n",
            encoding="utf-8")
        return
    winner = pick_winner(results)
    out.write_text(
        "# DECISIONS\n\n"
        f"**Winning config: `{winner}`** (ranked by hit@5, tie-broken by faithfulness).\n\n"
        "## Comparison\n\n"
        f"{comparison_table(results)}\n\n"
        "## Methodology\n\n"
        "Each config was ingested over the pinned FastAPI corpus and scored on the hand-verified "
        "eval set: retrieval (hit-rate@k, recall@k, MRR) and generation (faithfulness, answer "
        "relevancy, context precision, citation accuracy). Coverage is line-range overlap against "
        "ground-truth symbols. Framing follows the COIR code-retrieval benchmark.\n",
        encoding="utf-8")
```

- [ ] **Step 4: Run** `uv run pytest tests/test_report.py -v` → PASS.

- [ ] **Step 5: Generate the committed scaffold** — run:
```
uv run python -c "from code_rag_eval.eval.report import write_decisions; write_decisions([], 'DECISIONS.md')"
```
Confirm `DECISIONS.md` now exists with the "no experiment results yet" scaffold.

- [ ] **Step 6: Commit**
```
git add src/code_rag_eval/eval/report.py tests/test_report.py DECISIONS.md
git commit -m "feat: decision report generator + DECISIONS.md scaffold"
```
Add the Co-Authored-By trailer.

---

### Task 10: `experiment` CLI command (sweep runner)

**Files:** Modify `src/code_rag_eval/cli.py`. Test: `tests/test_cli.py` (add).

- [ ] **Step 1: Add the failing test** to `tests/test_cli.py` (append):
```python
def test_experiment_dry_run_lists_matrix():
    result = runner.invoke(app, ["experiment", "--dry-run"])
    assert result.exit_code == 0
    assert "fixed_openai_vector" in result.output
    assert "ast_voyage_hybrid" in result.output
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -v` → the new test FAILs (the current `experiment` stub prints "experiment stub").

- [ ] **Step 3: Replace the `experiment` stub** in `src/code_rag_eval/cli.py`. Add imports near the top:
```python
from code_rag_eval.eval.experiment import matrix_configs
from code_rag_eval.eval.report import load_results, comparison_table, write_decisions
```
Replace the `experiment` command with:
```python
@app.command()
def experiment(config: str = "configs/baseline.yaml",
               questions: str = "data/eval/questions.jsonl",
               results_dir: str = "results",
               dry_run: bool = False) -> None:
    """Run the full chunking x embedding x retrieval matrix and write a comparison + DECISIONS.md."""
    cfg = load_config(config)
    configs = matrix_configs(cfg)
    if dry_run:
        for c in configs:
            typer.echo(c.name)
        return
    load_dotenv()
    qpath = Path(questions)
    if not qpath.exists():
        typer.echo(f"eval set not found: {questions}. See README.")
        raise typer.Exit(code=1)
    records = load_eval_set(qpath)
    if not records:
        typer.echo(f"no questions in {questions}; nothing to evaluate.")
        raise typer.Exit(code=1)

    cache = EmbeddingCache(".emb_cache.sqlite")
    judge = AnthropicJudge(model=cfg.generation.model)
    out_dir = Path(results_dir)
    for c in configs:
        typer.echo(f"=== {c.name} ===")
        embed = CachedEmbeddingClient(make_embedding_client(c.embedding), cache)
        store = ChromaStore(collection_name=c.name, persist_dir=CHROMA_DIR)
        run_ingest(SOURCE_DIR, CORPUS_ROOT, c.chunking, embed, store)
        retriever = make_retriever(c, store, embed)
        llm = make_llm_client(c.generation)
        answer_fn = partial(generate_answer, llm=llm)
        rows = evaluate(records, retriever, answer_fn, judge)
        agg = aggregate(rows)
        write_report(c.name, rows, agg, out_dir)
        typer.echo(f"  hit@5={agg['hit_at_5']:.3f} mrr={agg['mrr']:.3f} faithfulness={agg['faithfulness']:.3f}")
    results = load_results(out_dir)
    typer.echo(comparison_table(results))
    write_decisions(results, Path("DECISIONS.md"))
    typer.echo("wrote DECISIONS.md")
```
(`partial`, `EmbeddingCache`, `CachedEmbeddingClient`, `AnthropicJudge`, `evaluate`, `aggregate`, `write_report`, `make_retriever`, `run_ingest`, `make_embedding_client`, `make_llm_client`, `generate_answer`, `load_eval_set`, `CHROMA_DIR`, `SOURCE_DIR`, `CORPUS_ROOT` are already imported/defined in cli.py from earlier tasks — verify and add any missing import.)

- [ ] **Step 4: Run** `uv run pytest tests/test_cli.py -v` → PASS. Then `uv run code-rag-eval experiment --help` and `uv run code-rag-eval experiment --dry-run` (offline; lists the 8 names). Do NOT run the live sweep.

- [ ] **Step 5: Commit**
```
git add src/code_rag_eval/cli.py tests/test_cli.py
git commit -m "feat: experiment sweep command (8-config matrix + report)"
```
Add the Co-Authored-By trailer.

---

## Phase 7 — Polish

### Task 11: Streamlit demo app

**Files:** Add `streamlit` dep; create `app/streamlit_app.py`. Test: `tests/test_streamlit_app.py`.

- [ ] **Step 1: Add the dependency** — run `uv add streamlit`.

- [ ] **Step 2: Write the failing test**

`tests/test_streamlit_app.py`:
```python
import importlib.util
from pathlib import Path


def test_streamlit_app_defines_main():
    p = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # runs module top-level; main() NOT called (guarded by __main__)
    assert callable(mod.main)
```

- [ ] **Step 3: Run** `uv run pytest tests/test_streamlit_app.py -v` → FAIL (file not found / no spec).

- [ ] **Step 4: Implement** `app/streamlit_app.py`:
```python
from __future__ import annotations
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from code_rag_eval.config import load_config
from code_rag_eval.factories import make_embedding_client, make_llm_client
from code_rag_eval.ingest.store import ChromaStore
from code_rag_eval.retrieve.factory import make_retriever
from code_rag_eval.generate.answer import generate_answer

CHROMA_DIR = ".chroma"
CONFIG = "configs/baseline.yaml"


def main() -> None:
    load_dotenv()
    st.title("code-rag-eval — FastAPI code Q&A")
    st.caption("Retrieval-augmented answers over the FastAPI source, with file:line citations.")
    cfg = load_config(CONFIG)
    question = st.text_input("Ask about the FastAPI codebase:")
    if not question:
        return
    embed = make_embedding_client(cfg.embedding)
    llm = make_llm_client(cfg.generation)
    store = ChromaStore(collection_name=cfg.name, persist_dir=CHROMA_DIR)
    retriever = make_retriever(cfg, store, embed)
    retrieved = retriever.retrieve(question, cfg.retrieval.top_k)
    answer = generate_answer(question, retrieved, llm)
    st.markdown(answer.text)
    st.subheader("Sources")
    for r in retrieved:
        st.write(f"`{r.chunk.file}:{r.chunk.start_line}-{r.chunk.end_line}` (score {r.score:.3f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run** `uv run pytest tests/test_streamlit_app.py -v` → PASS (module imports; `main` defined; no Streamlit server, no API call).

- [ ] **Step 6: Commit**
```
git add pyproject.toml uv.lock app/streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: Streamlit demo app (query + answer + sources)"
```
Add the Co-Authored-By trailer.

---

### Task 12: README + docs update

**Files:** Modify `README.md`. (Docs-only; no tests.)

- [ ] **Step 1: Update the status table** in `README.md` — flip these rows to ✅ Implemented: AST-aware (tree-sitter) chunking; code-specialized `voyage-code-3` embeddings; hybrid retrieval (BM25 + vector, RRF). Update the "8-config experiment sweep + DECISIONS.md" row to "✅ tooling built (run needs API keys)" and the Streamlit row to ✅ Implemented.

- [ ] **Step 2: Add the new commands to the Usage section.** After the `eval` subsection, add:
````markdown
### Run the experiment sweep

```bash
uv run code-rag-eval experiment --dry-run   # list the 8 configs (offline)
uv run code-rag-eval experiment             # full sweep: ingest+eval each config, writes results/ + DECISIONS.md
```

### Demo UI

```bash
uv run streamlit run app/streamlit_app.py
```
````
And update the options table to include the `experiment` row (`--config`, `--questions`, `--results-dir`, `--dry-run`).

- [ ] **Step 3: Update the config matrix note** to state the matrix is implemented, link `DECISIONS.md`, and note the sweep + finalized DECISIONS.md require API keys + the verified eval set. Update the Roadmap section: mark Phases 4–7 done except the live sweep run, the finalized DECISIONS.md numbers, and the 2-min video.

- [ ] **Step 4: Commit**
```
git add README.md
git commit -m "docs: update README for Phase 4-7 (AST, voyage, hybrid, sweep, UI)"
```
Add the Co-Authored-By trailer.

---

## Self-Review Notes

- **Spec coverage:** AST chunking (T2) ✓; code embedding (T3) ✓; hybrid BM25+vector RRF (T1,T5,T6,T7) ✓; 8-config sweep (T8,T10) ✓; DECISIONS.md generator + scaffold (T9) ✓; Streamlit UI (T11) ✓; README/diagram (T12) ✓. Reranking remains intentionally out of scope (the chosen matrix was core+hybrid). The 2-min video is not automatable.
- **No-keys compliance:** every task's tests use fakes/stubs or pure functions; the only API-touching code (`VoyageEmbeddingClient`, the live `experiment`/`ask`/`eval` runs, live Streamlit answers) is never executed by the implementer — verified via offline `--dry-run`/`--help`/import-smoke tests.
- **Type/interface consistency:** new retrievers honor the `retrieve(query, k) -> list[RetrievedChunk]` shape used by the harness and CLI; `make_retriever` consumes `ExperimentConfig`; `chunk_ast` returns the same `Chunk` type as `chunk_fixed`; `matrix_configs` returns `ExperimentConfig` objects consumed by the existing pipeline + harness.
- **Ordering/deps:** T1→T5/T6; T4→T7; T8→T10; T9→T10. Tasks are listed in a valid execution order.
