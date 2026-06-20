# code-rag-eval

**A code Q&A RAG system over the FastAPI codebase whose real deliverable is a rigorous, objective evaluation harness.**

Anyone can wire up retrieval-augmented generation. The value here is *proving whether retrieval finds the right code* with objective metrics, and *defending design choices with data*. In the code domain "the right code" is an objective fact — a specific function at a known location — which makes the retrieval metrics unusually trustworthy.

Ask a question like *"Where is the route handler built?"* and the system retrieves the most relevant FastAPI source chunks, answers from them, and cites `file:line`. Then an evaluation harness scores, against a hand-verified question set, whether retrieval surfaced the correct function and whether the answer is grounded in it.

> This is **Project 1 of 3** in a code/developer-tools portfolio. The retriever built here is designed to be reused by a SWE-bench issue-fixing agent (Project 2) and productionized for serving (Project 3).

---

## Status

**Phases 0–3 are complete**: a working baseline RAG pipeline plus the full evaluation harness, with 31 offline tests passing. Phases 4–7 (the experiment sweep and polish) are planned — see the [Roadmap](#roadmap).

| Capability | Status |
|---|---|
| Ingestion: walk → chunk → embed → vector store | ✅ Implemented |
| Fixed-size (line-window) chunking | ✅ Implemented |
| OpenAI `text-embedding-3-large` embeddings | ✅ Implemented |
| Chroma vector store (cosine) | ✅ Implemented |
| Vector top-k retrieval | ✅ Implemented |
| Prompt assembly + Claude answer with `file:line` citations | ✅ Implemented |
| Eval-set tooling (symbol enumerator, schema, draft generator) | ✅ Implemented |
| Retrieval metrics: hit-rate@k, recall@k, MRR | ✅ Implemented |
| Generation metrics: faithfulness, answer relevancy, context precision (LLM-judge) + citation accuracy | ✅ Implemented |
| On-disk embedding cache | ✅ Implemented |
| AST-aware (tree-sitter) chunking | 🔜 Phase 4 |
| Code-specialized `voyage-code-3` embeddings | 🔜 Phase 4 |
| Hybrid retrieval (BM25 + vector, RRF) | 🔜 Phase 4 |
| 8-config experiment sweep + `DECISIONS.md` | 🔜 Phase 5–6 |
| Streamlit demo UI | 🔜 Phase 7 |

Config values `ast` / `voyage` / `hybrid` are recognized but raise a clear "not until Phase 4" error, so adding them is additive — not a rewrite.

---

## Architecture

```
INGESTION (offline)
  FastAPI source ─► chunker {fixed | ast*} ─► embedding client {openai | voyage*}
                 ─► embedding cache ─► Chroma vector store (one collection per config)

QUERY (online)
  question ─► VectorRetriever (embed → top-k ANN)        (* bm25 / hybrid: Phase 4)
           ─► prompt assembly (file:line headers) ─► Claude ─► answer + file:line citations

EVALUATION (the deliverable)
  eval set {question, category, gold symbols/files/line-ranges, reference answer}
       ├─ RETRIEVAL: hit-rate@k, recall@k, MRR
       └─ GENERATION: faithfulness, answer relevancy, context precision (LLM-judge) + citation accuracy
              └─ run a config ─► results/<config>.json
```

The framework (Chroma) is glue. The **chunker, retriever, prompt assembly, LLM-judge, and the entire eval harness are hand-written** — this is the answer to "what did you build without the framework?"

Every seam is a small interface (`EmbeddingClient`, `VectorStore`, `LLMClient`, `LLMJudge`), so tests run fully offline against fakes and Phase 4 can swap in new implementations without touching callers.

---

## Project structure

```
code-rag-eval/
├── configs/
│   └── baseline.yaml             # the experiment config (chunking × embedding × retrieval × generation)
├── data/
│   ├── corpus/                   # FastAPI source (gitignored); COMMIT.txt pins the version
│   └── eval/questions.jsonl      # hand-verified eval set (you create this — see below)
├── scripts/
│   └── fetch_corpus.sh           # clone + pin the FastAPI corpus
├── src/code_rag_eval/
│   ├── config.py                 # pydantic experiment-config models + YAML loader
│   ├── paths.py                  # corpus path + module-prefix helpers
│   ├── types.py                  # Chunk, RetrievedChunk
│   ├── factories.py              # build embedding/LLM clients from config
│   ├── ingest/
│   │   ├── walk.py               # find .py files
│   │   ├── chunkers.py           # fixed-size line-window chunker (AST in Phase 4)
│   │   ├── embed.py              # EmbeddingClient protocol + OpenAI impl
│   │   ├── cache.py              # sqlite embedding cache + cached wrapper
│   │   ├── store.py              # Chroma vector store + Chunk↔metadata mapping
│   │   └── pipeline.py           # ingest orchestration
│   ├── retrieve/
│   │   └── vector.py             # VectorRetriever
│   ├── generate/
│   │   ├── prompt.py             # hand-written prompt assembly
│   │   └── answer.py             # LLM call + file:line citation extraction
│   ├── eval/
│   │   ├── symbols.py            # stdlib-ast symbol enumerator (ground truth)
│   │   ├── dataset.py            # EvalRecord schema + jsonl IO
│   │   ├── generate_questions.py # LLM draft tool (output is human-verified)
│   │   ├── coverage.py           # chunk ↔ gold-symbol line-overlap
│   │   ├── retrieval_metrics.py  # hit-rate@k, recall@k, MRR
│   │   ├── judge.py              # LLMJudge protocol + Anthropic judge
│   │   ├── generation_metrics.py # faithfulness, relevancy, precision, citation accuracy
│   │   └── harness.py            # evaluate → aggregate → write report
│   └── cli.py                    # ingest | ask | draft-questions | eval | experiment
├── tests/                        # one test module per source module; all offline (fakes)
└── docs/superpowers/
    ├── specs/2026-06-19-code-rag-eval-design.md   # the design spec
    └── plans/2026-06-19-code-rag-eval-phase0-3.md # the implementation plan
```

---

## Getting started

### Prerequisites

- **[uv](https://docs.astral.sh/uv/)** (manages Python + dependencies)
- **Python 3.12** — pinned via `.python-version`; uv will fetch it automatically. (The project requires ≥3.11; 3.12 is pinned because some dependencies lag on 3.14.)
- **git** (to fetch the corpus)
- API keys for the live commands (not needed for the test suite):
  - `OPENAI_API_KEY` — embeddings (`text-embedding-3-large`)
  - `ANTHROPIC_API_KEY` — generation + the LLM judge (`claude-sonnet-4-6`)

### Install

```bash
uv sync
```

### Configure secrets

```bash
cp .env.example .env
# then edit .env and fill in:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
```

The CLI loads `.env` automatically. `.env` is gitignored.

### Fetch the corpus

```bash
bash scripts/fetch_corpus.sh 0.115.0
```

This shallow-clones FastAPI at the given tag into `data/corpus/fastapi/` (gitignored) and records the exact tag + commit SHA in `data/corpus/COMMIT.txt` for reproducibility. The corpus source is **not** committed — only the pin record and the fetch script are.

---

## Usage

All commands run via `uv run code-rag-eval <command>`.

### Ingest the corpus

Chunk + embed the FastAPI source into a Chroma collection (one collection per config name, persisted under `.chroma/`):

```bash
uv run code-rag-eval ingest
# ingested 1234 chunks into collection 'baseline'
```

### Ask a question

```bash
uv run code-rag-eval ask "Where is the APIRoute request handler built?"
```

```
The request handler is built in APIRoute.get_route_handler in
fastapi/routing.py:421, which calls get_request_handler(...) (fastapi/routing.py:217).

--- sources ---
  fastapi/routing.py:400-460 (score 0.812)
  fastapi/routing.py:200-260 (score 0.741)
  ...
```

The model is instructed to answer **only** from retrieved context and to cite every claim as `file:line`.

### Run the evaluation harness

Scores a config over the hand-verified question set and writes a JSON report to `results/<config>.json`:

```bash
uv run code-rag-eval eval
# evaluated 50 questions -> results/baseline.json
#   hit_at_1: 0.620
#   hit_at_5: 0.840
#   mrr: 0.710
#   faithfulness: 0.880
#   citation_accuracy: 0.760
```

> Requires `data/eval/questions.jsonl` to exist — see [Building the eval set](#building-the-eval-set). The embedding cache (`.emb_cache.sqlite`) means re-runs only pay for new texts.

### Options

| Command | Key options |
|---|---|
| `ingest` | `--config configs/baseline.yaml` |
| `ask QUESTION` | `--config configs/baseline.yaml` |
| `draft-questions` | `--config`, `--per-category 15`, `--out data/eval/candidates.jsonl` |
| `eval` | `--config`, `--questions data/eval/questions.jsonl`, `--results-dir results` |
| `experiment` | *(Phase 5 — the full config sweep)* |

---

## Configuration

A run is fully described by one YAML file. `configs/baseline.yaml`:

```yaml
name: baseline
chunking:
  strategy: fixed        # fixed | ast (ast = Phase 4)
  window_lines: 40
  overlap_lines: 10
embedding:
  provider: openai       # openai | voyage (voyage = Phase 4)
  model: text-embedding-3-large
retrieval:
  method: vector         # vector | bm25 | hybrid (bm25/hybrid = Phase 4)
  top_k: 5
generation:
  provider: anthropic
  model: claude-sonnet-4-6
```

The planned experiment matrix (Phase 5) sweeps **chunking {fixed, AST} × embedding {OpenAI-general, Voyage-code} × retrieval {vector, hybrid}** = 8 configs, with a BM25-only reference baseline, to answer: *does a code-trained embedding and/or AST-aware chunking and/or lexical hybrid retrieval find the right FastAPI function more reliably?*

---

## Evaluation methodology

The eval harness is the point of the project.

### Eval set

~50 hand-verified records (`data/eval/questions.jsonl`), balanced across four question categories:

- **locate** — "where is X defined?"
- **explain** — "how does Y work?"
- **trace** — "what calls / uses Z?"
- **behavior** — "what does X return when …?"

Each record carries the question, its category, the **ground-truth** symbol(s)/file(s)/line-range(s), and a reference answer:

```json
{
  "id": "locate-0001",
  "category": "locate",
  "question": "Where is the APIRoute request handler built?",
  "gold_symbols": ["fastapi.routing.APIRoute.get_route_handler"],
  "gold_files": ["fastapi/routing.py"],
  "gold_line_ranges": [[400, 460]],
  "reference_answer": "In APIRoute.get_route_handler."
}
```

Ground truth comes from a stdlib-`ast` **symbol enumerator** that lists every function/class/method in the corpus with its exact line range — so "the right code" is a verifiable location, not a judgment call.

### Metrics

**Retrieval** (computed at k ∈ {1, 3, 5, 10}):

| Metric | Meaning |
|---|---|
| **hit-rate@k** | Did *any* gold symbol's chunk land in the top-k? (handles answers spanning files) |
| **recall@k** | Fraction of gold symbols whose chunk is in the top-k |
| **MRR** | Reciprocal rank of the first gold-symbol chunk |

A retrieved chunk "covers" a gold symbol when it is in the same file and its line range **overlaps** the gold range — robust to chunk boundaries (works for both fixed and AST chunking).

**Generation:**

| Metric | Meaning |
|---|---|
| **faithfulness** | Is the answer grounded in the retrieved context? (LLM-judge, 0–1) |
| **answer relevancy** | Does the answer address the question? (LLM-judge, 0–1) |
| **context precision** | Fraction of retrieved chunks the judge deems relevant |
| **citation accuracy** | Fraction of the answer's `file:line` citations that fall inside a gold range (deterministic) |

> **Note on RAGAS:** the design names RAGAS for the LLM-judged metrics; they are implemented here as a self-contained `LLMJudge` (own prompts, behind a protocol) for full offline-testability and to avoid coupling to a fast-moving third-party API. Swapping in RAGAS as an industry-standard cross-check is a planned follow-up. Methodology is framed against **COIR (Code Information Retrieval)**, the recognized code-retrieval benchmark.

### Building the eval set

The verified question set is the one deliberately manual deliverable (its honesty is the whole point):

```bash
# 1. Draft candidates with an LLM from real corpus symbols:
uv run code-rag-eval draft-questions --per-category 15
#    → writes data/eval/candidates.jsonl

# 2. Review EVERY candidate by hand: fix wording, confirm the gold
#    symbol/file/line-range is correct, drop bad ones, balance categories.

# 3. Save the curated, verified set to data/eval/questions.jsonl
```

`candidates.jsonl` stays gitignored; the verified `questions.jsonl` is committed as a project artifact.

---

## Testing

The full suite is **offline** — no API keys, no network. Embedding/LLM/judge calls are exercised through deterministic fakes (`tests/fakes.py`), and the real OpenAI/Anthropic clients are lazy-imported so importing the package never requires a key.

```bash
uv run pytest -q
# 31 passed
```

There is one focused test module per source module, written test-first (TDD).

---

## Design & decisions

- **Design spec:** [`docs/superpowers/specs/2026-06-19-code-rag-eval-design.md`](docs/superpowers/specs/2026-06-19-code-rag-eval-design.md)
- **Implementation plan (Phases 0–3):** [`docs/superpowers/plans/2026-06-19-code-rag-eval-phase0-3.md`](docs/superpowers/plans/2026-06-19-code-rag-eval-phase0-3.md)
- **`DECISIONS.md`** (the data-backed writeup of the winning config + trade-offs) lands after the Phase 5 experiment sweep.

---

## Roadmap

| Phase | Work |
|---|---|
| **4** | AST-aware chunking (tree-sitter), `voyage-code-3` embeddings, hybrid retrieval (BM25 + vector via Reciprocal Rank Fusion) |
| **5** | Run the full 8-config experiment sweep; tabulate to `results/` |
| **6** | `DECISIONS.md`: winning config + trade-offs, COIR-framed methodology |
| **7** | Streamlit demo (query box + answer + shown source), architecture diagram, walkthrough |

---

## Tech stack

Python 3.12 · [uv](https://docs.astral.sh/uv/) · [Typer](https://typer.tiangolo.com/) (CLI) · [Pydantic](https://docs.pydantic.dev/) · [ChromaDB](https://www.trychroma.com/) · OpenAI (`text-embedding-3-large`) · Anthropic (`claude-sonnet-4-6`) · pytest. Corpus: [FastAPI](https://github.com/fastapi/fastapi) 0.115.0.
