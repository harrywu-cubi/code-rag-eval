# code-rag-eval

A code Q&A RAG system over the FastAPI codebase, built to measure — objectively — whether retrieval finds the right code and whether the answer is grounded in it.

The interesting problem here is evaluation. In the code domain "the right code" is an objective fact — a specific function at a known location — so retrieval quality can be measured against verifiable ground truth instead of judgment calls. This project leans on that:

- **Objective, AST-based ground truth.** A stdlib-`ast` symbol enumerator lists every function/class/method in the corpus with its exact line range. "Did retrieval surface the right code?" reduces to a line-range overlap check, not an opinion.
- **Retrieval and generation are measured separately.** Retrieval metrics (hit-rate@k, recall@k, MRR) and generation metrics (faithfulness, answer relevancy, context precision, citation accuracy) are computed independently, so a good answer over bad context — or vice versa — is visible rather than averaged away.
- **A hand-written evaluation harness.** The chunker, retriever, prompt assembly, LLM judge, and the entire eval pipeline are written directly against small interfaces. The vector store (Chroma) is the only framework, and it's just glue.
- **An 8-config experiment matrix.** chunking {fixed, AST} × embedding {OpenAI-general, Voyage-code} × retrieval {vector, hybrid} = 8 configs, plus a BM25-only reference, run through one sweep command to answer a concrete question with data: *does code-trained embedding and/or AST-aware chunking and/or lexical-hybrid retrieval find the right FastAPI function more reliably?*

Ask a question like *"Where is the route handler built?"* and the system retrieves the most relevant FastAPI source chunks, answers from them, and cites `file:line`. The harness then scores that behavior against a hand-verified question set.

> Project 1 of 2 in a code/developer-tools portfolio. The retriever built here is designed to be reused by a SWE-bench issue-fixing agent (Project 2).

---

## Architecture

```
INGESTION (offline)
  FastAPI source ─► chunker {fixed | ast} ─► embedding client {openai | voyage}
                 ─► embedding cache ─► Chroma vector store (one collection per config)

QUERY (online)
  question ─► retriever {vector | bm25 | hybrid-RRF} (embed → top-k ANN or BM25 or both)
           ─► prompt assembly (file:line headers) ─► Claude ─► answer + file:line citations

EVALUATION
  eval set {question, category, gold symbols/files/line-ranges, reference answer}
       ├─ RETRIEVAL: hit-rate@k, recall@k, MRR
       └─ GENERATION: faithfulness, answer relevancy, context precision (LLM-judge) + citation accuracy
              └─ run a config ─► results/<config>.json ─► DECISIONS.md (experiment sweep)
```

Every seam is a small interface (`EmbeddingClient`, `VectorStore`, `LLMClient`, `LLMJudge`), so the test suite runs fully offline against fakes and new implementations slot in without touching callers.

---

## Project structure

```
code-rag-eval/
├── app/
│   └── streamlit_app.py          # Streamlit demo: query box → answer + shown sources
├── configs/
│   └── baseline.yaml             # an experiment config (chunking × embedding × retrieval × generation)
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
│   │   ├── chunkers.py           # fixed-size line-window chunker + AST (tree-sitter) chunker
│   │   ├── embed.py              # EmbeddingClient protocol + OpenAI + Voyage impls
│   │   ├── cache.py              # sqlite embedding cache + cached wrapper
│   │   ├── store.py              # Chroma vector store + Chunk↔metadata mapping
│   │   └── pipeline.py           # ingest orchestration
│   ├── retrieve/
│   │   ├── tokenize.py           # code-aware tokenizer
│   │   ├── vector.py             # VectorRetriever
│   │   ├── bm25.py               # BM25Retriever
│   │   ├── hybrid.py             # HybridRetriever (RRF fusion)
│   │   └── factory.py            # build retriever from config
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
│   │   ├── harness.py            # evaluate → aggregate → write report
│   │   ├── experiment.py         # 8-config experiment matrix + sweep runner
│   │   └── report.py             # DECISIONS.md generator
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
#   hit_at_1, hit_at_5, mrr, faithfulness, citation_accuracy, ...
```

> Requires `data/eval/questions.jsonl` to exist — see [Building the eval set](#building-the-eval-set). The embedding cache (`.emb_cache.sqlite`) means re-runs only pay for new texts.

### Run the experiment sweep

```bash
uv run code-rag-eval experiment --dry-run   # list the 8 configs (offline, no keys)
uv run code-rag-eval experiment             # full sweep: ingest+eval each config; writes results/ + DECISIONS.md
```

The 8-config matrix is **chunking {fixed, AST} × embedding {OpenAI-general, Voyage-code} × retrieval {vector, hybrid}**. The live sweep needs API keys and a verified `data/eval/questions.jsonl`; `--dry-run` works offline.

### Demo UI

```bash
uv run streamlit run app/streamlit_app.py
```

A thin query box → answer + shown sources (`file:line`). Requires an ingested corpus and API keys.

### Options

| Command | Key options |
|---|---|
| `ingest` | `--config configs/baseline.yaml` |
| `ask QUESTION` | `--config configs/baseline.yaml` |
| `draft-questions` | `--config`, `--per-category 15`, `--out data/eval/candidates.jsonl` |
| `eval` | `--config`, `--questions data/eval/questions.jsonl`, `--results-dir results` |
| `experiment` | `--config`, `--questions`, `--results-dir`, `--dry-run` |

---

## Configuration

A run is fully described by one YAML file. `configs/baseline.yaml`:

```yaml
name: baseline
chunking:
  strategy: fixed        # fixed | ast
  window_lines: 40
  overlap_lines: 10
embedding:
  provider: openai       # openai | voyage
  model: text-embedding-3-large
retrieval:
  method: vector         # vector | bm25 | hybrid
  top_k: 5
generation:
  provider: anthropic
  model: claude-sonnet-4-6
```

Config values `ast` / `voyage` / `hybrid` are fully supported — use them in any YAML config or let the `experiment` command sweep them. The matrix sweeps **chunking {fixed, AST} × embedding {OpenAI-general, Voyage-code} × retrieval {vector, hybrid}** = 8 configs, plus a BM25-only reference baseline.

---

## Evaluation methodology

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

The LLM-judged metrics are implemented as a self-contained `LLMJudge` (own prompts, behind a protocol) for full offline-testability and to avoid coupling to a fast-moving third-party API. Swapping in RAGAS as an industry-standard cross-check is a planned follow-up. Methodology is framed against **COIR (Code Information Retrieval)**, the recognized code-retrieval benchmark.

### Building the eval set

The verified question set is the one deliberately manual deliverable — its correctness is what makes the metrics trustworthy:

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

## Results

The harness, metrics, and the 8-config matrix are implemented and exercised by the offline test suite. The comparison table across all configs and the winning-config writeup are produced by running the sweep against a live API and a verified eval set:

```bash
uv run code-rag-eval experiment   # writes results/<config>.json + regenerates DECISIONS.md
```

`DECISIONS.md` is the data-backed writeup — the comparison table and the named winning config are filled in by that run. The Streamlit demo app (`app/streamlit_app.py`) drives the end-to-end flow — question in, answer out, sources shown as `file:line` — and a screen-recorded walkthrough will accompany the first live run.

---

## Testing

The full suite is **offline** — no API keys, no network. Embedding/LLM/judge calls are exercised through deterministic fakes (`tests/fakes.py`), and the real OpenAI/Anthropic clients are lazy-imported so importing the package never requires a key.

```bash
uv run pytest -q
# 55 passed
```

There is one focused test module per source module, written test-first (TDD).

---

## Design & decisions

- **Design spec:** [`docs/superpowers/specs/2026-06-19-code-rag-eval-design.md`](docs/superpowers/specs/2026-06-19-code-rag-eval-design.md)
- **Implementation plan (Phases 0–3):** [`docs/superpowers/plans/2026-06-19-code-rag-eval-phase0-3.md`](docs/superpowers/plans/2026-06-19-code-rag-eval-phase0-3.md)
- **[`DECISIONS.md`](DECISIONS.md)** — the data-backed writeup of the winning config + trade-offs; generated by `uv run code-rag-eval experiment` and filled in by a live run.

---

## Tech stack

Python 3.12 · [uv](https://docs.astral.sh/uv/) · [Typer](https://typer.tiangolo.com/) (CLI) · [Pydantic](https://docs.pydantic.dev/) · [ChromaDB](https://www.trychroma.com/) · OpenAI (`text-embedding-3-large`) · Anthropic (`claude-sonnet-4-6`) · pytest. Corpus: [FastAPI](https://github.com/fastapi/fastapi) 0.115.0.
</content>
</invoke>
