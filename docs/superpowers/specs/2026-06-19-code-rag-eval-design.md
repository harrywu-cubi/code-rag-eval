# Design: `code-rag-eval` — Code Q&A RAG with a Rigorous Evaluation Harness

**Status:** Approved design (2026-06-19) · **Next:** implementation plan (writing-plans)

## Context

This is **Project 1 of 3** in a code/developer-tools portfolio aimed at AI/ML Engineer,
Retrieval Engineer, and AI Infra roles. Anyone can build RAG; the value here is
**proving whether retrieval finds the right code** with objective metrics and
**defending design choices with data**. In the code domain "the right code" is an
objective fact (a specific function at a known location), which makes retrieval metrics
unusually trustworthy.

The repo is greenfield (only an idea brief + stub README exist today). This document is
the design we build from. The retriever built here is intended to be **reused by Project 2
(`swe-agent-bench`)** as its code-search tool, and the serving path is **productionized by
Project 3 (`code-llm-serving`)** — so interfaces are designed with those reuses in mind,
but neither sibling project is required to build this one.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Corpus** | **FastAPI** (`fastapi/fastapi`, pinned commit) | Modern, heavily documented, rich type hints; recognizable; clean structure → easy to hand-verify ground truth. |
| **Experiment matrix** | **AST vs fixed × general vs code embedding × vector vs hybrid** = 8 configs | Hybrid (BM25+vector) is well-motivated for code's exact symbol names → a strong, data-backed differentiator. |
| **Models** | **API-hosted** embeddings + LLM | Corpus is public OSS (no bank-data/egress concern); fastest path to strong quality. |
| **UI** | **CLI + thin Streamlit** | CLI drives eval/experiments; minimal Streamlit demo for the 2-min walkthrough. |

**Model picks (API):**

- General embedding: **OpenAI `text-embedding-3-large`** (standard general-purpose baseline)
- Code-specialized embedding: **Voyage `voyage-code-3`** (leading code-retrieval model; ~10% above general models on code benchmarks)
- Generation LLM: **Claude Sonnet** (default; swappable via config — citation parsing is model-agnostic)
- *Methodology refinement noted in DECISIONS.md:* a within-vendor control (Voyage general
  `voyage-3.x-large` vs `voyage-code-3`) isolates the general-vs-code variable from vendor
  differences. The primary comparison stays OpenAI-general vs Voyage-code (the real-world choice).

**Practical note:** API calls need outbound access to OpenAI / Voyage / Anthropic and API
keys. The model layer sits behind an interface, so local HuggingFace models can be swapped
in (e.g., on a locked-down machine) without touching the rest of the system.

## Resolved design questions

- **Chunk granularity:** function/method = primary unit; class-header chunks carry the
  class docstring/signature (method bodies are their own chunks). Metadata on every chunk:
  file path, fully-qualified symbol, signature, docstring, line range. Long functions →
  overlapping windows that **repeat the signature+docstring header** so each piece stays
  self-describing. Trivial one-liners → kept (cheap) or merged.
- **Context expansion / call-graph:** **deferred.** Pure top-k for the core experiment;
  call-graph expansion would confound the chunking comparison and bloat prompts.
- **Reranking:** **deferred** (not in the chosen matrix); the retriever interface stays
  rerank-ready so it can be added as a later variable.
- **Corpus scope:** one repo to start (cleaner ground truth, still a real challenge);
  code-only for the core retrieval metric. A "code+docs" variant is an optional later experiment.

## Architecture

```
INGESTION (offline, per config)
  FastAPI source ─► chunker {AST | fixed} ─► embedding client {OpenAI | Voyage}
                 ─► embedding cache ─► vector store (Chroma, one collection per config)
                 └─► BM25 index (per chunking config, embedding-independent)

QUERY (online)
  question ─► Retriever(method ∈ {vector, bm25, hybrid})
                vector: embed → top-n ANN
                bm25:   code-aware tokenize → top-n lexical
                hybrid: RRF fusion of the two
           ─► top-k chunks ─► prompt assembly (file:line headers) ─► LLM
           ─► answer + file:line citations

EVALUATION (the deliverable)
  eval set {question, category, gold symbols/files/lines, reference answer}
       ├─ RETRIEVAL: hit-rate@k, recall@k, MRR, context precision
       └─ GENERATION: faithfulness, answer relevancy (RAGAS) + citation accuracy
              └─ run all 8 configs ─► results tables ─► winner ─► DECISIONS.md
```

The framework (LlamaIndex) is glue only — embedding clients / store wrappers.
**Hand-written:** the AST chunker (tree-sitter), the retriever, the RRF fusion, prompt
assembly, and the entire eval harness. This answers "what did you build without the framework?"

## Repo structure

```
code-rag-eval/
  README.md                  # architecture diagram + run instructions
  DECISIONS.md               # data-backed writeup (key deliverable)
  pyproject.toml
  configs/                   # one YAML per experiment config (chunking × embedding × retrieval)
  data/
    corpus/                  # FastAPI source, pinned commit (gitignored or submodule)
    eval/questions.jsonl     # ~50 hand-verified triples
  src/code_rag_eval/
    ingest/
      chunkers.py            # AST (tree-sitter) + fixed-size; shared Chunk dataclass
      embed.py               # embedding clients (OpenAI, Voyage) + on-disk cache
      store.py               # VectorStore interface; Chroma impl (pgvector later)
      pipeline.py            # ingest orchestration per config
    retrieve/
      vector.py  bm25.py  hybrid.py  retriever.py   # unified config-driven Retriever
    generate/
      prompt.py              # hand-written prompt assembly
      answer.py              # LLM call + file:line citation extraction
    eval/
      retrieval_metrics.py   # hit-rate@k, recall@k, MRR, context precision
      generation_metrics.py  # RAGAS faithfulness/answer-relevancy + citation accuracy
      harness.py             # config → metrics → results
      report.py              # tabulate configs, pick winner
    cli.py                   # ingest | ask | eval | experiment
  app/streamlit_app.py       # thin demo: query box + answer + shown source
  results/                   # experiment outputs (json + markdown tables)
  tests/
```

## Component interfaces (the seams that matter)

- `Chunk`: `{text, file, symbol (qualified), signature, docstring, start_line, end_line, kind}`.
- `Chunker.chunk(file) -> list[Chunk]` — two impls (AST, fixed). Fixed is deliberately naive
  (token windows + overlap) to contrast against AST.
- `EmbeddingClient.embed(texts) -> vectors` — cached on disk keyed by `(model, content_hash)`
  so the 8-config matrix does not re-pay for embeddings.
- `VectorStore` — `add`, `query(vector, n)`; Chroma now, pgvector swap later (Project 3).
- `Retriever.retrieve(query, k) -> list[RetrievedChunk]` — method-agnostic; hybrid uses
  **Reciprocal Rank Fusion (k=60)** over the vector and BM25 result lists.
- BM25: `rank_bm25` over **code-aware tokens** (split identifiers, split snake_case/camelCase)
  so symbol names match lexically.

## Evaluation methodology

**Eval set** (~50 records, `data/eval/questions.jsonl`), four categories balanced:

- **locate** ("where is X defined") · **explain** ("how does Y work") ·
  **trace** ("what calls / uses Z") · **behavior** ("what does this return when …")

Record schema:

```json
{
  "id": "...",
  "category": "locate|explain|trace|behavior",
  "question": "...",
  "gold_symbols": ["fastapi.routing.APIRoute.get_route_handler"],
  "gold_files": ["fastapi/routing.py"],
  "gold_line_ranges": [[120, 180]],
  "reference_answer": "..."
}
```

Construction: a tool draws a symbol from the AST index and asks an LLM to draft a question +
reference answer; **every record is hand-verified/edited** before it enters the set.

**Retrieval metrics** (k ∈ {1, 3, 5, 10}):

- **hit-rate@k** — *any* gold symbol's chunk in top-k
- **recall@k** — fraction of gold symbols whose chunk is in top-k (handles multi-file answers)
- **MRR** — reciprocal rank of the first gold-symbol chunk
- **context precision** (RAGAS) — fraction of retrieved chunks that are relevant
- *Chunk→symbol mapping:* a chunk "covers" a gold symbol if it overlaps the gold line range
  in the gold file (AST chunk = symbol directly; fixed chunk = line-overlap test).
- *Multi-file "hit" rule:* hit = any gold symbol retrieved; also report a strict
  **all-gold-symbols-in-top-k** rate.

**Generation metrics:**

- **faithfulness** + **answer relevancy** (RAGAS) — is the answer grounded in retrieved context
- **citation accuracy** (custom) — cited `file:line` falls within a gold line range

## Experiment matrix (8 configs + 1 reference)

`chunking{AST, fixed} × embedding{OpenAI-general, Voyage-code} × retrieval{vector, hybrid}`
Plus **BM25-only** as an embedding-independent reference baseline (reported once).
Output: a metrics table per config in `results/`, then a winner + trade-off narrative.
Expected direction (to be confirmed by the run, not assumed): the winning config materially
improves correct-symbol hit-rate and answer faithfulness over the naive baseline.

## Build phases

0. **Scaffold** — package layout, config schema, CLI skeleton; clone + **pin FastAPI** at a commit.
1. **Baseline end-to-end** — fixed chunker · OpenAI embed · Chroma · vector top-k ·
   prompt assembly · LLM answer with citations · `cli ask`. Get it answering at all.
2. **Eval set** — schema + draft-generation tool; build & hand-verify ~50 triples.
3. **Eval harness** — retrieval + generation metrics against the baseline; embedding cache.
4. **Add variables** — AST chunker (tree-sitter), Voyage `voyage-code-3`, BM25 + RRF hybrid;
   everything config-driven so the harness can sweep the matrix.
5. **Run experiments** — full 8-config sweep; tabulate to `results/`.
6. **Decide with data** — `DECISIONS.md`: winner, trade-offs, COIR-framed methodology notes.
7. **Polish** — Streamlit demo, README + architecture diagram, 2-min walkthrough video.

## Definition of done

- [ ] Answers FastAPI code questions with `file:line` citations.
- [ ] Eval harness comparing AST vs fixed chunking × 2 embeddings × vector vs hybrid (8 configs).
- [ ] `DECISIONS.md` with the winning config + trade-offs (+ BM25-only baseline reference).
- [ ] README with architecture diagram + run instructions; 2-min walkthrough.

## Pitfalls to actively avoid

- Fixed-size chunks splitting functions mid-body — the AST comparison should *prove* this hurts.
- Eval set < 30 — proves nothing; target ~50, hand-verified.
- Measuring only the final answer — retrieval is scored separately and first.
- Leaning on framework defaults — chunker/retriever/fusion/prompt/eval are hand-written.
- Forgetting the embedding cache — without it the 8-config sweep is slow and costly.

## References

- **COIR (Code Information Retrieval)** — the recognized academic benchmark for code retrieval;
  used to frame methodology credibly in DECISIONS.md.
