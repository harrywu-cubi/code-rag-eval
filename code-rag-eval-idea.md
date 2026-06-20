# Idea Brief — `code-rag-eval`

> **Project 1 of 3.** Code Q&A RAG with a rigorous evaluation harness.
> **What this file is:** a self-contained brief to brainstorm and build with Claude Code. Decisions here are starting points, not final — the "Open questions" section is what to work through first.

---

## How to use this file with Claude Code

1. Drop this whole file into a Claude Code session as context.
2. Start by resolving the **Open questions** below (don't write code yet).
3. Then have it propose a repo structure + a Phase-1 plan, and build incrementally.
4. A ready-to-paste **kickoff prompt** is at the bottom.

## Context (why this project exists)

- It's the **foundation** of a 3-project portfolio in the **code / developer-tools** domain, aimed at AI/ML Engineer, AI Agent Engineer, and AI Infra roles. The other two: `swe-agent-bench` (a SWE-bench issue-fixing agent that **reuses this project's retriever as its code-search tool**) and `code-llm-serving` (productionizes one of the two).
- **Target roles for this repo specifically:** AI/ML Engineer, Retrieval Engineer.
- **The differentiator:** anyone can build RAG. The value is *proving whether it retrieves the right code* with objective metrics, and *defending the design choices with data*. In the code domain, "the right code" is an objective fact (a specific function at a known location), which makes the retrieval metrics unusually trustworthy.

## What you're building (mental model)

An LLM doesn't know a specific codebase and will hallucinate about it. RAG fixes this: at query time, retrieve the most relevant code chunks and feed them into the prompt so the model answers from real source.

- **Ingestion (offline):** repo → split into chunks → embed each → store vectors.
- **Query (online):** question ("which function validates the auth token?") → embed → retrieve top-k code chunks → assemble prompt → LLM answers, citing `file:line`.
- **Evaluation (the real deliverable):** a harness that measures whether retrieval finds the correct code and whether answers are grounded in it.

## Architecture

```
INGESTION
  repo files ──► AST-aware chunker ──► embedding model ──► vector store (Chroma / pgvector)
                 (split by function/class, not character count)

QUERY
  question ──► embed ──► vector search (top-k) ──► [optional: rerank]
            ──► prompt (code chunks + question) ──► LLM ──► answer + file:line citations

EVALUATION  (differentiator)
  test set {question, correct file/symbol, reference answer}
       ├─► RETRIEVAL: hit-rate/recall@k, MRR, context precision
       └─► GENERATION: faithfulness, answer relevance
              └─► compare configs (chunking × embedding) ──► winner ──► DECISIONS.md
```

## Stack (starting point)

- **Python.**
- **LlamaIndex or LangChain** — code-aware splitters + retrieval glue. *Write the prompt-assembly or a custom retriever yourself* so you can answer "what did you build without the framework?"
- **`tree-sitter`** — AST-aware chunking (split on function/class boundaries; never cut a function in half).
- **Embeddings (compare two):** a general model (e.g., `bge`/`e5` family) vs a code-trained embedding model.
- **Vector store:** Chroma to start (zero setup) → pgvector for production credibility and reuse by Projects 2/3.
- **Eval:** RAGAS (faithfulness, answer relevancy, context precision/recall) + a thin custom harness for code-specific retrieval scoring.
- **LLM API:** any.
- **UI (optional):** Streamlit/Gradio — query box + answer + shown source.

## Build phases

1. **Get + inspect the corpus.** Understand structure.
2. **Baseline pipeline.** Fixed-size chunking, one embedding model, top-k. Get it answering at all.
3. **Eval set (crucial).** 30–50 `{question, correct symbol, reference answer}` triples; semi-generate then **verify by hand**.
4. **Eval harness.** Retrieval metrics (hit-rate/recall@k, MRR, context precision) + generation metrics (faithfulness, answer relevance).
5. **Experiments.** AST-aware vs fixed-size chunking × general vs code embeddings. Tabulate.
6. **Decide with data → DECISIONS.md.**
7. **Polish.** UI, README + architecture diagram, 2-min video.

## Open questions to brainstorm (resolve these first)

1. **Corpus choice.** Which repo(s)? Criteria: substantial (tens of thousands of LOC), clear structure, good docstrings, familiar to you. One repo or 2–3? Include the repo's markdown docs in the corpus, or code only?
2. **Chunk granularity.** Function-level vs class-level vs sliding window over functions. How to handle very long functions and trivial one-liners? Attach signatures/docstrings/file path as metadata?
3. **Context expansion.** Should retrieval pull in called functions / imports (call-graph awareness), or stay pure top-k similarity? Trade-off: recall vs prompt bloat.
4. **Retrieval method.** Pure vector vs **hybrid (BM25 + vector)** — code has exact symbol names where lexical matching helps. Worth adding hybrid as a variable, or keep the experiment to chunking × embeddings?
5. **Embedding picks.** Which specific general model and which code-trained model to compare?
6. **Eval-set construction.** How to semi-automate question generation while keeping ground truth honest? How many? Which question categories — locate ("where is X"), explain ("how does Y work"), trace ("what calls Z"), behavior ("what does this return when…")?
7. **Metrics + thresholds.** Exactly which metrics, and what counts as a "hit" when the correct answer spans multiple files?
8. **Reranking.** Add a reranker as a third experiment variable, or keep scope tight?
9. **UI scope.** CLI-only vs a small Streamlit demo — how much to invest given Project 3 will deploy this anyway?

## Pitfalls to avoid

- Fixed-size chunks that split functions mid-body (your AST comparison should *prove* this hurts).
- Eval set too small (<30) — proves nothing.
- Measuring only the final answer, not retrieval separately.
- A trivial repo that isn't a real retrieval challenge.
- Leaning entirely on framework defaults.

## Definition of done

- [ ] Answers code questions with `file:line` citations.
- [ ] Eval harness comparing AST-aware vs fixed-size chunking × 2 embedding models.
- [ ] DECISIONS.md with the winning config + trade-offs.
- [ ] README with architecture diagram + run instructions; 2-min walkthrough.

## Resume bullets this should produce

- *Built a code-RAG system (Python, LlamaIndex, tree-sitter AST chunking, pgvector) with an evaluation harness; raised correct-symbol retrieval hit-rate 0.61→0.84 and faithfulness 0.70→0.88 across a 50-question test set.*
- *Benchmarked AST-aware vs fixed-size chunking and general vs code-trained embeddings; documented trade-offs in a public DECISIONS.md.*

---

## Kickoff prompt for Claude Code

> I'm building `code-rag-eval`, a code Q&A RAG system whose real point is a rigorous, objective evaluation harness (retrieval hit-rate on the correct function + answer faithfulness). The brief is in this file. Before writing any code, walk me through the "Open questions" section and recommend a concrete choice for each, with reasoning — start by proposing 2–3 candidate repos for the corpus and which you'd pick. Then propose a clean repo structure and a Phase-1 plan (baseline pipeline only). Keep the framework as a dependency but plan for me to write the retriever/prompt-assembly myself.
