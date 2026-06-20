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
