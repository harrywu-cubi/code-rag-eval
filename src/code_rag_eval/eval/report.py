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
