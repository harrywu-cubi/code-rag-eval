from __future__ import annotations
import re
from typing import Protocol


class LLMJudge(Protocol):
    def is_relevant(self, question: str, chunk_text: str) -> bool: ...
    def faithfulness(self, answer: str, context: str) -> float: ...
    def answer_relevancy(self, question: str, answer: str) -> float: ...


def _first_float(text: str, default: float = 0.0) -> float:
    # Match a standalone 0..1 score token only: "0", "1", "0.85", "1.0".
    # Word boundaries prevent mis-reading the "1" inside "10" or "7/10" as 1.0.
    m = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text)
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
