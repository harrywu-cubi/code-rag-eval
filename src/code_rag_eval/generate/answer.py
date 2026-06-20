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

    def __init__(self, model: str = "claude-sonnet-4-6", base_url: str | None = None):
        from code_rag_eval.anthropic_backend import build_anthropic
        self._client = build_anthropic(base_url)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        from code_rag_eval.anthropic_backend import THINKING_DISABLED
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking=THINKING_DISABLED,
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def generate_answer(question: str, retrieved: list[RetrievedChunk], llm: LLMClient) -> Answer:
    user = build_user_prompt(question, retrieved)
    text = llm.complete(SYSTEM, user)
    return Answer(text=text, citations=extract_citations(text))
