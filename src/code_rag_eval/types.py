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
