from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel

Category = Literal["locate", "explain", "trace", "behavior"]


class EvalRecord(BaseModel):
    id: str
    category: Category
    question: str
    gold_symbols: list[str]
    gold_files: list[str]
    gold_line_ranges: list[tuple[int, int]]
    reference_answer: str


def load_eval_set(path: str | Path) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(EvalRecord(**json.loads(line)))
    return records


def save_eval_set(records: list[EvalRecord], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
