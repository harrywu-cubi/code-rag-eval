from __future__ import annotations
from pathlib import Path


def iter_python_files(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.rglob("*.py")
        if "__pycache__" not in p.parts
    )
