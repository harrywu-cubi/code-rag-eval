from __future__ import annotations
from pathlib import Path


def relpath(path: Path, corpus_root: Path) -> str:
    return path.resolve().relative_to(corpus_root.resolve()).as_posix()


def module_prefix(rel_path: str) -> str:
    parts = rel_path[:-3].split("/")  # strip ".py"
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)
