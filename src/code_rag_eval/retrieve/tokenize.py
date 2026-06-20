from __future__ import annotations
import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[a-z]+|[A-Z]+|[0-9]+")


def tokenize_code(text: str) -> list[str]:
    """Tokenize code for BM25: keep each whole identifier (lowercased) plus its
    snake_case / camelCase subtokens, so both exact symbol names and their parts match."""
    out: list[str] = []
    for ident in _IDENT.findall(text):
        low = ident.lower()
        out.append(low)
        for part in ident.split("_"):
            for sub in _CAMEL.findall(part):
                s = sub.lower()
                if s and s != low:
                    out.append(s)
    return out
