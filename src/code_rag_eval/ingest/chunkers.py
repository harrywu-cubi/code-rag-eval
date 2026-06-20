from __future__ import annotations
from code_rag_eval.types import Chunk
from tree_sitter import Language, Parser
import tree_sitter_python

_PARSER = Parser(Language(tree_sitter_python.language()))


def chunk_fixed(text: str, file: str, window_lines: int, overlap_lines: int) -> list[Chunk]:
    """Deliberately naive: fixed line windows that ignore code structure.

    This is the baseline the AST chunker (Phase 4) is measured against.
    """
    lines = text.splitlines()
    if not lines:
        return []
    step = max(1, window_lines - overlap_lines)
    chunks: list[Chunk] = []
    i = 0
    n = len(lines)
    while i < n:
        window = lines[i:i + window_lines]
        chunks.append(Chunk(
            text="\n".join(window),
            file=file,
            start_line=i + 1,
            end_line=i + len(window),
            kind="fixed",
        ))
        if i + window_lines >= n:
            break
        i += step
    return chunks


def _name_of(def_node) -> str:
    name = def_node.child_by_field_name("name")
    return name.text.decode() if name is not None else "?"


def _inner_def(node):
    """Unwrap a decorated_definition to its function_definition/class_definition."""
    if node.type == "decorated_definition":
        for c in node.children:
            if c.type in ("function_definition", "class_definition"):
                return c
    return node


def _split_unit(unit_lines, start_line, file, symbol, kind, signature, max_lines, overlap_lines):
    n = len(unit_lines)
    if n <= max_lines:
        return [Chunk(text="\n".join(unit_lines), file=file, start_line=start_line,
                      end_line=start_line + n - 1, kind=kind, symbol=symbol, signature=signature)]
    out = []
    step = max(1, max_lines - overlap_lines)
    i = 0
    while i < n:
        window = unit_lines[i:i + max_lines]
        out.append(Chunk(text="\n".join(window), file=file, start_line=start_line + i,
                         end_line=start_line + i + len(window) - 1, kind=kind,
                         symbol=symbol, signature=signature))
        if i + max_lines >= n:
            break
        i += step
    return out


def chunk_ast(text: str, file: str, max_lines: int = 120, overlap_lines: int = 20) -> list[Chunk]:
    """AST-aware chunking via tree-sitter: one chunk per top-level function, per class
    header, and per method. Never splits a definition mid-body unless it exceeds
    max_lines (then it is windowed). Module-level statements between definitions are not
    separately indexed — eval gold symbols are always definitions, so retrieval metrics
    are unaffected; this is a deliberate scope choice for the AST strategy.
    """
    tree = _PARSER.parse(text.encode("utf-8"))
    root = tree.root_node
    lines = text.splitlines()
    units: list[tuple[int, int, str, str]] = []  # (start_line, end_line, kind, symbol) 1-based

    for node in root.children:
        inner = _inner_def(node)
        if inner.type == "function_definition":
            units.append((node.start_point[0] + 1, node.end_point[0] + 1, "function", _name_of(inner)))
        elif inner.type == "class_definition":
            cname = _name_of(inner)
            body = inner.child_by_field_name("body")
            methods = []
            if body is not None:
                for ch in body.children:
                    if _inner_def(ch).type == "function_definition":
                        methods.append(ch)
            if methods:
                header_start = node.start_point[0] + 1
                header_end = max(header_start, methods[0].start_point[0])  # line before first method
                units.append((header_start, header_end, "class", cname))
                for ch in methods:
                    m_inner = _inner_def(ch)
                    units.append((ch.start_point[0] + 1, ch.end_point[0] + 1, "method",
                                  f"{cname}.{_name_of(m_inner)}"))
            else:
                units.append((node.start_point[0] + 1, node.end_point[0] + 1, "class", cname))

    units.sort(key=lambda u: u[0])
    chunks: list[Chunk] = []
    for (s, e, kind, symbol) in units:
        unit_lines = lines[s - 1:e]
        if not unit_lines:
            continue
        signature = unit_lines[0].strip()
        chunks.extend(_split_unit(unit_lines, s, file, symbol, kind, signature, max_lines, overlap_lines))
    return chunks
