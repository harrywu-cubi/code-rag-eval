from code_rag_eval.ingest.chunkers import chunk_fixed
from code_rag_eval.ingest.chunkers import chunk_ast


def test_fixed_windows_with_overlap():
    text = "\n".join(f"line{i}" for i in range(1, 11))  # 10 lines
    chunks = chunk_fixed(text, "a.py", window_lines=4, overlap_lines=1)
    # step = 3 -> windows starting at lines 1, 4, 7
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 4), (4, 7), (7, 10)]
    assert chunks[0].text.splitlines()[0] == "line1"
    assert all(c.kind == "fixed" and c.file == "a.py" for c in chunks)


def test_fixed_handles_short_and_empty():
    assert chunk_fixed("", "a.py", 40, 10) == []
    short = chunk_fixed("only one line", "a.py", 40, 10)
    assert len(short) == 1 and short[0].start_line == 1 and short[0].end_line == 1


_AST_SRC = '''\
import os


def top():
    """doc"""
    return 1


@decorator
class Service:
    """svc"""

    def handle(self, x):
        return x

    def other(self):
        return 2
'''


def test_chunk_ast_splits_by_definition():
    chunks = chunk_ast(_AST_SRC, "m.py")
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["top"].kind == "function"
    assert by_symbol["top"].start_line == 4
    assert "return 1" in by_symbol["top"].text
    assert by_symbol["Service"].kind == "class"
    assert "class Service" in by_symbol["Service"].text
    assert "return x" not in by_symbol["Service"].text          # method bodies are separate
    assert by_symbol["Service.handle"].kind == "method"
    assert "return x" in by_symbol["Service.handle"].text       # never split a function
    assert "Service.other" in by_symbol


def test_chunk_ast_windows_long_units():
    long_src = "def big():\n" + "\n".join(f"    x{i} = {i}" for i in range(300))
    chunks = chunk_ast(long_src, "b.py", max_lines=100, overlap_lines=10)
    assert len(chunks) > 1
    assert all(c.symbol == "big" and c.kind == "function" for c in chunks)
