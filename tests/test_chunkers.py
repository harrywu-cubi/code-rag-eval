from code_rag_eval.ingest.chunkers import chunk_fixed


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
