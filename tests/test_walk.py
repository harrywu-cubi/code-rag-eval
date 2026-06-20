from pathlib import Path
from code_rag_eval.ingest.walk import iter_python_files


def test_iter_python_files_skips_pycache(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("a = 1", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("b = 2", encoding="utf-8")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "a.cpython.pyc").write_text("x", encoding="utf-8")
    (tmp_path / "pkg" / "notes.md").write_text("hi", encoding="utf-8")
    files = iter_python_files(tmp_path)
    names = [f.name for f in files]
    assert names == ["a.py", "b.py"]
