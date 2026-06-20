from pathlib import Path
from code_rag_eval.paths import relpath, module_prefix


def test_relpath_is_posix(tmp_path: Path):
    root = tmp_path / "fastapi"
    f = root / "fastapi" / "routing.py"
    f.parent.mkdir(parents=True)
    f.write_text("x = 1", encoding="utf-8")
    assert relpath(f, root) == "fastapi/routing.py"


def test_module_prefix():
    assert module_prefix("fastapi/routing.py") == "fastapi.routing"
    assert module_prefix("fastapi/__init__.py") == "fastapi"
    assert module_prefix("fastapi/security/oauth2.py") == "fastapi.security.oauth2"
