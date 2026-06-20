import importlib.util
from pathlib import Path


def test_streamlit_app_defines_main():
    p = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # runs module top-level; main() NOT called (guarded by __main__)
    assert callable(mod.main)
