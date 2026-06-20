from pathlib import Path
from code_rag_eval.config import ExperimentConfig, load_config


def test_load_baseline_config(tmp_path: Path):
    yaml_text = """
name: baseline
chunking:
  strategy: fixed
  window_lines: 40
  overlap_lines: 10
embedding:
  provider: openai
  model: text-embedding-3-large
retrieval:
  method: vector
  top_k: 5
generation:
  provider: anthropic
  model: claude-sonnet-4-6
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "baseline"
    assert cfg.chunking.strategy == "fixed"
    assert cfg.chunking.window_lines == 40
    assert cfg.embedding.model == "text-embedding-3-large"
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.rrf_k == 60  # default
