from code_rag_eval.config import ExperimentConfig
from code_rag_eval.eval.experiment import matrix_configs


def test_matrix_is_eight_named_configs():
    cfgs = matrix_configs(ExperimentConfig(name="base"))
    names = [c.name for c in cfgs]
    assert len(cfgs) == 8
    assert "fixed_openai_vector" in names
    assert "ast_voyage_hybrid" in names
    byname = {c.name: c for c in cfgs}
    c = byname["ast_voyage_hybrid"]
    assert c.chunking.strategy == "ast"
    assert c.embedding.provider == "voyage" and c.embedding.model == "voyage-code-3"
    assert c.retrieval.method == "hybrid"
    d = byname["fixed_openai_vector"]
    assert d.chunking.strategy == "fixed"
    assert d.embedding.provider == "openai" and d.embedding.model == "text-embedding-3-large"
    assert d.retrieval.method == "vector"
