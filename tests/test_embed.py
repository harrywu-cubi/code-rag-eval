from code_rag_eval.ingest.embed import EmbeddingClient
from tests.fakes import FakeEmbeddingClient


def test_fake_embed_is_deterministic_and_right_shape():
    client: EmbeddingClient = FakeEmbeddingClient(dim=16)
    a1 = client.embed(["hello"])[0]
    a2 = client.embed(["hello"])[0]
    b = client.embed(["world"])[0]
    assert len(a1) == 16
    assert a1 == a2          # deterministic
    assert a1 != b           # content-sensitive
