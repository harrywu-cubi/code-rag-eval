from __future__ import annotations
from typing import Protocol
import chromadb
from code_rag_eval.types import Chunk


def chunk_to_metadata(c: Chunk) -> dict:
    return {
        "file": c.file,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "kind": c.kind,
        "symbol": c.symbol or "",
        "signature": c.signature or "",
        "docstring": c.docstring or "",
    }


def metadata_to_chunk(text: str, m: dict) -> Chunk:
    return Chunk(
        text=text,
        file=str(m["file"]),
        start_line=int(m["start_line"]),
        end_line=int(m["end_line"]),
        kind=str(m.get("kind", "fixed")),
        symbol=(m.get("symbol") or None),
        signature=(m.get("signature") or None),
        docstring=(m.get("docstring") or None),
    )


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def query(self, vector: list[float], n: int) -> list[tuple[Chunk, float]]: ...


class ChromaStore:
    def __init__(self, collection_name: str, persist_dir: str | None = None):
        self._client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir else chromadb.EphemeralClient()
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[chunk_to_metadata(c) for c in chunks],
        )

    def query(self, vector: list[float], n: int) -> list[tuple[Chunk, float]]:
        res = self._col.query(query_embeddings=[vector], n_results=n)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            (metadata_to_chunk(doc, meta), 1.0 - float(dist))
            for doc, meta, dist in zip(docs, metas, dists)
        ]
