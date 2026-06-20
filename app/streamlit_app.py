from __future__ import annotations
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from code_rag_eval.config import load_config
from code_rag_eval.factories import make_embedding_client, make_llm_client
from code_rag_eval.ingest.store import ChromaStore
from code_rag_eval.retrieve.factory import make_retriever
from code_rag_eval.generate.answer import generate_answer

CHROMA_DIR = ".chroma"
CONFIG = "configs/baseline.yaml"


def main() -> None:
    load_dotenv()
    st.title("code-rag-eval — FastAPI code Q&A")
    st.caption("Retrieval-augmented answers over the FastAPI source, with file:line citations.")
    cfg = load_config(CONFIG)
    question = st.text_input("Ask about the FastAPI codebase:")
    if not question:
        return
    embed = make_embedding_client(cfg.embedding)
    llm = make_llm_client(cfg.generation)
    store = ChromaStore(collection_name=cfg.name, persist_dir=CHROMA_DIR)
    retriever = make_retriever(cfg, store, embed)
    retrieved = retriever.retrieve(question, cfg.retrieval.top_k)
    answer = generate_answer(question, retrieved, llm)
    st.markdown(answer.text)
    st.subheader("Sources")
    for r in retrieved:
        st.write(f"`{r.chunk.file}:{r.chunk.start_line}-{r.chunk.end_line}` (score {r.score:.3f})")


if __name__ == "__main__":
    main()
