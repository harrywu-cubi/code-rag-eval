from code_rag_eval.types import Chunk, RetrievedChunk
from code_rag_eval.generate.prompt import build_user_prompt, SYSTEM
from code_rag_eval.generate.answer import extract_citations, generate_answer, Citation
from tests.fakes import FakeLLMClient


def _retrieved():
    c = Chunk(text="def get_route_handler(): ...", file="fastapi/routing.py", start_line=120, end_line=180)
    return [RetrievedChunk(chunk=c, score=0.9, rank=1)]


def test_prompt_includes_file_line_header_and_question():
    prompt = build_user_prompt("Where is the route handler?", _retrieved())
    assert "# fastapi/routing.py:120-180" in prompt
    assert "Where is the route handler?" in prompt
    assert "def get_route_handler" in prompt


def test_extract_citations_dedupes():
    text = "See fastapi/routing.py:120 and again fastapi/routing.py:120 plus fastapi/params.py:5."
    cites = extract_citations(text)
    assert Citation("fastapi/routing.py", 120) in cites
    assert Citation("fastapi/params.py", 5) in cites
    assert len(cites) == 2


def test_generate_answer_calls_llm_and_parses_citations():
    llm = FakeLLMClient(answer="Defined at fastapi/routing.py:120.")
    ans = generate_answer("Where?", _retrieved(), llm)
    assert ans.text == "Defined at fastapi/routing.py:120."
    assert ans.citations == [Citation("fastapi/routing.py", 120)]
    assert llm.last_system == SYSTEM
    assert "fastapi/routing.py:120-180" in llm.last_user
