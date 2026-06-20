from code_rag_eval.eval.symbols import enumerate_symbols

SRC = '''\
def top():
    """top docstring"""
    return 1


class Service:
    """svc"""
    def handle(self, x):
        return x
'''


def test_enumerate_symbols_qualified_names_and_ranges():
    syms = enumerate_symbols(SRC, "fastapi/routing.py", "fastapi.routing")
    by_name = {s.qualified_name: s for s in syms}
    assert "fastapi.routing.top" in by_name
    assert "fastapi.routing.Service" in by_name
    assert "fastapi.routing.Service.handle" in by_name
    top = by_name["fastapi.routing.top"]
    assert top.kind == "function"
    assert top.start_line == 1
    assert top.docstring == "top docstring"
    assert by_name["fastapi.routing.Service.handle"].signature == "def handle(self, x)"
