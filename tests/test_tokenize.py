from code_rag_eval.retrieve.tokenize import tokenize_code


def test_tokenize_keeps_whole_identifier_and_subtokens():
    toks = tokenize_code("def get_route_handler(): camelCaseName = 1")
    assert "get_route_handler" in toks                 # whole snake identifier
    assert {"get", "route", "handler"} <= set(toks)    # snake subtokens
    assert "camelcasename" in toks                     # whole identifier, lowercased
    assert {"camel", "case", "name"} <= set(toks)      # camelCase subtokens
    assert "def" in toks


def test_tokenize_empty_is_empty():
    assert tokenize_code("") == []
