from code_rag_eval.eval.judge import _first_float


def test_first_float_parses_valid_scores():
    assert _first_float("0.8") == 0.8
    assert _first_float("1") == 1.0
    assert _first_float("0") == 0.0
    assert _first_float("0.95") == 0.95
    assert _first_float("1.0") == 1.0
    assert _first_float("The score is 0.7.") == 0.7


def test_first_float_rejects_out_of_range_and_garbage():
    assert _first_float("7 out of 10") == 0.0   # no standalone 0..1 token
    assert _first_float("no number here") == 0.0
    assert _first_float("") == 0.0
