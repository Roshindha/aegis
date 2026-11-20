from eval.gate import check, lookup

RES = {"retrieval": {"m": {"overall": {"mrr": 0.8}}}, "guardrail": {"g": {"false_refusal_rate": 0.05}}}


def test_lookup():
    assert lookup(RES, "retrieval.m.overall.mrr") == 0.8


def test_passes_within_bounds():
    rules = [{"path": "retrieval.m.overall.mrr", "min": 0.75}, {"path": "guardrail.g.false_refusal_rate", "max": 0.1}]
    assert check(RES, rules) == []


def test_fails_when_metric_drops_or_missing():
    assert check(RES, [{"path": "retrieval.m.overall.mrr", "min": 0.9}])
    assert check(RES, [{"path": "guardrail.g.false_refusal_rate", "max": 0.01}])
    assert check(RES, [{"path": "retrieval.nope.overall.mrr", "min": 0.1}])
