import pytest

from aegis.generate import Answer
from aegis.retrieve import Hit
from eval import llm


class RateLimit(Exception):
    pass


def _hit(cid):
    return Hit(cid, cid, "t", "g", f"{cid}: text", 1.0)


def test_backoff_retries_429_with_exponential_waits():
    calls, waits = {"n": 0}, []

    def fn():
        calls["n"] += 1
        if calls["n"] < 4:
            raise RateLimit("Rate limit reached")
        return "ok"
    assert llm.call_with_backoff(fn, sleep=waits.append, is_rate_limit=lambda e: isinstance(e, RateLimit)) == "ok"
    assert waits == [1.0, 2.0, 4.0]


def test_backoff_honours_retry_after_and_does_not_retry_other_errors():
    waits, n = [], {"n": 0}

    def fn():
        n["n"] += 1
        if n["n"] == 1:
            raise RateLimit("tokens per minute. Please try again in 7.5s.")
        return "ok"
    llm.call_with_backoff(fn, sleep=waits.append, is_rate_limit=lambda e: isinstance(e, RateLimit))
    assert waits == [7.5]
    with pytest.raises(KeyError):
        llm.call_with_backoff(lambda: (_ for _ in ()).throw(KeyError("x")), is_rate_limit=lambda e: isinstance(e, RateLimit))


def test_daily_limit_raises_instead_of_sleeping():
    def fn():
        raise RateLimit("Rate limit reached ... on tokens per day (TPD). Please try again in 1h2m3.5s.")
    with pytest.raises(llm.DailyLimitReached):
        llm.call_with_backoff(fn, sleep=lambda s: pytest.fail("must not sleep"), is_rate_limit=lambda e: True)


def test_retry_after_parser():
    assert llm._retry_after(Exception("try again in 12m3.456s")) == pytest.approx(723.456)
    assert llm._retry_after(Exception("try again in 1h2m3s")) == 3723
    assert llm._retry_after(Exception("nothing")) is None


def test_cache_hit_avoids_second_call(tmp_path):
    path, cache, calls = tmp_path / "c.jsonl", {}, {"n": 0}

    def raw(q, hits):
        calls["n"] += 1
        return Answer("Use MFA [AC-2] and [ZZ-1].", ["AC-2", "ZZ-1"], ["ZZ-1"], hits, "m")
    ans = llm.make_cached_answerer(raw, cache, "m", path)
    hits = [_hit("AC-2")]
    a1 = ans("q", hits)
    a2 = ans("q", hits)
    assert calls["n"] == 1 and a1.text == a2.text
    assert a2.invalid_citations == ["ZZ-1"]            # recomputed against the context on a cache hit
    # a fresh process reloads the disk cache
    assert llm.load_cache(path)[llm.cache_key("m", "q", hits)] == a1.text
    assert llm.cache_key("m", "q", hits) != llm.cache_key("m", "q2", hits)
