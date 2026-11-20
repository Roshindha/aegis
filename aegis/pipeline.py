"""The full pipeline as explicit, inspectable stages:

    pre-filter -> semantic + BM25 ranks -> RRF fusion -> rerank -> guardrail -> generation

`run_stages` yields one event per stage (the API streams them; the explorer page draws them),
`ask` returns the final result. Retrieval methods can be run alone for ablations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator

from aegis import crosswalk, guardrail
from aegis.generate import Answer, answer_question
from aegis.rerank import rerank
from aegis.retrieve import Hit, bm25_search, hybrid_search, rrf_fuse, semantic_search

CANDIDATES = 20
METHODS = ("semantic", "bm25", "hybrid", "hybrid_rerank")


def retrieve(method: str, query: str, k: int = 5) -> list[Hit]:
    """One retrieval method, for ablations. Every method returns its own top-k."""
    if method == "semantic":
        return semantic_search(query, k)
    if method == "bm25":
        return bm25_search(query, k)
    if method == "hybrid":
        return hybrid_search(query, k)
    if method == "hybrid_rerank":
        return rerank(query, hybrid_search(query, k=CANDIDATES), k=k)
    raise ValueError(f"unknown method {method!r}")


def hit_dict(h: Hit, with_content: bool = False) -> dict:
    d = {"control_id": h.control_id, "title": h.title, "group": h.group_title,
         "score": round(h.score, 4), "semantic_rank": h.semantic_rank, "bm25_rank": h.bm25_rank,
         "via": h.via}
    if with_content:
        d["content"] = h.content
    return d


@dataclass
class Result:
    question: str
    refused: bool
    answer: str
    refusal_reason: str | None = None
    hits: list[Hit] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    top_score: float | None = None
    llm_called: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hits"] = [hit_dict(h) for h in self.hits]
        return d


def run_stages(question: str, *, k: int = 5, use_guardrail: bool = True, generate: bool = True,
               answerer: Callable[[str, list[Hit]], Answer] = answer_question) -> Iterator[dict]:
    """Yield {"stage": name, "data": {...}} events. The last is {"stage": "result", "data": ..., "obj": Result}."""
    pre = guardrail.prefilter(question)
    yield {"stage": "prefilter", "data": {"blocked": pre.blocked, "reason": pre.reason}}

    if use_guardrail and pre.blocked:
        # Refuse without retrieval or any LLM call.
        decision = guardrail.decide(question, None)
        yield {"stage": "guardrail", "data": {"refuse": True, "reason": decision.reason,
                                              "feature": decision.feature, "score": None, "threshold": decision.threshold}}
        res = Result(question, True, guardrail.REFUSAL_TEXT, decision.reason)
        yield {"stage": "result", "data": res.to_dict(), "obj": res}
        return

    cw = crosswalk.analyse(question)
    if cw.active:
        yield {"stage": "crosswalk", "data": {"summary": cw.summary(), "forward": cw.forward,
                                              "reverse": cw.reverse, "controls": cw.controls}}

    sem = semantic_search(question, CANDIDATES)
    bm = bm25_search(question, CANDIDATES)
    yield {"stage": "semantic", "data": {"hits": [hit_dict(h) for h in sem[:k]]}}
    yield {"stage": "bm25", "data": {"hits": [hit_dict(h) for h in bm[:k]]}}
    fused = rrf_fuse([sem, bm], k=CANDIDATES)
    yield {"stage": "fusion", "data": {"hits": [hit_dict(h) for h in fused[:k]]}}
    base = rerank(question, fused, k=k)
    top = base[0].score if base else None
    feature, _ = guardrail.load_calibration()
    signal = guardrail.feature_value(feature, base, bm)
    yield {"stage": "rerank", "data": {"hits": [hit_dict(h) for h in base]}}
    ranked = crosswalk.merge(cw, base, limit=CANDIDATES)
    ranked = ranked[:crosswalk.context_size(ranked, k)]

    bypass = guardrail.structural_bypass(question)
    decision = guardrail.decide(question, signal, use_prefilter=use_guardrail, use_threshold=use_guardrail,
                                bypass_threshold=bypass is not None)
    yield {"stage": "guardrail", "data": {"refuse": decision.refuse, "reason": decision.reason,
                                          "feature": feature, "score": signal, "top_rerank_score": top,
                                          "threshold": decision.threshold, "bypassed_threshold": decision.bypassed_threshold,
                                          "bypass_reason": bypass}}
    if decision.refuse:
        res = Result(question, True, guardrail.REFUSAL_TEXT, decision.reason, ranked, top_score=top)
        yield {"stage": "result", "data": res.to_dict(), "obj": res}
        return
    if not generate:
        res = Result(question, False, "", None, ranked, top_score=top)
        yield {"stage": "result", "data": res.to_dict(), "obj": res}
        return

    ans = answerer(question, ranked)
    yield {"stage": "answer", "data": {"text": ans.text, "cited_ids": ans.cited_ids,
                                       "invalid_citations": ans.invalid_citations}}
    res = Result(question, False, ans.text, None, ranked, ans.cited_ids, ans.invalid_citations, top, True)
    yield {"stage": "result", "data": res.to_dict(), "obj": res}


def ask(question: str, **kw) -> Result:
    """Run every stage and return the final Result (with real Hit objects)."""
    obj = None
    for ev in run_stages(question, **kw):
        obj = ev.get("obj", obj)
    return obj
