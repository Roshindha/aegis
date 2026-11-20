"""Pure metric functions for the evaluation (no network, no models)."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

REFUSAL_PHRASE = "do not answer this question"
LEVEL_WORDS = {0: "must-have", 1: "should-have", 2: "good-to-have"}
CITATION_RE = re.compile(r"\[[A-Z]{2}-\d+\]")
DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−­"), "-")


def norm(text: str) -> str:
    """NFKC + all dash variants (incl. non-breaking hyphen U+2011) -> "-" + lowercase + collapsed spaces.
    Models emit U+2011 inside words like 'agency‑defined'; plain `in` checks would miss them."""
    text = unicodedata.normalize("NFKC", text or "").translate(DASHES).lower()
    return re.sub(r"\s+", " ", text).strip()


# ---------- retrieval ----------

def first_gold_rank(ranked: list[str], gold: set[str]) -> int | None:
    for i, cid in enumerate(ranked, start=1):
        if cid in gold:
            return i
    return None


def retrieval_metrics(ranked: list[str], gold: list[str]) -> dict[str, float]:
    """ranked = control IDs best-first (top-20). Returns per-question numbers."""
    g = set(gold)
    r = first_gold_rank(ranked, g)
    out = {"mrr": 1.0 / r if r else 0.0}
    for k in (5, 20):
        top = set(ranked[:k])
        out[f"set_recall@{k}"] = len(g & top) / len(g)
        out[f"hit@{k}"] = 1.0 if g & top else 0.0
    return out


def mean_by(rows: list[dict], keys: list[str]) -> dict[str, float]:
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys} if rows else {}


# ---------- abstention ----------

def prf(pred: list[bool], truth: list[bool]) -> dict[str, float]:
    """Positive class = 'should abstain'. pred[i] = system abstained."""
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum((not p) and t for p, t in zip(pred, truth))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def is_refusal(answer: str) -> bool:
    return REFUSAL_PHRASE in norm(answer)


# ---------- answer correctness ----------

_LEVEL_NUM = re.compile(r"level\s*(\d)")
_NO_LEVEL = re.compile(
    r"\bno (profile |applicability |assigned |specific |defined )*level|not assigned|no level|"
    r"(does not|doesn't|do not|is not|isn't|not) (have|specify|define|assign|list|provide|give|state|include)\b.{0,40}level|"
    r"level (is |are )?(not|un)\s*(assigned|defined|specified|set)|none assigned|no applicability|not (been )?assigned|unassigned")
_AGENCY = re.compile(r"agency-defined|agency defined|defined by the agency|left to the agency|determined by the agency|set by the agency|agency-specific|agency decides")
_INVENTED_VALUE = re.compile(r"\b\d+\s*(?:-|\s)?\s*(day|hour|character|month|year|minute|week)s?\b")


def _levels_in(text: str) -> set[int]:
    found = {int(m) for m in _LEVEL_NUM.findall(text)}
    for lv, word in LEVEL_WORDS.items():
        if word in text:
            found.add(lv)
    return found


def _sentences_about(text: str, term: str) -> str:
    parts = re.split(r"(?<=[.;\n])\s+|;\s*|\bwhile\b|\bwhereas\b|\bbut\b|\band\b", text)
    return " ".join(p for p in parts if term in p.replace(" ", "-") or term.replace("-", " ") in p)


def check_applicability(answer: str, risk_class: str, level: int) -> bool:
    a = norm(CITATION_RE.sub("", answer))
    scope = _sentences_about(a, risk_class) or a
    return _levels_in(scope) == {level}


def check_no_level(answer: str, risk_class: str | None) -> bool:
    a = norm(CITATION_RE.sub("", answer))
    if is_refusal(a):
        return False
    if risk_class:
        scope = _sentences_about(a, risk_class)
        if scope and _levels_in(scope):
            return False                        # claims a level for the class that has none
    elif _levels_in(a):
        return False                            # claims some level for a level-less control
    return bool(_NO_LEVEL.search(a))


def check_agency_param(answer: str) -> bool:
    a = norm(CITATION_RE.sub("", answer))
    return bool(_AGENCY.search(a)) and not _INVENTED_VALUE.search(a) and not is_refusal(a)


def ids_mentioned(answer: str, gold: list[str]) -> float:
    a = norm(answer)
    return sum(norm(g) in a for g in gold) / len(gold) if gold else 0.0


def refs_mentioned(answer: str, refs: list[list[str]]) -> float:
    a = norm(answer)
    return sum(norm(rid) in a for _, rid in refs) / len(refs) if refs else 0.0


def score_answer(q: dict, answer: str, refused: bool, cited: list[str], invalid: list[str]) -> dict:
    """Per-question generation scores. `refused` = the SYSTEM abstained (guardrail or the LLM)."""
    cat, ans = q["category"], q["gold_answer"]
    out: dict = {"abstained": refused}
    if not q["answerable"]:
        out["correct"] = refused
        return out
    out["false_refusal"] = refused
    if refused:
        out["correct"] = False
        return out
    out["has_citation"] = bool(cited)
    out["citations_valid"] = not invalid
    out["cited_n"], out["cited_invalid_n"] = len(cited), len(invalid)
    gold = q["gold_control_ids"]
    out["gold_cited_recall"] = len(set(cited) & set(gold)) / len(gold)
    if cat == "applicability":
        out["correct"] = check_applicability(answer, ans["risk_class"], ans["level"])
    elif cat == "no_level":
        out["correct"] = check_no_level(answer, ans.get("risk_class"))
    elif cat == "agency_param":
        out["correct"] = check_agency_param(answer)
    elif cat == "crosswalk_reverse":
        out["correct"] = refs_mentioned(answer, ans["refs"]) >= 0.999
    elif cat in ("group_list", "crosswalk", "multi_control"):
        out["correct"] = ids_mentioned(answer, gold) >= 0.999
    else:                                       # exact_id, paraphrase: names or cites the gold control
        out["correct"] = ids_mentioned(answer, gold) >= 0.999
    return out


def aggregate(rows: list[dict], keys: list[str]) -> dict:
    """Mean of boolean/numeric keys over rows that have them; includes n and unverified count."""
    out: dict = {"n": len(rows), "n_unverified": sum(not r["verified"] for r in rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    return out


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        d[r[key]].append(r)
    return dict(d)
