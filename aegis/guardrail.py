"""Guardrail: decide to refuse BEFORE the language model is ever called.

Two deterministic layers, neither of which depends on the LLM obeying a prompt:
  1. a regex pre-filter for prompt injection, other frameworks/laws, and unknown control IDs;
  2. a refusal threshold on the cross-encoder's top score (calibrated in eval/calibrate.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from aegis.corpus import REPO_ROOT, load_chunks

REFUSAL_TEXT = "The provided IM8 controls do not answer this question."
THRESHOLD_FILE = REPO_ROOT / "eval" / "threshold.json"
# Candidate relevance signals for the refusal threshold. eval/calibrate.py picks the one with the
# best held-out F1. Defaults apply only if no calibration file exists.
FEATURES = ("rerank_top1", "bm25_top1")
DEFAULT_FEATURE, DEFAULT_THRESHOLD = "rerank_top1", 0.0   # cross-encoder logit > 0 means "relevant"

# --- layer 1: obvious prompt injection / abuse (generic phrases, not tied to any one question)
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|any|the)\b.{0,30}\b(instructions?|prompts?|rules?|controls?|context)\b",
    r"\b(reveal|print|show|repeat|output|leak)\b.{0,30}\b(system|hidden|initial)\s+(prompt|message|instructions?)\b",
    r"\b(you are now|act as|pretend to be|from now on you)\b.{0,60}\b(no restrictions?|unrestricted|without (any )?(rules|limits|restrictions)|DAN)\b",
    r"\bjailbreak\b|\bDAN\b",
    r"\b(print|show|reveal|dump|list|leak)\b.{0,30}\b(environment variables?|api[ -]?keys?|secrets?|credentials?|passwords?)\b.{0,30}\b(you|your|have access)\b",
    r"\b(write|generate|create)\b.{0,30}\b(exploit|malware|ransomware|keylogger|port scanner|script that scans)\b",
]

# --- layer 1b: other frameworks, laws and topics the catalogue does not cover.
# NIST / MVSP / CIS / SLSA / CCM are deliberately NOT listed: the catalogue cross-references them.
OUT_OF_SCOPE_PATTERNS = [
    r"\bEU\s+AI\s+Act\b", r"\bGDPR\b", r"\bPCI[ -]?DSS\b", r"\bHIPAA\b", r"\bSOX\b|\bSarbanes",
    r"\bISO[ /-]?(IEC )?2700\d\b", r"\bSOC ?2\b", r"\bEssential Eight\b", r"\bCyber Essentials\b",
    r"\bAustralian\b.{0,30}\b(ISM|Information Security Manual)\b|\bAustralian ISM\b|\bInformation Security Manual\b",
    r"\bPDPA\b", r"\bPersonal Data Protection Act\b", r"\bMAS\b.{0,20}\b(TRM|Technology Risk)\b",
    r"\bCode of Practice\b.{0,40}\b(CII|Critical Information Infrastructure)\b", r"\bCybersecurity Act\b",
    r"\bMITRE ATT&CK\b", r"\bNIST SP 800-171\b", r"\bCIS Benchmarks?\b",
    r"\b(weather|recipe|stock price|football|movie)\b",
]
_INJECTION = [re.compile(p, re.I) for p in INJECTION_PATTERNS]
_SCOPE = [re.compile(p, re.I) for p in OUT_OF_SCOPE_PATTERNS]
_ID_RE = re.compile(r"\b([A-Z]{2})-(\d+)\b")
_OTHER_FW = re.compile(r"NIST|MVSP|CIS|SLSA|CCM|CSF", re.I)


@lru_cache(maxsize=1)
def known_ids() -> frozenset[str]:
    return frozenset(c["control_id"] for c in load_chunks())


def mentioned_ids(question: str) -> tuple[list[str], list[str]]:
    """IM8-style IDs in the question, split into (known, unknown). Skips questions about
    other frameworks, whose IDs (e.g. NIST AC-2) look the same."""
    if _OTHER_FW.search(question):
        return [], []
    known = known_ids()
    prefixes = {i.split("-")[0] for i in known}
    found = [f"{a}-{b}" for a, b in _ID_RE.findall(question) if a in prefixes]
    return [i for i in found if i in known], [i for i in found if i not in known]


@lru_cache(maxsize=1)
def _group_titles() -> tuple[str, ...]:
    return tuple(sorted({c["group_title"].lower() for c in load_chunks()}, key=len, reverse=True))


def structural_bypass(question: str) -> str | None:
    """Questions that are lookups, not relevance searches, so a low relevance score means nothing.
    Returns why ("control_id" | "group" | "crosswalk") or None."""
    known, _ = mentioned_ids(question)
    if known:
        return "control_id"
    q = question.lower()
    if any(t in q for t in _group_titles()):
        return "group"
    try:                                    # crosswalk detection lives in aegis.crosswalk (Step 11)
        from aegis.crosswalk import detect_references
        if detect_references(question):
            return "crosswalk"
    except ImportError:
        pass
    return None


@dataclass
class PrefilterResult:
    blocked: bool
    reason: str | None = None       # "prompt_injection" | "out_of_scope" | "unknown_control_id"
    pattern: str | None = None


def prefilter(question: str) -> PrefilterResult:
    for rx in _INJECTION:
        if rx.search(question):
            return PrefilterResult(True, "prompt_injection", rx.pattern)
    for rx in _SCOPE:
        if rx.search(question):
            return PrefilterResult(True, "out_of_scope", rx.pattern)
    _, unknown = mentioned_ids(question)
    if unknown:
        return PrefilterResult(True, "unknown_control_id", ", ".join(unknown))
    return PrefilterResult(False)


@lru_cache(maxsize=1)
def load_calibration(path: Path = THRESHOLD_FILE) -> tuple[str, float]:
    """(feature, threshold) from the committed calibration file."""
    if path.exists():
        d = json.loads(path.read_text())
        return d.get("feature", DEFAULT_FEATURE), float(d["threshold"])
    return DEFAULT_FEATURE, DEFAULT_THRESHOLD


def load_threshold() -> float:
    return load_calibration()[1]


def feature_value(feature: str, ranked: list, bm25_hits: list) -> float | None:
    """The calibrated relevance signal: top rerank logit, or top BM25 score (0 if no term matched)."""
    if feature == "rerank_top1":
        return ranked[0].score if ranked else None
    if feature == "bm25_top1":
        return bm25_hits[0].score if bm25_hits else 0.0
    raise ValueError(f"unknown feature {feature!r}")


@dataclass
class Decision:
    refuse: bool
    reason: str | None            # why: prefilter reason, "low_retrieval_score", or None
    top_score: float | None       # value of the calibrated feature (name in `feature`)
    threshold: float
    bypassed_threshold: bool = False
    feature: str = ""


def decide(question: str, top_score: float | None, *, threshold: float | None = None,
           use_prefilter: bool = True, use_threshold: bool = True,
           bypass_threshold: bool = False) -> Decision:
    """The refusal decision. Pure function of (question text, retrieval score).

    bypass_threshold: the question names a real control ID (or a recognised external
    reference), so retrieval is by identifier and the semantic-relevance score says nothing.
    """
    feature, cal_t = load_calibration()
    t = cal_t if threshold is None else threshold
    if use_prefilter:
        pre = prefilter(question)
        if pre.blocked:
            return Decision(True, pre.reason, top_score, t, feature=feature)
    if not use_threshold:
        return Decision(False, None, top_score, t, feature=feature)
    if bypass_threshold:
        return Decision(False, None, top_score, t, bypassed_threshold=True, feature=feature)
    if top_score is None or top_score < t:
        return Decision(True, "low_retrieval_score", top_score, t, feature=feature)
    return Decision(False, None, top_score, t, feature=feature)
