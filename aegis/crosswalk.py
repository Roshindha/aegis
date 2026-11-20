"""Crosswalk mode: map between IM8 controls and the external frameworks they reference.

Why this exists: the chunk text embedded for search does not contain the references, so a
question like "which controls reference MVSP 2.5?" cannot be answered by semantic/keyword
search over prose (baseline set-recall@20 ~0.2). The references live in structured metadata,
so we answer from an inverted index instead:

  forward  : (framework, ref_id) -> IM8 control IDs      "Which controls reference MVSP 2.5?"
  reverse  : IM8 control ID -> its external references   "Which NIST/MVSP refs does AS-12 map to?"

Detection reuses `parse_reference` (via the index build), and the matched controls are put
FIRST in the retrieved context, with their reference list appended so the LLM can cite it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import lru_cache

from aegis.corpus import load_chunks
from aegis.ingest import parse_reference
from aegis.retrieve import Hit

# Frameworks whose ref_id is only meaningful next to the framework name ("MVSP 2.5").
# Others (legacy IM8 circulars, "NIST FIPS 140-3", ...) carry their name inside ref_id.
ALIASES: dict[str, tuple[str, ...]] = {
    "MVSP": ("mvsp",),
    "NIST SP 800-53": ("nist sp 800-53", "nist 800-53", "nist sp800-53", "nist", "800-53"),
    "NIST CSF 2.0": ("nist csf 2.0", "nist csf", "csf"),
    "SLSA": ("slsa",),
    "CIS Controls v8": ("cis controls v8", "cis controls", "cis v8", "cis"),
    "CCM": ("ccm", "cloud controls matrix"),
}
FILLER = r"(?:\s+(?:sp|controls?|v8|section|requirement|item|ref(?:erence)?))*\s*[:#-]?\s*"
REVERSE_INTENT = re.compile(r"\b(maps?|mapped|mapping|references?|crosswalk|correspond\w*|align\w*|equivalent|cites?|external)\b", re.I)
ID_RE = re.compile(r"\b([A-Z]{2}-\d+)\b")
MAX_CONTEXT = 12


@dataclass
class Analysis:
    forward: list[tuple[str, str]] = field(default_factory=list)   # [(framework, ref_id)]
    reverse: list[str] = field(default_factory=list)               # IM8 control IDs
    controls: list[str] = field(default_factory=list)              # control IDs to put first

    @property
    def active(self) -> bool:
        return bool(self.forward or self.reverse)

    def summary(self) -> str:
        if self.forward:
            return "Index lookup " + ", ".join(f"{f} {r}" for f, r in self.forward) + " -> " + ", ".join(self.controls)
        if self.reverse:
            return "Reverse lookup " + ", ".join(self.reverse) + " -> external references"
        return ""


@lru_cache(maxsize=1)
def _data() -> tuple[dict, dict, dict]:
    """(index: (framework, ref_id) -> [control IDs], refs: control ID -> [raw ref strings], chunks)."""
    chunks = {c["control_id"]: c for c in load_chunks()}
    index: dict[tuple[str, str], set[str]] = {}
    for cid, c in chunks.items():
        for r in c["references"]:
            p = parse_reference(r["raw"])
            index.setdefault((p.framework, p.ref_id), set()).add(cid)
    index_sorted = {k: sorted(v) for k, v in index.items()}
    refs = {cid: [r["raw"] for r in c["references"]] for cid, c in chunks.items()}
    return index_sorted, refs, chunks


def reference_index() -> dict[tuple[str, str], list[str]]:
    return _data()[0]


def _ref_pattern(ref_id: str) -> str:
    # not glued to other word characters, not the start of a longer id ("AC-2" vs "AC-2(3)", "2.5" vs "2.55")
    return r"(?<![\w.\-])" + re.escape(ref_id) + r"(?![\w\-(]|\.\d)"


def _forward(question: str) -> list[tuple[str, str]]:
    found = []
    for (fw, ref_id), _ in reference_index().items():
        if fw in ALIASES:
            alias = "|".join(re.escape(a) for a in sorted(ALIASES[fw], key=len, reverse=True))
            pat = r"(?<![\w])(?:" + alias + r")" + FILLER + _ref_pattern(ref_id)
        else:                                   # name is part of the ref_id itself
            pat = _ref_pattern(ref_id)
        if re.search(pat, question, re.I):
            found.append((fw, ref_id))
    return found


def analyse(question: str) -> Analysis:
    index, _, chunks = _data()
    fwd = _forward(question)
    if fwd:
        controls = sorted({c for key in fwd for c in index[key]})
        return Analysis(forward=fwd, controls=controls)
    ids = [i for i in dict.fromkeys(ID_RE.findall(question)) if i in chunks]
    if ids and REVERSE_INTENT.search(question):
        return Analysis(reverse=ids, controls=ids)
    return Analysis()


def detect_references(question: str) -> bool:
    return analyse(question).active


def _crosswalk_hit(cid: str) -> Hit:
    _, refs, chunks = _data()
    c = chunks[cid]
    ref_line = "External references: " + ("; ".join(refs[cid]) if refs[cid] else "none recorded")
    return Hit(c["chunk_id"], cid, c["title"], c["group_title"], f"{c['text']}\n{ref_line}", 0.0, via="crosswalk")


def merge(a: Analysis, base: list[Hit], limit: int = 20) -> list[Hit]:
    """Crosswalk-matched controls first (with their references attached), then the usual hits."""
    if not a.active:
        return base[:limit]
    first = [_crosswalk_hit(cid) for cid in a.controls]
    seen = {h.control_id for h in first}
    return (first + [h for h in base if h.control_id not in seen])[:limit]


def crosswalk_search(question: str, base_fn) -> list[Hit]:
    """Retrieval method for the eval: base retriever + crosswalk override."""
    return merge(analyse(question), base_fn(question))


def context_size(hits: list[Hit], k: int = 5) -> int:
    """How many hits go to the LLM: k, or all crosswalk-matched controls if there are more."""
    n_cw = sum(h.via == "crosswalk" for h in hits)
    return min(max(k, n_cw), MAX_CONTEXT)
