"""Template-generated benchmark questions. Deterministic, seeded, no LLM.

Every gold label is copied from the chunk data itself, so a label can never disagree
with the catalogue. eval/validate.py recomputes the labels independently as a check.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from aegis.corpus import load_chunks
from aegis.ingest import parse_reference

SEED = 7
RISK_CLASSES = ("low-risk", "medium-risk")
LEVEL_NAMES = {0: "must-have", 1: "should-have", 2: "good-to-have"}
PREFERRED_FRAMEWORKS = ("MVSP", "NIST SP 800-53", "SLSA", "CIS Controls v8", "CCM")
N_GROUPS = 10
OUT = Path("eval/generated.jsonl")


def reference_index(chunks: list[dict]) -> dict[tuple[str, str], set[str]]:
    """(framework, ref_id) -> control IDs, built by re-parsing each raw reference string."""
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in chunks:
        for r in c["references"]:
            ref = parse_reference(r["raw"])
            index[(ref.framework, ref.ref_id)].add(c["control_id"])
    return index


def _q(cat: str, n: int, question: str, ids: list[str], answer: dict, **extra) -> dict:
    return {"qid": f"gen-{cat}-{n:03d}", "source": "generated", "category": cat,
            "question": question, "gold_control_ids": ids, "gold_answer": answer,
            "answerable": True, "verified": False, **extra}


def _spread_by_level(pairs: list[tuple[str, str, int]], n: int, rng: random.Random):
    """Pick n (control, class, level) triples, round-robin over levels so 0/1/2 all appear."""
    buckets: dict[int, list] = defaultdict(list)
    for p in sorted(pairs):
        buckets[p[2]].append(p)
    for b in buckets.values():
        rng.shuffle(b)
    out, levels = [], sorted(buckets)
    while len(out) < n and any(buckets.values()):
        for lv in levels:
            if buckets[lv] and len(out) < n:
                out.append(buckets[lv].pop())
    return out


def generate(chunks: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    by_id = {c["control_id"]: c for c in chunks}
    ids = sorted(by_id)
    qs: list[dict] = []

    # 1. exact_id: gold = the control named in the question
    for n, cid in enumerate(sorted(rng.sample(ids, 10)), 1):
        qs.append(_q("exact_id", n, f"What does {cid} require?", [cid], {}))

    # 2. applicability: ask about a class the control HAS a level for
    both = [(c["control_id"], rc, c["levels"][rc]) for c in chunks
            if len(c["levels"]) == 2 for rc in RISK_CLASSES]
    one_class = sorted(c["control_id"] for c in chunks if len(c["levels"]) == 1)
    n = 0
    for cid, rc, lv in _spread_by_level(both, 7, rng):
        n += 1
        qs.append(_q("applicability", n,
                     f"What is the applicability level of {cid} for {rc} systems?", [cid],
                     {"risk_class": rc, "level": lv, "level_name": LEVEL_NAMES[lv]}))
    for cid in one_class:
        (rc, lv), = by_id[cid]["levels"].items()
        n += 1
        qs.append(_q("applicability", n,
                     f"What is the applicability level of {cid} for {rc} systems?", [cid],
                     {"risk_class": rc, "level": lv, "level_name": LEVEL_NAMES[lv]}))

    # 3. no_level: (a) every control with no level, (b) the missing class of one-class controls
    n = 0
    for cid in ids:
        if not by_id[cid]["levels"]:
            n += 1
            qs.append(_q("no_level", n, f"What applicability level does {cid} have?", [cid],
                         {"level": None}))
    for cid in one_class:
        (have,) = by_id[cid]["levels"]
        missing = next(rc for rc in RISK_CLASSES if rc != have)
        n += 1
        qs.append(_q("no_level", n,
                     f"What is the applicability level of {cid} for {missing} systems?", [cid],
                     {"risk_class": missing, "level": None}))

    # 4. group_list: the N_GROUPS smallest groups, gold = every control in the group
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        groups[c["group_id"]].append(c)
    smallest = sorted(groups, key=lambda g: (len(groups[g]), g))[:N_GROUPS]
    for n, gid in enumerate(smallest, 1):
        members = sorted(c["control_id"] for c in groups[gid])
        title = groups[gid][0]["group_title"]
        qs.append(_q("group_list", n, f"List all the controls in the {title} group.", members,
                     {"group_id": gid}))

    # 5. crosswalk: external reference -> IM8 controls; 10 shared keys + 5 single-control keys
    index = reference_index(chunks)
    pref = [(k, v) for k, v in sorted(index.items()) if k[0] in PREFERRED_FRAMEWORKS]
    shared = [kv for kv in pref if len(kv[1]) > 1]
    single = [kv for kv in pref if len(kv[1]) == 1]
    picked = rng.sample(shared, min(10, len(shared)))
    by_fw: dict[str, list] = defaultdict(list)
    for kv in single:
        by_fw[kv[0][0]].append(kv)
    for lst in by_fw.values():
        rng.shuffle(lst)
    singles = []
    while len(singles) < 5 and any(by_fw.values()):          # round-robin over frameworks
        for fw in sorted(by_fw):
            if by_fw[fw] and len(singles) < 5:
                singles.append(by_fw[fw].pop())
    for n, ((fw, rid), cids) in enumerate(sorted(picked + singles), 1):
        qs.append(_q("crosswalk", n, f"Which IM8 Reform controls reference {fw} {rid}?",
                     sorted(cids), {"framework": fw, "ref_id": rid}))

    # 6. crosswalk_reverse: IM8 control -> its external references (preferred frameworks)
    rev_ids = [cid for cid in ids if any(
        parse_reference(r["raw"]).framework in PREFERRED_FRAMEWORKS for r in by_id[cid]["references"])]
    for n, cid in enumerate(sorted(rng.sample(rev_ids, 5)), 1):
        refs = sorted({(p.framework, p.ref_id) for r in by_id[cid]["references"]
                       if (p := parse_reference(r["raw"])).framework in PREFERRED_FRAMEWORKS})
        qs.append(_q("crosswalk_reverse", n,
                     f"Which NIST, MVSP, CIS, SLSA or CCM references does {cid} map to?", [cid],
                     {"refs": [list(r) for r in refs]}))
    return qs


def main() -> None:
    qs = generate(load_chunks())
    OUT.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8")
    print(f"Wrote {len(qs)} generated questions to {OUT}")


if __name__ == "__main__":
    main()
