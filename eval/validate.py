"""Validate the benchmark. Recomputes every derivable gold label from the chunks and
exits non-zero on any error (so it can gate CI).

    python -m eval.validate [path/to/questions.jsonl]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from aegis.corpus import load_chunks
from aegis.ingest import parse_reference

CATEGORIES = {"exact_id", "applicability", "no_level", "group_list", "crosswalk",
              "crosswalk_reverse", "paraphrase", "agency_param", "multi_control", "out_of_scope"}
SOURCES = {"generated", "curated"}
REQUIRED = ("qid", "source", "category", "question", "gold_control_ids", "gold_answer",
            "answerable", "verified")
ID_RE = re.compile(r"\b([A-Z]{2})-(\d+)\b")
OTHER_FRAMEWORK_RE = re.compile(r"NIST|MVSP|CIS|SLSA|CCM", re.I)


def load_questions(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(questions: list[dict], chunks: list[dict]) -> list[str]:
    errors: list[str] = []
    by_id = {c["control_id"]: c for c in chunks}
    prefixes = {cid.split("-")[0] for cid in by_id}
    index: dict[tuple[str, str], set[str]] = {}
    for c in chunks:
        for r in c["references"]:
            p = parse_reference(r["raw"])
            index.setdefault((p.framework, p.ref_id), set()).add(c["control_id"])
    seen: set[str] = set()

    for q in questions:
        qid = q.get("qid", "<missing qid>")
        def err(msg: str) -> None:
            errors.append(f"{qid}: {msg}")

        missing = [f for f in REQUIRED if f not in q]
        if missing:
            err(f"missing fields {missing}")
            continue
        if qid in seen:
            err("duplicate qid")
        seen.add(qid)
        if q["category"] not in CATEGORIES:
            err(f"unknown category {q['category']!r}")
        if q["source"] not in SOURCES:
            err(f"unknown source {q['source']!r}")
        if not isinstance(q["verified"], bool):
            err("verified must be a boolean")
        gold, ans, cat = q["gold_control_ids"], q["gold_answer"], q["category"]

        for cid in gold:
            if cid not in by_id:
                err(f"unknown control ID in gold: {cid}")
        if not q["answerable"] and gold:
            err("unanswerable question must have empty gold_control_ids")
        if q["answerable"] and not gold:
            err("answerable question needs gold_control_ids")
        if (cat == "out_of_scope") == q["answerable"]:
            err("out_of_scope must be (and only be) unanswerable")

        # IM8-style IDs mentioned in the question text must exist, unless it is a deliberate probe
        if not q.get("unknown_id_probe") and not OTHER_FRAMEWORK_RE.search(q["question"]):
            for pre, num in ID_RE.findall(q["question"]):
                if pre in prefixes and f"{pre}-{num}" not in by_id:
                    err(f"question mentions unknown control ID {pre}-{num}")
        if q.get("unknown_id_probe") and not any(
                f"{a}-{b}" not in by_id for a, b in ID_RE.findall(q["question"])):
            err("unknown_id_probe set but the question contains no unknown ID")

        known = [g for g in gold if g in by_id]
        if len(known) != len(gold):
            continue                                    # label checks need valid IDs

        if cat == "exact_id":
            if len(gold) != 1 or gold[0] not in q["question"]:
                err("exact_id gold must be the single ID named in the question")
        elif cat in ("applicability", "no_level"):
            if len(gold) != 1 or gold[0] not in q["question"]:
                err(f"{cat} gold must be the single ID named in the question")
            elif "level" not in ans:
                err("gold_answer.level missing")
            else:
                levels = by_id[gold[0]]["levels"]
                rc = ans.get("risk_class")
                actual = levels.get(rc) if rc else None
                if rc is None and ans["level"] is not None:
                    err("level given without risk_class")
                if (rc is not None or ans["level"] is not None) and ans["level"] != actual:
                    err(f"level {ans['level']} for {rc} does not match chunk levels {levels}")
                if cat == "no_level" and ans["level"] is not None:
                    err("no_level question must have level null")
                if cat == "applicability" and ans["level"] is None:
                    err("applicability question must have a level")
                if cat == "no_level" and rc is None and levels:
                    err("no_level question about a control that has levels, without risk_class")
        elif cat == "group_list":
            members = sorted(c["control_id"] for c in chunks if c["group_id"] == ans.get("group_id"))
            if not members or sorted(gold) != members:
                err(f"group_list gold != members of group {ans.get('group_id')}")
        elif cat == "crosswalk":
            key = (ans.get("framework"), ans.get("ref_id"))
            if sorted(gold) != sorted(index.get(key, set())):
                err(f"crosswalk gold does not match reference index for {key}")
        elif cat == "crosswalk_reverse":
            have = {(p.framework, p.ref_id) for r in by_id[gold[0]]["references"]
                    if (p := parse_reference(r["raw"]))}
            for fw, rid in ans.get("refs", []):
                if (fw, rid) not in have:
                    err(f"reverse ref {(fw, rid)} not among {gold[0]} references")
            if not ans.get("refs"):
                err("crosswalk_reverse needs gold_answer.refs")
        elif cat == "agency_param":
            labels = ans.get("param_labels", [])
            if not ans.get("agency_defined") or not labels:
                err("agency_param needs agency_defined=true and param_labels")
            for label in labels:
                if not any(f"[agency-defined: {label}]" in by_id[g]["text"] for g in gold):
                    err(f"param label {label!r} not found in gold control text")
        elif cat == "multi_control":
            if len(gold) < 2:
                err("multi_control needs at least 2 gold controls")
    return errors


def summary(questions: list[dict]) -> str:
    lines = [f"Total questions: {len(questions)}"]
    for title, key in (("By category", "category"), ("By source", "source")):
        lines.append(f"{title}: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(q[key] for q in questions).items())))
    ans = sum(q["answerable"] for q in questions)
    lines.append(f"Answerable: {ans}  Unanswerable: {len(questions) - ans}")
    unv = sum(not q["verified"] for q in questions)
    lines.append(f"Verified: {len(questions) - unv}  Unverified: {unv}")
    return "\n".join(lines)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/questions.jsonl")
    questions = load_questions(path)
    errors = validate(questions, load_chunks())
    print(summary(questions))
    if errors:
        print(f"\n{len(errors)} validation error(s):")
        for e in errors:
            print(" -", e)
        return 1
    print("\nOK: benchmark is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
