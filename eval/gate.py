"""CI eval gate: fail if retrieval-only metrics drop below the committed floors.

    python -m eval.run_eval --retrieval-only --out eval/results/ci
    python -m eval.gate eval/results/ci.json            # floors from eval/gate.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FLOORS = Path("eval/gate.json")


def lookup(results: dict, path: str):
    node = results
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def check(results: dict, rules: list[dict]) -> list[str]:
    """Each rule: {"path": "retrieval.m.overall.mrr", "min": 0.7} and/or {"max": ...}."""
    failures = []
    for r in rules:
        try:
            v = lookup(results, r["path"])
        except KeyError:
            failures.append(f"{r['path']}: missing from results")
            continue
        if v is None:
            failures.append(f"{r['path']}: no value")
        elif "min" in r and v < r["min"]:
            failures.append(f"{r['path']}: {v:.4f} < floor {r['min']}")
        elif "max" in r and v > r["max"]:
            failures.append(f"{r['path']}: {v:.4f} > ceiling {r['max']}")
    return failures


def main() -> int:
    results = json.loads(Path(sys.argv[1]).read_text())
    rules = json.loads(FLOORS.read_text())["rules"]
    failures = check(results, rules)
    for r in rules:
        try:
            print(f"{r['path']:70} {lookup(results, r['path']):.4f}  (min {r.get('min')}, max {r.get('max')})")
        except KeyError:
            pass
    if failures:
        print("\nEVAL GATE FAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print(f"\nEval gate passed ({len(rules)} rules).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
