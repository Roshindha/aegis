"""Generate template questions, merge with curated ones, validate, write questions.jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aegis.corpus import load_chunks
from eval.generate_questions import generate
from eval.validate import load_questions, summary, validate

CURATED = Path("eval/curated.jsonl")
OUT = Path("eval/questions.jsonl")


def build(chunks: list[dict], curated: list[dict]) -> list[dict]:
    return generate(chunks) + curated


def main() -> int:
    chunks = load_chunks()
    questions = build(chunks, load_questions(CURATED))
    errors = validate(questions, chunks)
    if errors:
        print("\n".join(errors))
        return 1
    OUT.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    print(summary(questions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
