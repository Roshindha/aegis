"""Step 2: build control-aware chunks and save them to data/processed/chunks.jsonl."""
from collections import Counter
from pathlib import Path

from aegis.ingest import build_chunks, save_jsonl

OUT = Path("data/processed/chunks.jsonl")


def main() -> None:
    chunks = build_chunks()
    save_jsonl(chunks, OUT)

    words = sorted(len(c.text.split()) for c in chunks)
    frameworks = Counter(r.framework for c in chunks for r in c.references)
    unlevelled = [c.control_id for c in chunks if not c.levels]
    leftover = [c.control_id for c in chunks if "{{" in c.text]

    print(f"Saved {len(chunks)} chunks -> {OUT}")
    print(f"Words per chunk: min={words[0]} median={words[len(words)//2]} max={words[-1]}")
    print(f"Reference frameworks: {dict(frameworks.most_common())}")
    print(f"Controls without a profile level ({len(unlevelled)}): {unlevelled}")
    print(f"Unresolved placeholders: {leftover or 'none'}")
    print("\n--- example chunk text ---")
    print(next(c for c in chunks if c.control_id == "TP-4").text)


if __name__ == "__main__":
    main()