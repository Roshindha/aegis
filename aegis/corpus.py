"""Load the chunked corpus from the committed snapshot (no database needed)."""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "corpus" / "chunks.jsonl"


def corpus_path() -> Path:
    return Path(os.environ.get("AEGIS_CORPUS", DEFAULT_CORPUS))


def load_chunks(path: Path | None = None) -> list[dict]:
    p = Path(path) if path else corpus_path()
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
