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


DEFAULT_EMBEDDINGS = REPO_ROOT / "corpus" / "embeddings.npz"


def chunk_digest(rows: list[tuple]) -> str:
    """Fingerprint of the exact (chunk_id, text) pairs, in order. Detects stale embeddings."""
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        h.update(r[0].encode()); h.update(b"\0"); h.update(r[4].encode()); h.update(b"\0")
    return h.hexdigest()


def load_embeddings(rows: list[tuple], path: Path | None = None):
    """Precomputed document vectors for `rows` (chunk_id, ..., content), or None if the file is
    missing or does not match these rows exactly."""
    import numpy as np
    p = Path(path) if path else DEFAULT_EMBEDDINGS
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=False)
    if str(data["digest"]) != chunk_digest(rows) or data["vectors"].shape[0] != len(rows):
        return None
    return data["vectors"].astype("float32")
