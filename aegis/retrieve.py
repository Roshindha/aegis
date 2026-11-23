"""Retrieval: semantic (vector) search + BM25 keyword search, merged with
reciprocal rank fusion (RRF)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from aegis.corpus import load_chunks, load_embeddings
from aegis.embed import embed_documents, embed_query

# Two interchangeable backends hold the same data and use the same embedding model:
#   "local" (default): read corpus/chunks.jsonl, keep vectors in memory. No database needed,
#                      so tests, CI, Docker and Hugging Face Spaces all work offline.
#   "db":              Supabase Postgres + pgvector (set AEGIS_BACKEND=db and DATABASE_URL).
def backend() -> str:
    return os.environ.get("AEGIS_BACKEND", "local").lower()


def _connect():
    from aegis.db import connect   # imported lazily: the local backend never needs psycopg
    return connect()

# ---------- data type ----------


@dataclass
class Hit:
    chunk_id: str
    control_id: str
    title: str
    group_title: str
    content: str
    score: float                      # the score of the method that produced this hit
    semantic_rank: int | None = None  # 1-based rank in the semantic list (None = absent)
    bm25_rank: int | None = None      # 1-based rank in the BM25 list (None = absent)
    via: str = ""                     # "crosswalk" if injected by an index lookup, else ""


# ---------- semantic search (Step 3) ----------

SEMANTIC_SQL = """
    select chunk_id, control_id, title, group_title, content,
           1 - (embedding <=> %(q)s) as score
    from chunks
    order by embedding <=> %(q)s
    limit %(k)s
"""


def _local_rows() -> list[tuple]:
    """Same columns and order as the database query: (chunk_id, control_id, title, group_title, content)."""
    chunks = sorted(load_chunks(), key=lambda c: c["chunk_id"])
    return [(c["chunk_id"], c["control_id"], c["title"], c["group_title"], c["text"]) for c in chunks]


@lru_cache(maxsize=1)
def _local_index() -> tuple[list[tuple], np.ndarray]:
    rows = _local_rows()
    vectors = load_embeddings(rows)                 # committed corpus/embeddings.npz (torch-built)
    if vectors is None:                             # stale/missing file: embed with the ONNX model
        vectors = embed_documents([r[4] for r in rows], show_progress=False)
    return rows, vectors                            # unit-length vectors


def semantic_search(query: str, k: int = 5) -> list[Hit]:
    q = embed_query(query)
    if backend() == "db":
        with _connect() as conn:
            rows = conn.execute(SEMANTIC_SQL, {"q": q, "k": k}).fetchall()
        return [Hit(*row, semantic_rank=i) for i, row in enumerate(rows, start=1)]
    rows, matrix = _local_index()
    sims = matrix @ q                       # cosine similarity: both sides are unit length
    top = np.argsort(-sims)[:k]
    return [Hit(*rows[i], score=float(sims[i]), semantic_rank=r) for r, i in enumerate(top, start=1)]


# ---------- BM25 keyword search (Step 4) ----------

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "how",
    "in", "is", "it", "of", "on", "or", "the", "to", "what", "which", "with",
}
# One token = a word, OR an ID like "as-5" / "lm-14" kept whole.
TOKEN_RE = re.compile(r"[a-z]+-\d+|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    # "AS-5" must survive even though "as" is a stopword, so filter AFTER tokenising.
    return [t for t in tokens if t not in STOPWORDS]


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[list[tuple], BM25Okapi]:
    """Read all chunks once and build the BM25 index in memory (137 rows: instant)."""
    if backend() == "db":
        with _connect() as conn:
            rows = conn.execute(
                "select chunk_id, control_id, title, group_title, content from chunks order by chunk_id"
            ).fetchall()
    else:
        rows = _local_rows()
    return rows, BM25Okapi([tokenize(r[4]) for r in rows])


def bm25_search(query: str, k: int = 5) -> list[Hit]:
    rows, bm25 = _load_corpus()
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        Hit(*rows[i], score=float(scores[i]), bm25_rank=rank)
        for rank, i in enumerate(ranked, start=1)
        if scores[i] > 0                      # a score of 0 means "no query word matched"
    ]


# ---------- fusion (Step 4) ----------

RRF_K = 60   # standard constant from the original RRF paper; damps the top ranks


def rrf_fuse(lists: list[list[Hit]], k: int = 5) -> list[Hit]:
    """Reciprocal rank fusion: each list votes 1/(RRF_K + rank) for every chunk it returns.
    Uses ranks, not raw scores, so cosine similarity and BM25 never need to be compared."""
    fused: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in lists:
        for rank, hit in enumerate(hits, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            if hit.chunk_id not in best:
                best[hit.chunk_id] = replace(hit)       # copy: never mutate the caller's hits
            else:                                   # merge rank info from both methods
                best[hit.chunk_id].semantic_rank = best[hit.chunk_id].semantic_rank or hit.semantic_rank
                best[hit.chunk_id].bm25_rank = best[hit.chunk_id].bm25_rank or hit.bm25_rank
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    for cid in top:
        best[cid].score = fused[cid]
    return [best[cid] for cid in top]


def hybrid_search(query: str, k: int = 5, candidates: int = 20) -> list[Hit]:
    return rrf_fuse([semantic_search(query, candidates), bm25_search(query, candidates)], k)
