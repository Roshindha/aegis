"""Cross-encoder reranking: read (query, chunk) TOGETHER and score how well they match."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from aegis.retrieve import Hit, hybrid_search

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def rerank(query: str, hits: list[Hit], k: int = 5) -> list[Hit]:
    """Re-score candidates with the cross-encoder and keep the best k.

    Returns copies of the hits with .score = the cross-encoder score (a raw logit:
    higher = more relevant, can be negative). The old RRF score is dropped on purpose.
    """
    if not hits:
        return []
    pairs = [(query, h.content) for h in hits]
    scores = get_reranker().predict(pairs, activation_fn=torch.nn.Identity())  # raw logits
    scored = [replace(h, score=float(s)) for h, s in zip(hits, scores)]
    return sorted(scored, key=lambda h: h.score, reverse=True)[:k]


def search(query: str, k: int = 5, candidates: int = 20) -> list[Hit]:
    """The full retrieval pipeline: hybrid (wide net) -> rerank (careful read)."""
    return rerank(query, hybrid_search(query, k=candidates), k=k)
