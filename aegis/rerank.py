"""Cross-encoder reranking: read (query, chunk) TOGETHER and score how well they match."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from aegis.retrieve import Hit, hybrid_search

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ONNX_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # ONNX export of the same cross-encoder


class OnnxReranker:
    """fastembed (ONNX) cross-encoder: no torch at serving time."""

    def __init__(self) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        from aegis.embed import low_memory_onnx, model_dir
        with low_memory_onnx():
            self._model = TextCrossEncoder(ONNX_RERANK_MODEL, cache_dir=model_dir(), threads=1, enable_cpu_mem_arena=False)

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, docs, batch_size=4)]   # raw logits, same order as docs


@lru_cache(maxsize=1)
def get_reranker() -> OnnxReranker:
    return OnnxReranker()


def rerank(query: str, hits: list[Hit], k: int = 5) -> list[Hit]:
    """Re-score candidates with the cross-encoder and keep the best k.

    Returns copies of the hits with .score = the cross-encoder score (a raw logit:
    higher = more relevant, can be negative). The old RRF score is dropped on purpose.
    """
    if not hits:
        return []
    scores = get_reranker().score(query, [h.content for h in hits])  # raw logits
    scored = [replace(h, score=float(s)) for h, s in zip(hits, scores)]
    return sorted(scored, key=lambda h: h.score, reverse=True)[:k]


def search(query: str, k: int = 5, candidates: int = 20) -> list[Hit]:
    """The full retrieval pipeline: hybrid (wide net) -> rerank (careful read)."""
    return rerank(query, hybrid_search(query, k=candidates), k=k)
