"""Torch reference implementations of the embedding model and cross-encoder.

Offline use only (building corpus/embeddings.npz, loading Supabase, the ONNX equivalence check).
The serving path never imports this module, so serving does not need torch."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from aegis.embed import MODEL_NAME, QUERY_PREFIX
from aegis.rerank import RERANK_MODEL


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def embed_documents(texts: list[str], show_progress: bool = True) -> np.ndarray:
    return _embedder().encode(texts, batch_size=32, normalize_embeddings=True,
                              show_progress_bar=show_progress)


def embed_query(query: str) -> np.ndarray:
    return _embedder().encode(QUERY_PREFIX + query, normalize_embeddings=True)


def rerank_scores(query: str, docs: list[str]) -> list[float]:
    """Raw logits (no sigmoid)."""
    import torch
    scores = _cross_encoder().predict([(query, d) for d in docs], activation_fn=torch.nn.Identity())
    return [float(s) for s in scores]
