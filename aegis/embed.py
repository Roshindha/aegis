"""Turn text into vectors with a small, CPU-friendly embedding model."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384 dimensions, ~33M parameters
EMBEDDING_DIM = 384

# bge models are trained to expect this prefix on *queries* (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load once, reuse forever. Loading takes seconds; encoding takes milliseconds."""
    return SentenceTransformer(MODEL_NAME)


def embed_documents(texts: list[str], show_progress: bool = True) -> np.ndarray:
    return get_model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=show_progress
    )


def embed_query(query: str) -> np.ndarray:
    return get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
