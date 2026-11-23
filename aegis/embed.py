"""Runtime embeddings: ONNX (fastembed) version of BAAI/bge-small-en-v1.5. No torch needed.

The torch reference implementation lives in aegis/torch_ref.py (offline use only)."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384 dimensions, ~33M parameters
EMBEDDING_DIM = 384

# bge models are trained to expect this prefix on *queries* (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def model_dir() -> str:
    """Where ONNX model files are cached (baked into the Docker image at build time)."""
    return os.environ.get("AEGIS_MODEL_DIR", os.path.expanduser("~/.cache/aegis-models"))


@lru_cache(maxsize=1)
def get_model():
    from fastembed import TextEmbedding
    return TextEmbedding(MODEL_NAME, cache_dir=model_dir())


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def embed_documents(texts: list[str], show_progress: bool = True) -> np.ndarray:
    vecs = np.array(list(get_model().embed(texts, batch_size=32)), dtype=np.float32)
    return _unit(vecs)


def embed_query(query: str) -> np.ndarray:
    return embed_documents([QUERY_PREFIX + query])[0]
