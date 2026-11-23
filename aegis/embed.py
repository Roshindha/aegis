"""Runtime embeddings: ONNX (fastembed) version of BAAI/bge-small-en-v1.5. No torch needed.

The torch reference implementation lives in aegis/torch_ref.py (offline use only)."""
from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384 dimensions, ~33M parameters
EMBEDDING_DIM = 384

# bge models are trained to expect this prefix on *queries* (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def model_dir() -> str:
    """Where ONNX model files are cached (baked into the Docker image at build time)."""
    return os.environ.get("AEGIS_MODEL_DIR", os.path.expanduser("~/.cache/aegis-models"))


@contextmanager
def low_memory_onnx():
    """Build ONNX sessions without graph optimisation: ORT's optimiser keeps a second copy of the
    weights (~110 MB for bge-small), which matters on a 512 MB host. Outputs are unchanged to ~1e-5
    (checked by scripts/equivalence.py)."""
    import onnxruntime as ort
    original = ort.SessionOptions

    off = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    level = original.graph_optimization_level

    class LowMemoryOptions(original):
        def __init__(self) -> None:
            super().__init__()
            level.__set__(self, off)
            self.enable_mem_pattern = False

        @property
        def graph_optimization_level(self):
            return level.__get__(self)

        @graph_optimization_level.setter
        def graph_optimization_level(self, _ignored) -> None:      # fastembed sets ENABLE_ALL: keep it off
            level.__set__(self, off)
    ort.SessionOptions = LowMemoryOptions
    try:
        yield
    finally:
        ort.SessionOptions = original


@lru_cache(maxsize=1)
def get_model():
    from fastembed import TextEmbedding
    with low_memory_onnx():
        return TextEmbedding(MODEL_NAME, cache_dir=model_dir(), threads=1, enable_cpu_mem_arena=False)


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def embed_documents(texts: list[str], show_progress: bool = True) -> np.ndarray:
    vecs = np.array(list(get_model().embed(texts, batch_size=32)), dtype=np.float32)
    return _unit(vecs)


def embed_query(query: str) -> np.ndarray:
    return embed_documents([QUERY_PREFIX + query])[0]
