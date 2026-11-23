"""Embed every chunk with the torch reference model and save corpus/embeddings.npz.

Offline only (needs requirements-eval.txt). The server loads this file instead of embedding at start-up.
    python -m scripts.build_embeddings
"""
import numpy as np

from aegis.corpus import DEFAULT_EMBEDDINGS, chunk_digest
from aegis.retrieve import _local_rows
from aegis.torch_ref import embed_documents


def main() -> None:
    rows = _local_rows()
    vectors = embed_documents([r[4] for r in rows], show_progress=False).astype("float32")
    np.savez_compressed(DEFAULT_EMBEDDINGS, ids=np.array([r[0] for r in rows]), vectors=vectors,
                        digest=np.array(chunk_digest(rows)))
    print(f"Saved {vectors.shape} -> {DEFAULT_EMBEDDINGS} ({DEFAULT_EMBEDDINGS.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
