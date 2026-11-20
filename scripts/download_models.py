"""Pre-download the two model weights (used at Docker build time so the container starts offline)."""
from sentence_transformers import CrossEncoder, SentenceTransformer

from aegis.embed import MODEL_NAME
from aegis.rerank import RERANK_MODEL

if __name__ == "__main__":
    SentenceTransformer(MODEL_NAME)
    CrossEncoder(RERANK_MODEL)
    print("models cached")
