"""Pre-download the two ONNX models (used at Docker build time so the container starts offline)."""
from aegis.embed import get_model
from aegis.rerank import get_reranker

if __name__ == "__main__":
    get_model()
    get_reranker()
    print("models cached")
