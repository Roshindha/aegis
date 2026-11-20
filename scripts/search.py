"""Compare retrieval methods:  python -m scripts.search "AS-5" """
import sys

from aegis.rerank import search
from aegis.retrieve import bm25_search, hybrid_search, semantic_search


def show(name: str, hits) -> None:
    print(f"--- {name} ---")
    for rank, h in enumerate(hits, start=1):
        print(f"{rank}. {h.control_id:6} {h.score:7.3f}  {h.title}  "
              f"(sem#{h.semantic_rank}, bm25#{h.bm25_rank})")
    print()


def main() -> None:
    query = " ".join(sys.argv[1:]) or "How often should backups be tested?"
    print(f"Query: {query}\n")
    show("semantic", semantic_search(query, k=5))
    show("bm25", bm25_search(query, k=5))
    show("hybrid (RRF)", hybrid_search(query, k=5))
    show("hybrid + rerank (final)", search(query, k=5))


if __name__ == "__main__":
    main()
