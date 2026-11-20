"""Ask a question end to end:  python -m scripts.ask "Do admins need MFA?" """
import sys

from aegis.generate import answer_question
from aegis.rerank import search


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How often should backups be tested?"
    hits = search(question, k=5)
    ans = answer_question(question, hits)

    print(f"Q: {question}\n")
    print(f"A: {ans.text}\n")
    print(f"Model: {ans.model}")
    print(f"Cited: {ans.cited_ids}")
    if ans.invalid_citations:
        print(f"WARNING - cited IDs that were never retrieved: {ans.invalid_citations}")
    print("\nRetrieved:")
    for h in hits:
        mark = "*" if h.control_id in ans.cited_ids else " "
        print(f" {mark} {h.control_id:6} {h.score:7.3f}  {h.title}")


if __name__ == "__main__":
    main()
