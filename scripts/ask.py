"""Ask a question end to end (with guardrail):  python -m scripts.ask "Do admins need MFA?" """
import sys

from aegis.pipeline import ask


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How often should backups be tested?"
    res = ask(question)
    print(f"Q: {question}\n")
    print(f"A: {res.answer}\n")
    if res.refused:
        print(f"REFUSED by guardrail ({res.refusal_reason}); the LLM was not called.")
    print(f"Cited: {res.cited_ids}")
    if res.invalid_citations:
        print(f"WARNING - cited IDs that were never retrieved: {res.invalid_citations}")
    print("\nRetrieved:")
    for h in res.hits:
        mark = "*" if h.control_id in res.cited_ids else " "
        print(f" {mark} {h.control_id:6} {h.score:7.3f}  {h.title}")


if __name__ == "__main__":
    main()
