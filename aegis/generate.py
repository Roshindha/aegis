"""Generation: turn retrieved controls into a short, cited answer."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

from aegis.retrieve import Hit

load_dotenv()

# Model names change (llama-3.1-8b-instant was retired on Groq in Aug 2026), so the model
# is configuration, not code. Override with GROQ_MODEL in .env.
DEFAULT_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You answer questions about the Singapore Government's IM8 Reform security \
controls. You are given a set of controls. Follow these rules strictly:

1. Use ONLY the controls provided. Never use outside knowledge.
2. After every claim, cite the control ID in square brackets, e.g. [AS-5]. Cite only IDs \
that appear in the provided controls.
3. If the provided controls do not answer the question, reply exactly: \
"The provided IM8 controls do not answer this question." Do not guess.
4. Text like [agency-defined: time period (days)] means the catalogue leaves that value to \
the agency. Say it is agency-defined. NEVER invent a number.
5. Only state a control's applicability level (Level 0/1/2) if the control text gives it. \
If it says no profile level is assigned, say so.
6. Be concise: at most 5 sentences."""

CITATION_RE = re.compile(r"\[([A-Z]{2}-\d+)\]")


@dataclass
class Answer:
    text: str
    cited_ids: list[str]                 # control IDs the model cited, in order, deduplicated
    invalid_citations: list[str]         # cited IDs that were NOT in the retrieved context
    sources: list[Hit] = field(default_factory=list)
    model: str = ""


@lru_cache(maxsize=1)
def _client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env (console.groq.com/keys).")
    return Groq(api_key=key)


def build_context(hits: list[Hit]) -> str:
    # hit.content already starts with "AS-5: Password Requirements" and has the applicability line
    return "\n\n".join(f"### Control\n{h.content}" for h in hits)


def extract_citations(text: str, hits: list[Hit]) -> tuple[list[str], list[str]]:
    allowed = {h.control_id for h in hits}
    cited = list(dict.fromkeys(CITATION_RE.findall(text)))      # dedupe, keep order
    return cited, [c for c in cited if c not in allowed]


def answer_question(question: str, hits: list[Hit]) -> Answer:
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    user_msg = f"Controls:\n\n{build_context(hits)}\n\nQuestion: {question}"
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_msg}],
        temperature=0,                    # same question -> same answer (repeatable evals)
        reasoning_effort="low",           # gpt-oss models think before answering; keep it cheap
        max_completion_tokens=2000,       # thinking tokens count too; too small = empty answer
    )
    text = (resp.choices[0].message.content or "").strip()
    cited, invalid = extract_citations(text, hits)
    return Answer(text=text, cited_ids=cited, invalid_citations=invalid, sources=hits, model=model)
