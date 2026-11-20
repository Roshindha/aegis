"""Groq calls for the eval: exponential backoff on HTTP 429 and an on-disk response cache."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Callable

from aegis.generate import Answer, DEFAULT_MODEL, extract_citations
from aegis.retrieve import Hit

CACHE = Path("eval/cache/llm.jsonl")


class DailyLimitReached(RuntimeError):
    """Groq's per-day quota is exhausted: retrying today is pointless."""


def cache_key(model: str, question: str, hits: list[Hit]) -> str:
    h = hashlib.sha256()
    for part in (model, question, *[x.content for x in hits]):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def load_cache(path: Path = CACHE) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["key"]] = d["text"]
    return out


def append_cache(key: str, text: str, path: Path = CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")


def _retry_after(err: Exception) -> float | None:
    m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(err))
    if m:
        return float(m.group(3)) + 60 * float(m.group(2) or 0) + 3600 * float(m.group(1) or 0)
    return None


def call_with_backoff(fn: Callable[[], object], *, max_retries: int = 8, base: float = 2.0,
                      sleep: Callable[[float], None] = time.sleep, is_rate_limit=None) -> object:
    """Retry fn() on HTTP 429 with exponential backoff (honouring Groq's 'try again in Xs')."""
    if is_rate_limit is None:
        from groq import RateLimitError
        is_rate_limit = lambda e: isinstance(e, RateLimitError)
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:                                   # noqa: BLE001
            if not is_rate_limit(e):
                raise
            msg = str(e).lower()
            if "per day" in msg or "tokens per day" in msg or "requests per day" in msg:
                wait = _retry_after(e)
                if wait is None or wait > 600:                   # more than 10 min = daily cap
                    raise DailyLimitReached(str(e)) from e
            if attempt == max_retries:
                raise
            sleep(_retry_after(e) or base ** attempt)
    raise AssertionError("unreachable")


def make_cached_answerer(raw_answer: Callable[[str, list[Hit]], Answer], cache: dict[str, str],
                         model: str | None = None, path: Path = CACHE,
                         sleep: Callable[[float], None] = time.sleep, is_rate_limit=None):
    """Wrap an answer function with disk caching + 429 backoff. Cache key = model+question+context."""
    model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    def answerer(question: str, hits: list[Hit]) -> Answer:
        key = cache_key(model, question, hits)
        if key in cache:
            cited, invalid = extract_citations(cache[key], hits)
            return Answer(cache[key], cited, invalid, hits, model)
        ans = call_with_backoff(lambda: raw_answer(question, hits), sleep=sleep, is_rate_limit=is_rate_limit)
        cache[key] = ans.text
        append_cache(key, ans.text, path)
        return ans

    return answerer
