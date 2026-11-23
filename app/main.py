"""FastAPI app: JSON endpoint, SSE stage stream, and (locally) the static Pipeline Explorer from docs/.

    uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import os
import threading
import unicodedata
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Iterator

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aegis import pipeline
from aegis.generate import DEFAULT_MODEL, Answer, answer_question
from app.ratelimit import RateLimiter

DOCS = Path(__file__).resolve().parent.parent / "docs"
MAX_QUESTION = 500
LOCAL_ORIGIN_RE = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
DEFAULT_ALLOWED_ORIGINS = "https://roshindha.github.io"
QUOTA_MESSAGE = "Demo quota reached \u2014 the answer service is rate-limited on the free tier. Try again later; retrieval results are still shown."
LLM_ERROR_MESSAGE = "The answer service had a problem. Retrieval results are still shown."


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION)
    guardrail: bool = True
    k: int = Field(default=5, ge=1, le=10)


def normalise_question(q: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", q).lower().split())


def classify_llm_error(exc: Exception) -> dict:
    """Groq 429 or a missing key = "demo quota reached"; anything else = generic. No details leak."""
    status = getattr(exc, "status_code", None)
    if status == 429 or type(exc).__name__ == "RateLimitError" or "GROQ_API_KEY" in str(exc):
        return {"kind": "quota", "message": QUOTA_MESSAGE}
    return {"kind": "llm_error", "message": LLM_ERROR_MESSAGE}


def client_ip(request: Request) -> str:
    """Behind Render's proxy the rightmost X-Forwarded-For entry is the one the proxy appended."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


class AnswerCache:
    """In-memory LRU of full event lists, keyed by normalised question."""

    def __init__(self, size: int = 256) -> None:
        self.size, self._d, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, events) -> None:
        with self._lock:
            self._d[key] = events
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


def create_app(answerer: Callable = answer_question, warmup: bool | None = None,
               limiter: RateLimiter | None = None) -> FastAPI:
    if warmup is None:
        warmup = os.environ.get("AEGIS_WARMUP", "1") == "1"
    limiter = limiter or RateLimiter(int(os.environ.get("RATE_PER_MIN", 10)), int(os.environ.get("RATE_PER_DAY", 50)))
    cache = AnswerCache()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if warmup:                              # load the models once so the first request is fast
            from aegis.rerank import get_reranker
            from aegis.retrieve import bm25_search, db_ping, semantic_search
            db_ping()                           # if the database is down, the snapshot is used from the start
            semantic_search("warmup", 1)
            bm25_search("warmup", 1)
            get_reranker().score("warmup", ["warmup"])
        yield

    app = FastAPI(title="Aegis", version="1.0", lifespan=lifespan,
                  description="Retrieval and grounded answers over the GovTech IM8 Reform controls.")
    origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_origin_regex=LOCAL_ORIGIN_RE,
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

    def events_for(question: str, k: int, guardrail: bool) -> Iterator[dict]:
        """Pipeline events. A generation failure becomes an 'error' event (the answer stage is replaced),
        so retrieval stages and the final result are still delivered. Complete, error-free runs are cached."""
        key = (normalise_question(question), k, guardrail)
        hit = cache.get(key)
        if hit is not None:
            yield from hit
            return
        err: dict = {}

        def guarded(q, hits):
            try:
                return answerer(q, hits)
            except Exception as e:                                      # noqa: BLE001
                err.update(classify_llm_error(e))
                return Answer(text="", cited_ids=[], invalid_citations=[], sources=hits, model="unavailable")

        out: list[dict] = []
        try:
            for ev in pipeline.run_stages(question, k=k, use_guardrail=guardrail, answerer=guarded):
                e = {"stage": ev["stage"], "data": ev["data"]}
                if e["stage"] == "answer" and err:
                    e = {"stage": "error", "data": dict(err)}
                out.append(e)
                yield e
        except Exception as e:                                          # noqa: BLE001
            yield {"stage": "error", "data": {"kind": "pipeline_error", "message": "Retrieval failed unexpectedly."}}
            return
        if not err:
            cache.put(key, out)

    def limited(request: Request):
        ok, retry, which = limiter.check(client_ip(request))
        if ok:
            return None
        msg = ("Daily limit reached for this demo (50 questions per day). Please come back tomorrow."
               if which == "day" else "Too many questions \u2014 the demo allows 10 per minute. Please wait a moment and try again.")
        return JSONResponse({"error": "rate_limited", "limit": which, "message": msg, "retry_after": retry},
                            status_code=429, headers={"Retry-After": str(retry)})

    @app.get("/health")
    def health() -> dict:
        from aegis.retrieve import backend, db_ping
        db_ping()                                # one cheap query: keeps Supabase active, detects outages
        return {"status": "ok", "backend": backend(), "model": os.environ.get("GROQ_MODEL", DEFAULT_MODEL)}

    @app.post("/ask")
    def ask(req: AskRequest, request: Request):
        if (blocked := limited(request)) is not None:
            return blocked
        events = list(events_for(req.question, req.k, req.guardrail))
        final = next((e["data"] for e in reversed(events) if e["stage"] == "result"), None)
        err = next((e["data"] for e in events if e["stage"] == "error"), None)
        if final is None:
            return JSONResponse({"error": err or {"kind": "pipeline_error", "message": "pipeline failed"}}, status_code=502)
        return {**final, "error": err, "stages": [e for e in events if e["stage"] != "result"]}

    @app.get("/ask/stream")
    def ask_stream(request: Request, question: str = Query(min_length=3, max_length=MAX_QUESTION),
                   guardrail: bool = True, k: int = Query(5, ge=1, le=10)):
        if (blocked := limited(request)) is not None:
            return blocked

        def gen() -> Iterator[str]:
            for ev in events_for(question, k, guardrail):
                yield f"event: {ev['stage']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if DOCS.exists():                            # the same static page GitHub Pages serves, for local use
        app.mount("/", StaticFiles(directory=DOCS, html=True), name="site")
    return app


app = create_app()
