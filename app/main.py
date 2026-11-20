"""FastAPI app: JSON endpoint, SSE stage stream, and the Pipeline Explorer page.

    uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from aegis import pipeline
from aegis.generate import answer_question

STATIC = Path(__file__).parent / "static"
MAX_QUESTION = 500


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION)
    guardrail: bool = True
    k: int = Field(default=5, ge=1, le=10)


def _safe_events(question: str, k: int, guardrail: bool, answerer: Callable) -> Iterator[dict]:
    """Pipeline events; a generation failure (e.g. missing GROQ_API_KEY) becomes an 'error' event
    instead of a broken stream, so retrieval stages are still shown."""
    try:
        for ev in pipeline.run_stages(question, k=k, use_guardrail=guardrail, answerer=answerer):
            yield {"stage": ev["stage"], "data": ev["data"]}
    except Exception as e:                                          # noqa: BLE001
        yield {"stage": "error", "data": {"message": f"{type(e).__name__}: {e}"[:300]}}


def create_app(answerer: Callable = answer_question, warmup: bool | None = None) -> FastAPI:
    if warmup is None:
        warmup = os.environ.get("AEGIS_WARMUP", "1") == "1"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if warmup:                              # load the models once so the first request is fast
            from aegis.rerank import get_reranker
            from aegis.retrieve import bm25_search, semantic_search
            semantic_search("warmup", 1)
            bm25_search("warmup", 1)
            get_reranker()
        yield

    app = FastAPI(title="Aegis", version="1.0", lifespan=lifespan,
                  description="Retrieval and grounded answers over the GovTech IM8 Reform controls.")

    @app.get("/health")
    def health() -> dict:
        from aegis.retrieve import backend
        from aegis import guardrail
        feature, threshold = guardrail.load_calibration()
        return {"status": "ok", "backend": backend(), "guardrail_feature": feature, "threshold": threshold}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        events = list(_safe_events(req.question, req.k, req.guardrail, answerer))
        final = next((e["data"] for e in reversed(events) if e["stage"] == "result"), None)
        err = next((e["data"] for e in events if e["stage"] == "error"), None)
        if final is None:
            raise HTTPException(status_code=502, detail=err["message"] if err else "pipeline failed")
        return {**final, "error": err["message"] if err else None,
                "stages": [e for e in events if e["stage"] != "result"]}

    @app.get("/ask/stream")
    def ask_stream(question: str = Query(min_length=3, max_length=MAX_QUESTION),
                   guardrail: bool = True, k: int = Query(5, ge=1, le=10)) -> StreamingResponse:
        def gen() -> Iterator[str]:
            for ev in _safe_events(question, k, guardrail, answerer):
                yield f"event: {ev['stage']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
