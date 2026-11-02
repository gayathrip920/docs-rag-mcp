"""The web API.

Run it from the repo root:  uvicorn api.main:app --port 8000

The visitor's OpenRouter key arrives in the X-OpenRouter-Key header on /ask
calls. It is handed to OpenRouter for that one request. It is not stored,
not logged, and the server never falls back to a key of its own.

For a public site, run it in "search only" mode (see the README): set
ENABLE_ASK=false and SEARCH_MODES=vector. Then no AI key is involved at all,
and the slow re-ranking model is never loaded.
"""

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag import answer as answer_module
from rag import llm
from rag.db import get_pool
from rag.llm import LLMError, public_message
from rag.retrieve import MODES, retrieve

log = logging.getLogger("api")

MODEL_NAME_RE = r"^[A-Za-z0-9._:/\-]{1,100}$"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


# Settings, read once when the server starts.
ENABLE_ASK = _env_bool("ENABLE_ASK", True)
SEARCH_MODES = tuple(
    m
    for m in (s.strip() for s in os.environ.get("SEARCH_MODES", ",".join(MODES)).split(","))
    if m in MODES
) or ("vector",)
DEFAULT_SEARCH_MODE = os.environ.get("DEFAULT_SEARCH_MODE", "hybrid_rerank")
if DEFAULT_SEARCH_MODE not in SEARCH_MODES:
    DEFAULT_SEARCH_MODE = SEARCH_MODES[0]
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))
GLOBAL_RATE_LIMIT_PER_MIN = int(os.environ.get("GLOBAL_RATE_LIMIT_PER_MIN", "0"))
MAX_CONCURRENT_SEARCHES = int(os.environ.get("MAX_CONCURRENT_SEARCHES", "4"))

app = FastAPI(title="docs-rag-mcp", version="0.1.0")

origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in origins if o],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-OpenRouter-Key"],
)


class RateLimiter:
    """Allow N calls per minute for each caller. Kept in memory, so it resets on restart."""

    def __init__(self, limit: int):
        self.limit = limit
        self.calls: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, caller: str) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic()
        with self.lock:
            window = self.calls[caller]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.limit:
                raise HTTPException(429, "Too many requests. Please wait a minute.")
            window.append(now)
            # Forget callers who have gone quiet, so the table cannot grow forever.
            if len(self.calls) > 10_000:
                for key in [k for k, v in self.calls.items() if not v]:
                    del self.calls[key]


limiter = RateLimiter(RATE_LIMIT_PER_MIN)
global_limiter = RateLimiter(GLOBAL_RATE_LIMIT_PER_MIN)  # all visitors together
search_slots = threading.BoundedSemaphore(max(MAX_CONCURRENT_SEARCHES, 1))


def limit_caller(request: Request) -> None:
    """Apply the per-visitor limit and the all-visitors limit."""
    limiter.check(request.client.host if request.client else "unknown")
    global_limiter.check("everyone")


@contextmanager
def search_slot():
    """Allow only a few searches at once. Others get a polite 'busy' answer."""
    if not search_slots.acquire(blocking=False):
        raise HTTPException(503, "The server is busy. Please try again in a few seconds.")
    try:
        yield
    finally:
        search_slots.release()


@contextmanager
def db_session():
    with get_pool().connection() as conn:
        yield conn


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    mode: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=10)
    model: str | None = Field(default=None, pattern=MODEL_NAME_RE)


def _require_ask_enabled() -> None:
    if not ENABLE_ASK:
        raise HTTPException(403, "Answers are turned off on this server. Search still works.")


@app.get("/health")
def health():
    """Quick check that the server is up. It does not touch the database."""
    return {"status": "ok"}


@app.get("/config")
def config():
    """What this server can do, so the web page knows which parts to show."""
    return {"ask_enabled": ENABLE_ASK, "search_modes": list(SEARCH_MODES)}


@app.post("/search")
def search(body: SearchRequest, request: Request):
    limit_caller(request)
    mode = body.mode or DEFAULT_SEARCH_MODE
    if mode not in SEARCH_MODES:
        raise HTTPException(422, f"mode must be one of {list(SEARCH_MODES)}")
    with search_slot(), db_session() as conn:
        hits = retrieve(conn, body.query, mode=mode, limit=body.k)
    return {
        "results": [
            {
                "page_path": h.page_path,
                "heading": h.heading,
                "url": h.url,
                "content": h.content,
                "score": h.score,
            }
            for h in hits
        ]
    }


def _llm_http_error(exc: LLMError) -> HTTPException:
    return HTTPException(exc.status_code, public_message(exc))


def _check_key_or_401(key: str | None) -> None:
    try:
        llm.clean_key(key)
    except LLMError as exc:
        raise _llm_http_error(exc) from exc


@app.post("/ask")
def ask(
    body: AskRequest,
    request: Request,
    x_openrouter_key: str | None = Header(default=None),
):
    _require_ask_enabled()
    limit_caller(request)
    _check_key_or_401(x_openrouter_key)  # before any slow work
    try:
        with search_slot(), db_session() as conn:
            result = answer_module.answer_question(
                conn,
                body.question,
                api_key=x_openrouter_key,
                model=body.model,
                k=body.k,
                mode=DEFAULT_SEARCH_MODE,
            )
    except LLMError as exc:
        raise _llm_http_error(exc) from exc
    return {
        "answer": result.answer,
        "found": result.found,
        "citations_valid": result.citations_valid,
        "sources": result.sources,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
    }


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/ask/stream")
def ask_stream(
    body: AskRequest,
    request: Request,
    x_openrouter_key: str | None = Header(default=None),
):
    _require_ask_enabled()
    limit_caller(request)
    _check_key_or_401(x_openrouter_key)  # before any slow work

    def events():
        try:
            with search_slot(), db_session() as conn:
                for event in answer_module.answer_stream(
                    conn,
                    body.question,
                    api_key=x_openrouter_key,
                    model=body.model,
                    k=body.k,
                    mode=DEFAULT_SEARCH_MODE,
                ):
                    yield _sse(event)
        except LLMError as exc:
            yield _sse({"type": "error", "message": public_message(exc)})
        except HTTPException as exc:
            yield _sse({"type": "error", "message": str(exc.detail)})
        except Exception:
            log.exception("stream failed")
            yield _sse({"type": "error", "message": "Something went wrong on the server."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/pages/{page_path:path}")
def get_page(page_path: str, request: Request):
    limit_caller(request)
    if not re.fullmatch(r"[A-Za-z0-9_\-./]{1,200}", page_path) or ".." in page_path:
        raise HTTPException(404, "Page not found.")
    with db_session() as conn:
        row = conn.execute(
            "SELECT path, title, url, body FROM pages WHERE path = %s", (page_path,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Page not found.")
    return {"path": row[0], "title": row[1], "url": row[2], "body": row[3]}


@app.get("/metrics")
def metrics(request: Request):
    """Numbers about the questions asked so far, taken from the queries table."""
    limit_caller(request)
    with db_session() as conn:
        row = conn.execute(
            "SELECT count(*), "
            "avg(CASE WHEN found THEN 1 ELSE 0 END), "
            "percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms), "
            "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), "
            "avg(cost_usd) FROM queries"
        ).fetchone()

    def number(value, digits):
        return None if value is None else round(float(value), digits)

    return {
        "questions": row[0],
        "found_rate": number(row[1], 3),
        "latency_ms_p50": number(row[2], 0),
        "latency_ms_p95": number(row[3], 0),
        "avg_cost_usd": number(row[4], 6),
    }
