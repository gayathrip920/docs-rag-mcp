"""Database connection and table setup (Postgres with pgvector, for example on Neon)."""

import os
import threading

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from rag.config import EMBEDDING_DIM, load_settings

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pages (
    path text PRIMARY KEY,
    title text NOT NULL,
    url text NOT NULL,
    body text NOT NULL,
    page_hash text NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id serial PRIMARY KEY,
    page_path text NOT NULL REFERENCES pages(path) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    heading text NOT NULL,
    url text NOT NULL,
    content text NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector({EMBEDDING_DIM}) NOT NULL,
    UNIQUE (page_path, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS queries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    question text NOT NULL,
    source text NOT NULL,            -- web or mcp
    latency_ms int,
    cost_usd numeric,
    found boolean,
    created_at timestamptz NOT NULL DEFAULT now()
);
"""


def _url() -> str:
    url = load_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    return url


def connect() -> psycopg.Connection:
    """Open one connection. Used by scripts such as ingest and the evals."""
    # prepare_threshold=None keeps this working with Neon's pooled connection string.
    conn = psycopg.connect(_url(), prepare_threshold=None)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        # First run on a new database: the vector type does not exist yet.
        conn.rollback()
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
    return conn


_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """A small shared pool for the web API, so each request does not open a new connection."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                _url(),
                min_size=1,
                max_size=int(os.environ.get("POOL_MAX", "5")),
                max_idle=240,  # drop idle connections before Neon suspends them
                kwargs={"prepare_threshold": None},
                configure=register_vector,
                check=ConnectionPool.check_connection,
                open=False,
            )
            _pool.open(wait=False)
        return _pool


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()
