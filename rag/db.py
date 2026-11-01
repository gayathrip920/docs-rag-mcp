"""Database connection and table setup (Postgres with pgvector, for example on Neon)."""

import psycopg
from pgvector.psycopg import register_vector

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


def connect() -> psycopg.Connection:
    """Open a connection. Needs DATABASE_URL to be set."""
    url = load_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    # prepare_threshold=None keeps this working with Neon's pooled connection string.
    conn = psycopg.connect(url, prepare_threshold=None)
    # The extension must exist before the vector type can be registered.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()
