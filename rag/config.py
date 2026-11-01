"""Settings read from environment variables (or a local .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional, plain env vars still work
    pass

ROOT = Path(__file__).resolve().parent.parent

# The embedding model below produces vectors of this size.
# If you change EMBEDDING_MODEL, change this number and the database column too.
EMBEDDING_DIM = 384


@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_model: str
    rerank_model: str
    data_dir: Path
    docs_base_url: str


def load_settings() -> Settings:
    return Settings(
        database_url=os.environ.get("DATABASE_URL", ""),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        rerank_model=os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        data_dir=Path(os.environ.get("DATA_DIR", ROOT / "data" / "raw")),
        docs_base_url=os.environ.get("DOCS_BASE_URL", "https://fastapi.tiangolo.com").rstrip("/"),
    )
