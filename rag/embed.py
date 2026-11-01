"""Turn text into vectors with a local sentence-transformers model (free, no API key)."""

from functools import lru_cache

from rag.config import EMBEDDING_DIM, load_settings


@lru_cache(maxsize=1)
def _model():
    # Imported here so that tests and tools that never embed do not need torch installed.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(load_settings().embedding_model)
    size = model.get_sentence_embedding_dimension()
    if size != EMBEDDING_DIM:
        raise RuntimeError(
            f"The embedding model gives {size} numbers per text but the database expects "
            f"{EMBEDDING_DIM}. Update EMBEDDING_DIM in rag/config.py and recreate the tables."
        )
    return model


def embed_texts(texts: list[str], batch_size: int = 32):
    """Return a numpy array with one normalized vector per text."""
    return _model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
