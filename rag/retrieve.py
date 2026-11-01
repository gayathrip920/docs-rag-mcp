"""Find the chunks that best match a question.

Three modes, from simple to fancy:

    vector         meaning search only
    hybrid         meaning search plus keyword search, merged
    hybrid_rerank  hybrid, then a second model re-scores the top results
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from rag.config import load_settings

MODES = ("vector", "hybrid", "hybrid_rerank")
RRF_K = 60  # standard constant for reciprocal rank fusion

COLUMNS = "id, page_path, heading, url, content"


@dataclass
class Hit:
    id: int
    page_path: str
    heading: str
    url: str
    content: str
    score: float = 0.0


def build_tsquery(question: str) -> str | None:
    """Turn a question into an OR query like 'upload | file | users'.

    Postgres plain search needs every word to match. Questions have filler
    words, so we ask for any word instead and let the ranking sort it out.
    """
    words = re.findall(r"[a-z0-9]+", question.lower())
    unique = list(dict.fromkeys(w for w in words if len(w) > 1))
    return " | ".join(unique) if unique else None


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge several ranked id lists. Items ranked high in any list float to the top."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def rank_pages(hits: list[Hit]) -> list[str]:
    """Page paths in order of first appearance, without repeats."""
    return list(dict.fromkeys(hit.page_path for hit in hits))


def _row_to_hit(row) -> Hit:
    return Hit(id=row[0], page_path=row[1], heading=row[2], url=row[3], content=row[4])


def vector_search(conn, query_vector, limit: int) -> list[Hit]:
    rows = conn.execute(
        f"SELECT {COLUMNS} FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (query_vector, limit),
    ).fetchall()
    return [_row_to_hit(r) for r in rows]


def keyword_search(conn, question: str, limit: int) -> list[Hit]:
    tsquery = build_tsquery(question)
    if tsquery is None:
        return []
    rows = conn.execute(
        f"SELECT {COLUMNS} FROM chunks WHERE tsv @@ to_tsquery('english', %s) "
        "ORDER BY ts_rank_cd(tsv, to_tsquery('english', %s)) DESC LIMIT %s",
        (tsquery, tsquery, limit),
    ).fetchall()
    return [_row_to_hit(r) for r in rows]


@lru_cache(maxsize=1)
def _reranker():
    # Imported here so that tests do not need the model libraries.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(load_settings().rerank_model)


def rerank(question: str, hits: list[Hit]) -> list[Hit]:
    if not hits:
        return []
    scores = _reranker().predict([(question, hit.content) for hit in hits])
    for hit, score in zip(hits, scores, strict=True):
        hit.score = float(score)
    return sorted(hits, key=lambda hit: hit.score, reverse=True)


def retrieve(
    conn,
    question: str,
    mode: str = "hybrid_rerank",
    limit: int = 5,
    candidates: int = 30,
    query_vector=None,
) -> list[Hit]:
    """Return the best `limit` chunks for a question.

    `candidates` is how many chunks each search gives before merging and
    re-ranking. `query_vector` can be passed in to avoid embedding twice.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    if query_vector is None:
        from rag.embed import embed_texts

        query_vector = embed_texts([question])[0]

    vector_hits = vector_search(conn, query_vector, candidates)
    if mode == "vector":
        for hit, rank in zip(vector_hits, range(1, len(vector_hits) + 1), strict=True):
            hit.score = 1.0 / rank
        return vector_hits[:limit]

    keyword_hits = keyword_search(conn, question, candidates)
    by_id = {hit.id: hit for hit in vector_hits + keyword_hits}
    fused = reciprocal_rank_fusion([[h.id for h in vector_hits], [h.id for h in keyword_hits]])
    merged = []
    for chunk_id, score in fused:
        hit = by_id[chunk_id]
        hit.score = score
        merged.append(hit)

    if mode == "hybrid":
        return merged[:limit]
    return rerank(question, merged[:candidates])[:limit]
