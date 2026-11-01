"""Measure how well each search method finds the right doc page.

Run from the repo root (needs DATABASE_URL and the model libraries):

    python -m evals.run_retrieval_eval

For every question it asks for the best chunks, turns them into a list of
pages, and checks where the correct page landed. Results go to
evals/results/retrieval.json.
"""

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from evals.metrics import mean, percentile, recall_at_k, reciprocal_rank
from rag.chunking import MAX_WORDS, OVERLAP_WORDS
from rag.config import load_settings
from rag.db import connect
from rag.embed import embed_texts
from rag.retrieve import MODES, rank_pages, retrieve

ROOT = Path(__file__).resolve().parent.parent
K = 5  # recall@5
MRR_CUTOFF = 10  # MRR@10
CANDIDATES = 30  # chunks taken from each search before merging


def load_questions() -> list[dict]:
    lines = (ROOT / "evals" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate_mode(conn, mode: str, questions: list[dict], vectors) -> dict:
    recalls, ranks, times, misses = [], [], [], []
    for question, vector in zip(questions, vectors, strict=True):
        started = time.perf_counter()
        hits = retrieve(
            conn,
            question["question"],
            mode=mode,
            limit=CANDIDATES,
            candidates=CANDIDATES,
            query_vector=vector,
        )
        times.append((time.perf_counter() - started) * 1000)
        pages = rank_pages(hits)
        recalls.append(recall_at_k(pages, question["gold_pages"], K))
        ranks.append(reciprocal_rank(pages, question["gold_pages"], MRR_CUTOFF))
        if recalls[-1] < 1.0:
            misses.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "gold_pages": question["gold_pages"],
                    "top_pages": pages[:K],
                }
            )
    return {
        f"recall@{K}": round(mean(recalls), 4),
        f"mrr@{MRR_CUTOFF}": round(mean(ranks), 4),
        "retrieval_ms_p50": round(percentile(times, 50), 1),
        "retrieval_ms_p95": round(percentile(times, 95), 1),
        "missed_questions": misses,
    }


def main() -> int:
    settings = load_settings()
    questions = load_questions()
    print(f"{len(questions)} questions, embedding them once...")
    vectors = embed_texts([q["question"] for q in questions])

    conn = connect()
    results = {}
    for mode in MODES:
        print(f"running {mode}...")
        results[mode] = evaluate_mode(conn, mode, questions, vectors)
        row = results[mode]
        print(f"  recall@{K} {row[f'recall@{K}']}  mrr@{MRR_CUTOFF} {row[f'mrr@{MRR_CUTOFF}']}")

    output = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "questions": len(questions),
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model,
        "chunk_max_words": MAX_WORDS,
        "chunk_overlap_words": OVERLAP_WORDS,
        "candidates_per_search": CANDIDATES,
        "note": (
            f"recall@{K} is the share of correct pages found in the top {K} pages. "
            f"mrr@{MRR_CUTOFF} is the mean of 1/position of the first correct page."
        ),
        "results": results,
    }
    out_path = ROOT / "evals" / "results" / "retrieval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
