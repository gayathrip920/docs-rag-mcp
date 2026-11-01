"""Scores for judging a search. Pages are compared by their path."""


def recall_at_k(ranked_pages: list[str], gold_pages: list[str], k: int) -> float:
    """Share of the correct pages that appear in the first k results."""
    if not gold_pages:
        return 0.0
    found = set(ranked_pages[:k]) & set(gold_pages)
    return len(found) / len(gold_pages)


def reciprocal_rank(ranked_pages: list[str], gold_pages: list[str], cutoff: int = 10) -> float:
    """1 divided by the position of the first correct page, or 0 if not in the top `cutoff`."""
    for position, page in enumerate(ranked_pages[:cutoff], start=1):
        if page in gold_pages:
            return 1.0 / position
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    """Simple percentile with linear steps between neighbours. pct is 0 to 100."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
