import pytest

from evals.metrics import mean, percentile, recall_at_k, reciprocal_rank
from rag.retrieve import Hit, build_tsquery, rank_pages, reciprocal_rank_fusion, retrieve


def test_tsquery_uses_or_and_drops_repeats_and_symbols():
    assert build_tsquery("How do I upload a file? Upload!") == "how | do | upload | file"


def test_tsquery_splits_code_names():
    assert build_tsquery("max_length") == "max | length"


def test_tsquery_empty_when_nothing_usable():
    assert build_tsquery("?! a") is None


def test_fusion_puts_items_found_by_both_lists_first():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]])
    order = [item for item, _ in fused]
    assert order[:2] == [1, 3]
    assert set(order) == {1, 2, 3, 4}


def test_fusion_with_one_empty_list_keeps_the_other_order():
    fused = reciprocal_rank_fusion([[5, 6, 7], []])
    assert [item for item, _ in fused] == [5, 6, 7]


def test_rank_pages_removes_repeats_but_keeps_order():
    hits = [Hit(i, p, "h", "u", "c") for i, p in enumerate(["a.md", "b.md", "a.md", "c.md"])]
    assert rank_pages(hits) == ["a.md", "b.md", "c.md"]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        retrieve(None, "question", mode="magic")


def test_recall_counts_share_of_gold_pages_found():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert recall_at_k(["a"], [], 5) == 0.0


def test_reciprocal_rank_uses_first_correct_position():
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_reciprocal_rank_respects_cutoff():
    pages = [f"p{i}" for i in range(12)] + ["gold"]
    assert reciprocal_rank(pages, ["gold"], cutoff=10) == 0.0


def test_mean_and_percentile():
    assert mean([1, 2, 3]) == 2
    assert mean([]) == 0
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([5], 95) == 5
