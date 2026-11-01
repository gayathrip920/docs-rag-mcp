from pathlib import Path

from rag.chunking import MAX_WORDS, OVERLAP_WORDS, chunk_page, page_url, strip_front_matter

BASE = "https://example.com"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def words(n, word="word"):
    return " ".join(f"{word}{i}" for i in range(n))


def test_page_url_for_normal_index_and_nested_index():
    assert page_url(BASE, "tutorial/first-steps.md") == f"{BASE}/tutorial/first-steps/"
    assert page_url(BASE, "index.md") == f"{BASE}/"
    assert page_url(BASE, "tutorial/security/index.md") == f"{BASE}/tutorial/security/"


def test_front_matter_is_removed():
    text = "---\ntitle: x\n---\n# Real\n\nbody"
    assert strip_front_matter(text).startswith("# Real")


def test_chunk_keeps_title_and_heading():
    text = f"# Query Params\n\n{words(20)}\n\n## Optional parameters\n\n{words(20)}\n"
    title, chunks = chunk_page(text, "tutorial/query-params.md", BASE)
    assert title == "Query Params"
    assert chunks[-1].content.startswith("Query Params > Optional parameters")
    assert chunks[-1].heading == "Query Params > Optional parameters"


def test_explicit_anchor_is_used_in_url_and_removed_from_title():
    text = f"# Page\n\n## Some thing {{ #custom-id }}\n\n{words(20)}\n"
    _, chunks = chunk_page(text, "a.md", BASE)
    assert chunks[0].url == f"{BASE}/a/#custom-id"
    assert "{" not in chunks[0].heading


def test_heading_inside_code_block_is_not_a_heading():
    text = f"# Page\n\n```bash\n# just a comment\necho hi\n```\n\n{words(20)}\n"
    _, chunks = chunk_page(text, "a.md", BASE)
    assert len(chunks) == 1
    assert "# just a comment" in chunks[0].content


def test_code_include_lines_and_markers_are_dropped():
    text = (
        "# Page\n\n{* ../../docs_src/app.py hl[1] *}\n\n"
        f"/// tip | Handy trick\n\n{words(20)}\n\n///\n"
    )
    _, chunks = chunk_page(text, "a.md", BASE)
    body = chunks[0].content
    assert "docs_src" not in body
    assert "///" not in body
    assert "Handy trick" in body


def test_tiny_sections_are_skipped():
    text = f"# Page\n\n## Empty\n\n## Real\n\n{words(30)}\n"
    _, chunks = chunk_page(text, "a.md", BASE)
    assert [c.heading for c in chunks] == ["Page > Real"]


def test_long_section_is_split_with_overlap_and_size_limit():
    paragraphs = "\n\n".join(words(100, f"p{i}x") for i in range(10))
    text = f"# Page\n\n{paragraphs}\n"
    _, chunks = chunk_page(text, "a.md", BASE)
    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.content.split("\n\n", 1)[1]
        assert len(body.split()) <= MAX_WORDS + OVERLAP_WORDS
    first_tail = chunks[0].content.split()[-OVERLAP_WORDS:]
    assert " ".join(first_tail) in chunks[1].content


def test_one_huge_paragraph_is_still_split():
    text = f"# Page\n\n{words(1000)}\n"
    _, chunks = chunk_page(text, "a.md", BASE)
    assert len(chunks) >= 3


def test_chunk_indexes_are_in_order():
    paragraphs = "\n\n".join(words(100, f"p{i}x") for i in range(10))
    _, chunks = chunk_page(f"# Page\n\n{paragraphs}\n", "a.md", BASE)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_every_real_page_chunks_cleanly():
    pages = list(RAW.rglob("*.md"))
    assert len(pages) > 100
    for path in pages:
        rel = path.relative_to(RAW).as_posix()
        title, chunks = chunk_page(path.read_text(encoding="utf-8"), rel, BASE)
        assert title
        if not chunks:
            # Only tiny link pages (a title and one line) may produce nothing.
            assert len(path.read_text(encoding="utf-8").split()) < 20, rel
            continue
        for chunk in chunks:
            assert "docs_src" not in chunk.content, rel
            assert len(chunk.content.split()) <= MAX_WORDS + OVERLAP_WORDS + 40, rel
