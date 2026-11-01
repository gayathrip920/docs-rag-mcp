"""Cut a markdown page into chunks that keep their page title and heading.

Sizes are counted in words. About 300 words is roughly 400 tokens, which sits
inside the 300 to 500 token goal.
"""

import re
from dataclasses import dataclass

MAX_WORDS = 300
OVERLAP_WORDS = 40
MIN_WORDS = 8

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$")
ANCHOR_RE = re.compile(r"\s*\{\s*#([A-Za-z0-9_\-]+)\s*\}\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
# Lines like {* ../../docs_src/app.py hl[1] *} point to code files that are not in the snapshot.
INCLUDE_RE = re.compile(r"^\s*\{[*!][^}]*[*!]\}\s*$")
EMPTY_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n\s*(```|~~~)[ \t]*$", re.MULTILINE)
# Docs-site markers such as "/// tip | Title" and "////" for tabs.
MARKER_RE = re.compile(r"^\s*/{3,4}\s*(.*)$")


@dataclass
class Chunk:
    page_path: str
    chunk_index: int
    heading: str
    url: str
    content: str


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_]+", "-", text).strip("-")


def page_url(base_url: str, page_path: str) -> str:
    route = page_path[:-3] if page_path.endswith(".md") else page_path
    if route == "index":
        return base_url + "/"
    if route.endswith("/index"):
        route = route[: -len("/index")]
    return f"{base_url}/{route}/"


def strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :].lstrip("\n")
    return text


def clean_lines(text: str) -> list[str]:
    """Drop docs-site markers and code include lines, keep the real text."""
    lines = []
    in_code = False
    for line in text.splitlines():
        if INCLUDE_RE.match(line):
            continue
        if FENCE_RE.match(line):
            in_code = not in_code
            lines.append(line)
            continue
        if in_code:
            lines.append(line)
            continue
        marker = MARKER_RE.match(line)
        if marker:
            label = marker.group(1)
            if "|" in label:
                lines.append(label.split("|", 1)[1].strip())
            continue
        lines.append(line)
    return lines


def split_sections(lines: list[str]) -> list[tuple[list[tuple[str, str]], list[str]]]:
    """Return (heading path, body lines) pairs. A path item is (title, anchor)."""
    sections = []
    stack: list[tuple[int, str, str]] = []  # level, title, anchor
    body: list[str] = []
    in_code = False

    def flush():
        path = [(title, anchor) for _, title, anchor in stack]
        sections.append((path, body.copy()))
        body.clear()

    for line in lines:
        if FENCE_RE.match(line):
            in_code = not in_code
        match = None if in_code else HEADING_RE.match(line)
        if not match:
            body.append(line)
            continue
        flush()
        level = len(match.group(1))
        raw = match.group(2)
        anchor_match = ANCHOR_RE.search(raw)
        anchor = anchor_match.group(1) if anchor_match else ""
        title = ANCHOR_RE.sub("", raw).strip()
        if not anchor:
            anchor = slugify(title)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title, anchor))
    flush()
    return sections


def split_blocks(lines: list[str]) -> list[str]:
    """Split body lines into paragraphs. A code fence stays in one block."""
    blocks, current = [], []
    in_code = False
    for line in lines:
        if FENCE_RE.match(line):
            in_code = not in_code
        if not in_code and not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def shrink_block(block: str, max_words: int) -> list[str]:
    """Break a block that is too long into pieces by lines, then by words."""
    pieces, current, count = [], [], 0
    for line in block.split("\n"):
        words = len(line.split())
        if words > max_words:
            if current:
                pieces.append("\n".join(current))
                current, count = [], 0
            tokens = line.split()
            for i in range(0, len(tokens), max_words):
                pieces.append(" ".join(tokens[i : i + max_words]))
            continue
        if count + words > max_words and current:
            pieces.append("\n".join(current))
            current, count = [], 0
        current.append(line)
        count += words
    if current:
        pieces.append("\n".join(current))
    return pieces


def pack_blocks(blocks: list[str], max_words: int, overlap_words: int) -> list[str]:
    """Group paragraphs into texts of up to max_words, with a small overlap."""
    flat = []
    for block in blocks:
        if len(block.split()) > max_words:
            flat.extend(shrink_block(block, max_words))
        else:
            flat.append(block)

    texts, current, count = [], [], 0
    for block in flat:
        words = len(block.split())
        if current and count + words > max_words:
            text = "\n\n".join(current)
            texts.append(text)
            tail = text.split()[-overlap_words:] if overlap_words else []
            current = [" ".join(tail)] if tail else []
            count = len(tail)
        current.append(block)
        count += words
    if current:
        texts.append("\n\n".join(current))
    return texts


def chunk_page(text: str, page_path: str, base_url: str) -> tuple[str, list[Chunk]]:
    """Return (page title, chunks) for one markdown page."""
    lines = clean_lines(strip_front_matter(text))
    # A code block whose only content was an include line is now empty, so remove it.
    lines = EMPTY_FENCE_RE.sub("", "\n".join(lines)).splitlines()
    sections = split_sections(lines)

    title = page_path
    for path, _ in sections:
        if path:
            title = path[0][0]
            break

    base_page_url = page_url(base_url, page_path)
    chunks: list[Chunk] = []
    for path, body in sections:
        blocks = split_blocks(body)
        if sum(len(b.split()) for b in blocks) < MIN_WORDS:
            continue
        heading = " > ".join(name for name, _ in path) or title
        # The path already starts with the page title, so it works as the prefix.
        prefix = heading
        anchor = path[-1][1] if len(path) > 1 else ""
        url = f"{base_page_url}#{anchor}" if anchor else base_page_url
        for piece in pack_blocks(blocks, MAX_WORDS, OVERLAP_WORDS):
            chunks.append(
                Chunk(
                    page_path=page_path,
                    chunk_index=len(chunks),
                    heading=heading,
                    url=url,
                    content=f"{prefix}\n\n{piece}",
                )
            )
    return title, chunks
