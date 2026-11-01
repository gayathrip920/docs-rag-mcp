"""Read the docs, cut them into chunks, embed them, and store them in the database.

Run it from the repo root:

    python -m rag.ingest              # real run, needs DATABASE_URL
    python -m rag.ingest --dry-run    # only chunk the files and print stats

Running it twice is safe. Pages whose text did not change are skipped, and
pages that were removed from the folder are removed from the database.
"""

import argparse
import hashlib
import sys
from pathlib import Path

from rag.chunking import chunk_page, page_url
from rag.config import load_settings


def read_pages(data_dir: Path) -> dict[str, str]:
    pages = {}
    for path in sorted(data_dir.rglob("*.md")):
        key = path.relative_to(data_dir).as_posix()
        pages[key] = path.read_text(encoding="utf-8")
    return pages


def page_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dry_run(pages: dict[str, str], base_url: str) -> None:
    sizes = []
    for path, text in pages.items():
        _, chunks = chunk_page(text, path, base_url)
        sizes.extend(len(c.content.split()) for c in chunks)
    sizes.sort()
    print(f"pages: {len(pages)}")
    print(f"chunks: {len(sizes)}")
    if sizes:
        print(f"words per chunk: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")


def ingest(pages: dict[str, str], base_url: str) -> None:
    # These imports need the database driver and the embedding model.
    from rag.db import connect, init_schema
    from rag.embed import embed_texts

    conn = connect()
    init_schema(conn)

    existing = dict(conn.execute("SELECT path, page_hash FROM pages").fetchall())
    added = skipped = 0

    for path, text in pages.items():
        digest = page_hash(text)
        if existing.get(path) == digest:
            skipped += 1
            continue
        title, chunks = chunk_page(text, path, base_url)
        if not chunks:
            continue
        vectors = embed_texts([c.content for c in chunks])
        with conn.transaction():
            # Deleting the page also deletes its old chunks (ON DELETE CASCADE).
            conn.execute("DELETE FROM pages WHERE path = %s", (path,))
            conn.execute(
                "INSERT INTO pages (path, title, url, body, page_hash) VALUES (%s, %s, %s, %s, %s)",
                (path, title, page_url(base_url, path), text, digest),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (page_path, chunk_index, heading, url, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (c.page_path, c.chunk_index, c.heading, c.url, c.content, vec)
                        for c, vec in zip(chunks, vectors, strict=True)
                    ],
                )
        added += 1
        print(f"stored {path} ({len(chunks)} chunks)")

    removed = [p for p in existing if p not in pages]
    for path in removed:
        conn.execute("DELETE FROM pages WHERE path = %s", (path,))
    conn.commit()

    print(f"done: {added} pages stored, {skipped} unchanged, {len(removed)} removed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="chunk only, no database or model")
    args = parser.parse_args()

    settings = load_settings()
    pages = read_pages(settings.data_dir)
    if not pages:
        print(f"No markdown files found in {settings.data_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        dry_run(pages, settings.docs_base_url)
    else:
        ingest(pages, settings.docs_base_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
