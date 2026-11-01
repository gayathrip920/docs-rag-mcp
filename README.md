# docs-rag-mcp

Ask questions about a public documentation set and get short answers with links back to the source pages.

You can use it two ways:

- a chat web app
- an MCP server, so any MCP-compatible tool can search the docs for you

This project is being built in small steps. Right now only the repo foundation is in place. The code, results tables and setup guide will be added as each piece lands.

## Bring your own API key

The web app never uses a key stored on the server. You type your own OpenRouter key into the page, and it is used only to answer your question. It is not saved in the database or written to logs.

## Data

The search runs over a pinned snapshot of the FastAPI docs. See [data/SOURCE.md](data/SOURCE.md) for the exact commit and credit.

## Loading the docs into the database

```bash
pip install -r requirements.txt -r requirements-ml.txt
cp .env.example .env     # then put your own DATABASE_URL in .env
python -m rag.ingest
```

The database is Postgres with the pgvector extension, for example a free Neon project. Each page is cut into chunks that keep the page title and heading. Running the command again is safe: unchanged pages are skipped.

To see how the docs get chunked without a database or model, run `python -m rag.ingest --dry-run`.

## Test questions

`evals/questions.jsonl` has 60 questions, each with the doc page that answers it. `evals/out_of_scope.jsonl` has 10 questions the docs cannot answer, used to check that the app says "not found". Run `python evals/check_questions.py` to confirm every listed page exists.

## License

MIT. See [LICENSE](LICENSE). The FastAPI docs in `data/` keep their own MIT license and credit, in [data/LICENSE-fastapi](data/LICENSE-fastapi).
