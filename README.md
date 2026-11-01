# docs-rag-mcp

[![CI](https://github.com/gayathrip920/docs-rag-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/gayathrip920/docs-rag-mcp/actions/workflows/ci.yml)

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

## Search results

Measured on 2026-11-01 with 60 questions (see `evals/questions.jsonl`). Each question has one correct doc page. The search returns chunks, which are turned into a list of pages.

| Method | recall@5 | MRR@10 | Search time (median / p95) |
|---|---|---|---|
| Vector only | 0.950 | 0.821 | 44 ms / 115 ms |
| Hybrid (vector + keyword) | 0.950 | 0.801 | 140 ms / 214 ms |
| Hybrid + rerank | 0.983 | 0.828 | 3.0 s / 3.6 s |

- recall@5: the share of correct pages found in the top 5 pages.
- MRR@10: the average of 1 divided by the position of the correct page (0 if it is not in the top 10).
- Models: `all-MiniLM-L6-v2` for embeddings and `ms-marco-MiniLM-L-6-v2` for re-ranking. Chunks are up to 300 words with 40 words of overlap.
- Times are for the search step only, on a CPU inside a container. The re-ranker is the slow part.

Things to keep in mind:

- With 60 questions, one question is worth 1.7 points of recall, so small gaps are not strong evidence.
- Adding keyword search did not help here. It matched vector search on recall@5 and was slightly lower on MRR.
- Re-ranking found the correct page for 59 of 60 questions. The one miss is "How do I create my first app and start the development server?", where other pages ranked above the first steps page.
- Full per-question misses are in `evals/results/retrieval.json`. To repeat the run: `python -m evals.run_retrieval_eval`.

## How answers work

1. The question goes through the best search method (hybrid plus re-ranking) and the top 5 chunks are picked.
2. The chunks are shown to the model as numbered sources, with rules: use only the sources, cite them like [1], and reply "Not found in the docs." if the answer is not there.
3. The reply is checked. Citation numbers that do not match a real source are removed and flagged. If nothing is retrieved at all, the model is not called.
4. The time, cost and whether an answer was found are saved in the `queries` table. The question text is saved too, but the API key never is.

The code is in `rag/answer.py` and `rag/llm.py`. The model is reached through OpenRouter, and the key is passed in with each request. It is not read from the server for the web app.

The answer logic is covered by tests that use a fake model. It has not yet been tried against the real OpenRouter service, so no answer quality numbers exist yet.

## Test questions

`evals/questions.jsonl` has 60 questions, each with the doc page that answers it. `evals/out_of_scope.jsonl` has 10 questions the docs cannot answer, used to check that the app says "not found". Run `python evals/check_questions.py` to confirm every listed page exists.

## License

MIT. See [LICENSE](LICENSE). The FastAPI docs in `data/` keep their own MIT license and credit, in [data/LICENSE-fastapi](data/LICENSE-fastapi).
