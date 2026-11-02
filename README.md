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

## Running the API

```bash
pip install -r requirements.txt -r requirements-ml.txt
uvicorn api.main:app --port 8000
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Says the server is up. Does not touch the database. |
| `POST /search` | Returns the best chunks for a query. No API key needed. |
| `POST /ask` | Returns a short answer with citations. Needs your OpenRouter key in the `X-OpenRouter-Key` header. |
| `POST /ask/stream` | Same as `/ask` but streams the answer as it is written. |
| `GET /pages/{path}` | Returns the full text of one doc page. |
| `GET /metrics` | Question count, found rate, latency p50 and p95, and average cost. |

The key is used for that one request only. It is not stored or logged, and the server has no key of its own to fall back on. The key is checked before any searching starts, so requests without one cost almost nothing. Limits and settings are listed below.

## Web page

The page is a small Next.js app in `frontend/`. It sends requests to `/api/...`, and Next forwards them to the Python API, so the browser never talks to the API directly.

```bash
cd frontend
npm install
API_URL=http://localhost:8000 npm run dev    # then open http://localhost:3000
```

- **Search** works for everyone and needs no key. Results link back to the exact doc section.
- **Written answers** appear only when the API has `ENABLE_ASK=true`. The page then shows a box for your own OpenRouter key. The key stays in your browser tab (session storage) and is sent only with your question. It is not saved on the server.
- In the public search-only setup, the key box and the Ask button are hidden.
- Answers are shown as plain text, never as raw HTML. Links are only shown for https addresses.
- `/metrics` shows question count, found rate, latency and cost for answered questions.

`API_URL` is read when the app is built, so set it before `npm run build`.

## Use it as an MCP server

The MCP server lets any MCP-compatible tool search these docs. It runs on your own computer and talks to the search API, so you need the API running first (or a hosted one, set with `DOCS_API_URL`).

```bash
pip install -r requirements-mcp.txt
python -m mcp_server.server        # speaks MCP over stdio
```

Add it to your MCP client's settings. The exact file differs by client, but the entry looks like this:

```json
{
  "mcpServers": {
    "docs-rag": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/docs-rag-mcp",
      "env": {
        "DOCS_API_URL": "http://localhost:8000",
        "OPENROUTER_API_KEY": "optional, only for ask_docs"
      }
    }
  }
}
```

| Tool | What it does | Needs a key? |
|---|---|---|
| `search_docs(query, k)` | Best matching doc sections with links. `k` is 1 to 20. | No |
| `get_page(path)` | Full text of one page, for example `tutorial/cors.md`. Long pages are cut at 20,000 characters. | No |
| `ask_docs(question)` | Short answer with numbered citations, or "Not found in the docs." | Yes, your own `OPENROUTER_API_KEY` |

`ask_docs` is only offered when `OPENROUTER_API_KEY` is set. It uses your key on your computer. The search API never sees it.

Safety notes:

- All tools are read-only. None can write to a database or read your files.
- Inputs are checked: query length, `k`, and page paths (no `..`).
- Results are plain documentation text. Treat them as data, not instructions.
- Errors show short fixed messages, and the key is never included.

## Run with Docker

```bash
cp .env.example .env          # put your DATABASE_URL in .env
docker compose up --build     # page: http://localhost:3000   API: http://localhost:8000
```

The database stays on Neon. To use a different env file, run `ENV_FILE=.env.reader docker compose up`. To run the public search-only setup, add `ENABLE_ASK=false`, `SEARCH_MODES=vector` and `DEFAULT_SEARCH_MODE=vector` to that file.

Notes on the images:

- The API image uses the CPU build of torch and bakes in both models, so a starting container does not need the internet.
- Both images run as a non-root user.
- Secrets are never copied into an image. `.env` files are excluded at build time.

## Run on Kubernetes (local, with kind)

You need `docker`, `kind` and `kubectl`.

```bash
make k8s-up      # makes a local cluster, builds the images, deploys, waits until ready
kubectl -n docs-rag get pods
kubectl -n docs-rag port-forward svc/web 3000:3000    # then open http://localhost:3000
make k8s-down    # removes the cluster
```

What is in `deploy/k8s/`:

- `configmap.yaml` holds the public settings (search-only mode and the limits).
- `secret.example.yaml` is an example only. `make k8s-up` creates the real Secret from your local git-ignored file (`.env.reader` by default, change it with `SECRET_ENV_FILE`). Use the read-only database login for it.
- The API and the page each have a Deployment and a Service. Both have readiness and liveness probes. The API also has a startup probe. Containers run as non-root with a read-only filesystem, no extra permissions, and CPU and memory limits.
- The page reaches the API at `http://api:8000`, so keep that Service name.

The manifests are checked against the Kubernetes schemas in CI (`kubeconform`).

## Public search-only mode

For a site anyone can visit, run the API with answers turned off. No AI key is involved, and the slow re-ranking model is never loaded, so it is cheap and hard to abuse.

```bash
ENABLE_ASK=false SEARCH_MODES=vector DEFAULT_SEARCH_MODE=vector RATE_LIMIT_PER_MIN=30 GLOBAL_RATE_LIMIT_PER_MIN=200 MAX_CONCURRENT_SEARCHES=2 uvicorn api.main:app --port 8000
```

| Setting | What it does | Default |
|---|---|---|
| `ENABLE_ASK` | Turns `/ask` and `/ask/stream` on or off | `true` |
| `SEARCH_MODES` | Which search methods visitors may use (`vector`, `hybrid`, `hybrid_rerank`) | all three |
| `DEFAULT_SEARCH_MODE` | Method used when a request does not name one | `hybrid_rerank` |
| `RATE_LIMIT_PER_MIN` | Calls per minute for each visitor | `30` |
| `GLOBAL_RATE_LIMIT_PER_MIN` | Calls per minute for all visitors together (0 means off) | `0` |
| `MAX_CONCURRENT_SEARCHES` | Searches allowed at the same time. Extra ones get a polite "busy" reply | `4` |
| `POOL_MAX` | Most database connections kept open | `5` |
| `CORS_ORIGINS` | Websites allowed to call the API from a browser | `http://localhost:3000` |

`GET /config` tells the web page what is turned on. In public mode, a quick test on a laptop used about 470 MB of memory.

Behind a proxy or ingress, every visitor can look like the same address. Start uvicorn with `FORWARDED_ALLOW_IPS` set to your proxy's address so the per-visitor limit sees the real visitor.

### Use a read-only database login

The API only needs to read pages and write the query log. `deploy/sql/readonly_role.sql` creates a login for that. Run it once as the owner with a strong password, then use that login in the API's `DATABASE_URL`. Keep the owner login for `rag.ingest` only.

## Test questions

`evals/questions.jsonl` has 60 questions, each with the doc page that answers it. `evals/out_of_scope.jsonl` has 10 questions the docs cannot answer, used to check that the app says "not found". Run `python evals/check_questions.py` to confirm every listed page exists.

## License

MIT. See [LICENSE](LICENSE). The FastAPI docs in `data/` keep their own MIT license and credit, in [data/LICENSE-fastapi](data/LICENSE-fastapi).
