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

## License

MIT. See [LICENSE](LICENSE). The FastAPI docs in `data/` keep their own MIT license and credit, in [data/LICENSE-fastapi](data/LICENSE-fastapi).
