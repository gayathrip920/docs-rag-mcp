"""A tiny client for the docs search API. The MCP server uses it to find chunks and pages."""

import os
import re

import httpx

PAGE_PATH_RE = re.compile(r"[A-Za-z0-9_\-./]{1,200}")


class DocsAPIError(Exception):
    """A problem reaching the docs API. The message is safe to show."""


class DocsAPI:
    def __init__(self, base_url: str | None = None, client: httpx.Client | None = None):
        self.base_url = (
            base_url or os.environ.get("DOCS_API_URL", "http://localhost:8000")
        ).rstrip("/")
        self.client = client or httpx.Client(timeout=30)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.client.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise DocsAPIError(
                "Could not reach the docs search server. Is DOCS_API_URL correct?"
            ) from exc
        if response.status_code >= 400:
            # The API's own error texts are short and fixed, so they are safe to pass on.
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            message = detail if isinstance(detail, str) else f"status {response.status_code}"
            raise DocsAPIError(f"The docs search server said: {message[:200]}")
        return response.json()

    def search(self, query: str, k: int = 5) -> list[dict]:
        data = self._request("POST", "/search", json={"query": query, "k": k})
        return data["results"]

    def get_page(self, path: str) -> dict:
        if not PAGE_PATH_RE.fullmatch(path) or ".." in path:
            raise DocsAPIError("That page path does not look right.")
        return self._request("GET", f"/pages/{path}")
