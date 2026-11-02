import asyncio
import json

import httpx
import pytest

from mcp_server import server as srv
from mcp_server.client import DocsAPI, DocsAPIError
from rag import llm

SECRET = "sk-or-user-secret"


def api_with(handler) -> DocsAPI:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return DocsAPI("http://docs.test", client=client)


def result_row(n=1, text="some text"):
    return {
        "page_path": f"p{n}.md",
        "heading": f"Page {n} > Part",
        "url": f"https://x/{n}/",
        "content": text,
        "score": 0.5,
    }


def search_api(rows=None):
    rows = [result_row(1), result_row(2)] if rows is None else rows
    return api_with(lambda request: httpx.Response(200, json={"results": rows}))


def test_client_search_sends_query_and_k():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"results": []})

    api_with(handler).search("how to cors", 3)
    assert seen["body"] == {"query": "how to cors", "k": 3}
    assert seen["url"] == "http://docs.test/search"


def test_client_turns_server_errors_into_short_messages():
    api = api_with(lambda r: httpx.Response(429, json={"detail": "Too many requests."}))
    with pytest.raises(DocsAPIError, match="Too many requests"):
        api.search("x")


def test_client_handles_unreachable_server():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(DocsAPIError, match="Could not reach"):
        api_with(handler).search("x")


def test_client_rejects_bad_page_paths_without_any_request():
    api = api_with(lambda r: pytest.fail("no request should be sent"))
    for bad in ["../secret", "a..b", "a b", "a?x=1", "x" * 300]:
        with pytest.raises(DocsAPIError):
            api.get_page(bad)


def test_search_docs_shapes_and_trims_results():
    long_text = "word " * 1000
    out = srv.run_search(search_api([result_row(1, long_text)]), "cors", 5)
    assert set(out[0]) == {"page_path", "heading", "url", "text"}
    assert len(out[0]["text"]) == srv.MAX_SNIPPET_CHARS


@pytest.mark.parametrize("query", ["", "   ", "x" * 501])
def test_search_docs_rejects_bad_queries(query):
    with pytest.raises(srv.ToolError):
        srv.run_search(search_api(), query)


@pytest.mark.parametrize("k", [0, -1, 21, True, "5"])
def test_search_docs_rejects_bad_k(k):
    with pytest.raises(srv.ToolError):
        srv.run_search(search_api(), "cors", k)


def test_get_page_flags_truncation():
    page = {"path": "a.md", "title": "A", "url": "https://x/a/", "body": "y" * 30000}
    api = api_with(lambda r: httpx.Response(200, json=page))
    out = srv.run_get_page(api, "a.md")
    assert out["truncated"] is True
    assert len(out["text"]) == srv.MAX_PAGE_CHARS


def test_get_page_not_found_message():
    api = api_with(lambda r: httpx.Response(404, json={"detail": "Page not found."}))
    with pytest.raises(srv.ToolError, match="Page not found"):
        srv.run_get_page(api, "missing.md")


def test_ask_uses_the_callers_key_and_checks_citations(monkeypatch):
    seen = {}

    def fake_chat(messages, model, api_key, **kwargs):
        seen["key"] = api_key
        return llm.LLMResult("Use it [1] and [9].", "m", 5, 5, 0.01, 10)

    monkeypatch.setattr(llm, "chat", fake_chat)
    out = srv.run_ask(search_api(), "How?", SECRET)
    assert seen["key"] == SECRET
    assert out["found"] is True
    assert out["citations_valid"] is False  # [9] does not exist
    assert "[9]" not in out["answer"]
    assert SECRET not in json.dumps(out)


def test_ask_checks_the_key_before_searching():
    api = api_with(lambda r: pytest.fail("must not search without a key"))
    with pytest.raises(srv.ToolError, match="API key"):
        srv.run_ask(api, "How?", "")


def test_ask_provider_errors_become_fixed_messages(monkeypatch):
    def fake_chat(*args, **kwargs):
        raise llm.LLMError("bad_key")

    monkeypatch.setattr(llm, "chat", fake_chat)
    with pytest.raises(srv.ToolError) as caught:
        srv.run_ask(search_api(), "How?", SECRET)
    assert str(caught.value) == "OpenRouter rejected the API key."
    assert SECRET not in str(caught.value)


def test_ask_with_no_matches_does_not_call_the_model(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("no model call"))
    out = srv.run_ask(search_api([]), "How?", SECRET)
    assert out["answer"] == "Not found in the docs."
    assert out["found"] is False


def tool_names(server) -> set[str]:
    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_ask_tool_exists_only_when_a_key_is_given():
    assert tool_names(srv.build_server(search_api())) == {"search_docs", "get_page"}
    with_key = srv.build_server(search_api(), api_key=SECRET)
    assert tool_names(with_key) == {"search_docs", "get_page", "ask_docs"}


def test_a_tool_call_through_the_server_returns_results():
    server = srv.build_server(search_api())
    result = asyncio.run(server.call_tool("search_docs", {"query": "cors", "k": 2}))
    assert not result.is_error
    assert "p1.md" in json.dumps(result.model_dump(mode="json"))


def test_bad_input_through_the_server_shows_the_helpful_message():
    server = srv.build_server(search_api())
    with pytest.raises(srv.ToolError, match="query cannot be empty"):
        asyncio.run(server.call_tool("search_docs", {"query": ""}))
