import json
import threading
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from api import main
from rag import answer, llm
from rag.retrieve import Hit

SECRET = "sk-or-visitor-secret"


class FakeConn:
    def __init__(self, row=None):
        self.row = row
        self.closed = False

    def execute(self, sql, params=()):
        return self

    def fetchone(self):
        return self.row


def use_conn(monkeypatch, conn):
    @contextmanager
    def session():
        try:
            yield conn
        finally:
            conn.closed = True

    monkeypatch.setattr(main, "db_session", session)


def no_limits(monkeypatch):
    monkeypatch.setattr(main, "limiter", main.RateLimiter(0))
    monkeypatch.setattr(main, "global_limiter", main.RateLimiter(0))


@pytest.fixture
def client(monkeypatch):
    conn = FakeConn()
    use_conn(monkeypatch, conn)
    no_limits(monkeypatch)
    test_client = TestClient(main.app)
    test_client.conn = conn
    return test_client


def hits():
    return [Hit(1, "a.md", "A > Part", "https://x/a/", "text a", 0.9)]


def test_health_does_not_need_the_database(monkeypatch):
    def boom():
        raise AssertionError("health must not connect")

    monkeypatch.setattr(main, "db_session", boom)
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_tells_the_page_what_is_available(client):
    data = client.get("/config").json()
    assert data["ask_enabled"] is True
    assert "vector" in data["search_modes"]


def test_search_returns_results_without_any_key(client, monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda conn, q, mode, limit: hits())
    response = client.post("/search", json={"query": "upload a file", "k": 3})
    assert response.status_code == 200
    first = response.json()["results"][0]
    assert first["page_path"] == "a.md"
    assert first["url"] == "https://x/a/"
    assert client.conn.closed


def test_search_uses_the_default_mode_when_none_is_given(client, monkeypatch):
    seen = {}

    def fake(conn, q, mode, limit):
        seen["mode"] = mode
        return []

    monkeypatch.setattr(main, "retrieve", fake)
    monkeypatch.setattr(main, "DEFAULT_SEARCH_MODE", "vector")
    client.post("/search", json={"query": "x"})
    assert seen["mode"] == "vector"


def test_public_mode_refuses_the_slow_reranker(client, monkeypatch):
    monkeypatch.setattr(main, "SEARCH_MODES", ("vector",))
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: pytest.fail("must not search"))
    response = client.post("/search", json={"query": "x", "mode": "hybrid_rerank"})
    assert response.status_code == 422


def test_search_rejects_bad_input(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "k": 99}).status_code == 422
    assert client.post("/search", json={"query": "x", "mode": "magic"}).status_code == 422
    assert client.post("/search", json={"query": "x" * 501}).status_code == 422


def test_ask_passes_the_visitor_key_through_and_never_echoes_it(client, monkeypatch):
    seen = {}

    def fake_ask(conn, question, api_key, model, k, mode):
        seen.update(key=api_key, model=model, k=k, question=question)
        return answer.AnswerResult("Use it [1].", True, True, [{"n": 1}], "m", 10, 0.01)

    monkeypatch.setattr(main.answer_module, "answer_question", fake_ask)
    response = client.post(
        "/ask",
        json={"question": "How?", "model": "openai/gpt-4o-mini"},
        headers={"X-OpenRouter-Key": SECRET},
    )
    assert response.status_code == 200
    assert seen == {"key": SECRET, "model": "openai/gpt-4o-mini", "k": 5, "question": "How?"}
    assert SECRET not in response.text
    assert response.json()["answer"] == "Use it [1]."


def test_ask_without_a_key_is_a_401_before_any_searching(client, monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: pytest.fail("must not search"))
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: pytest.fail("must not search"))
    response = client.post("/ask", json={"question": "How?"})
    assert response.status_code == 401
    assert "API key" in response.json()["detail"]
    assert client.post("/ask/stream", json={"question": "How?"}).status_code == 401


def test_ask_is_switched_off_in_search_only_mode(client, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_ASK", False)
    headers = {"X-OpenRouter-Key": SECRET}
    assert client.post("/ask", json={"question": "x"}, headers=headers).status_code == 403
    assert client.post("/ask/stream", json={"question": "x"}, headers=headers).status_code == 403
    assert client.get("/config").json()["ask_enabled"] is False


def test_ask_maps_provider_errors_to_friendly_messages(client, monkeypatch):
    def fake_ask(*args, **kwargs):
        raise llm.LLMError("no_credits")

    monkeypatch.setattr(main.answer_module, "answer_question", fake_ask)
    response = client.post("/ask", json={"question": "How?"}, headers={"X-OpenRouter-Key": SECRET})
    assert response.status_code == 402
    assert SECRET not in response.text


def test_ask_rejects_odd_model_names(client):
    response = client.post("/ask", json={"question": "How?", "model": "bad name; drop"})
    assert response.status_code == 422


def sse_events(text):
    return [json.loads(line[6:]) for line in text.split("\n\n") if line.startswith("data: ")]


def test_stream_sends_events_in_order(client, monkeypatch):
    def fake_stream(conn, question, api_key, model, k, mode):
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": "Hi"}
        yield {"type": "done", "answer": "Hi", "found": True}

    monkeypatch.setattr(main.answer_module, "answer_stream", fake_stream)
    response = client.post(
        "/ask/stream", json={"question": "How?"}, headers={"X-OpenRouter-Key": "k"}
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [e["type"] for e in sse_events(response.text)] == ["sources", "token", "done"]


def test_stream_turns_errors_into_an_error_event(client, monkeypatch):
    def fake_stream(*args, **kwargs):
        raise llm.LLMError("bad_key")
        yield

    monkeypatch.setattr(main.answer_module, "answer_stream", fake_stream)
    response = client.post(
        "/ask/stream", json={"question": "How?"}, headers={"X-OpenRouter-Key": SECRET}
    )
    assert response.status_code == 200
    assert sse_events(response.text)[0] == {
        "type": "error",
        "message": "OpenRouter rejected the API key.",
    }
    assert SECRET not in response.text


def test_stream_hides_unexpected_error_details(client, monkeypatch):
    def fake_stream(*args, **kwargs):
        raise RuntimeError("secret internal detail")
        yield

    monkeypatch.setattr(main.answer_module, "answer_stream", fake_stream)
    response = client.post(
        "/ask/stream", json={"question": "How?"}, headers={"X-OpenRouter-Key": "k"}
    )
    assert "secret internal detail" not in response.text
    assert "went wrong" in response.text


def test_page_found_and_not_found(monkeypatch):
    conn = FakeConn(("a.md", "Title", "https://x/a/", "body text"))
    use_conn(monkeypatch, conn)
    no_limits(monkeypatch)
    client = TestClient(main.app)
    assert client.get("/pages/a.md").json()["title"] == "Title"
    conn.row = None
    assert client.get("/pages/missing.md").status_code == 404


def test_page_rejects_path_tricks(client):
    assert client.get("/pages/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/pages/a..b").status_code == 404


def test_metrics_with_data_and_when_empty(monkeypatch):
    no_limits(monkeypatch)
    conn = FakeConn((12, 0.75, 1200.0, 3400.0, 0.000321))
    use_conn(monkeypatch, conn)
    data = TestClient(main.app).get("/metrics").json()
    assert data == {
        "questions": 12,
        "found_rate": 0.75,
        "latency_ms_p50": 1200.0,
        "latency_ms_p95": 3400.0,
        "avg_cost_usd": 0.000321,
    }
    conn.row = (0, None, None, None, None)
    empty = TestClient(main.app).get("/metrics").json()
    assert empty["questions"] == 0 and empty["found_rate"] is None


def test_per_visitor_rate_limit(monkeypatch):
    use_conn(monkeypatch, FakeConn())
    monkeypatch.setattr(main, "limiter", main.RateLimiter(2))
    monkeypatch.setattr(main, "global_limiter", main.RateLimiter(0))
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: [])
    client = TestClient(main.app)
    codes = [client.post("/search", json={"query": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_global_limit_protects_against_many_visitors(monkeypatch):
    use_conn(monkeypatch, FakeConn())
    monkeypatch.setattr(main, "limiter", main.RateLimiter(0))
    monkeypatch.setattr(main, "global_limiter", main.RateLimiter(2))
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: [])
    client = TestClient(main.app)
    codes = [client.post("/search", json={"query": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_busy_server_answers_503_instead_of_piling_up(monkeypatch, client):
    monkeypatch.setattr(main, "search_slots", threading.BoundedSemaphore(1))
    main.search_slots.acquire()  # pretend one search is already running
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: [])
    response = client.post("/search", json={"query": "x"})
    assert response.status_code == 503


def test_slot_is_given_back_after_each_search(monkeypatch, client):
    monkeypatch.setattr(main, "search_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: [])
    codes = [client.post("/search", json={"query": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 200]
