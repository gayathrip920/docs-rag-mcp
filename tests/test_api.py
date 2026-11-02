import json

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

    def close(self):
        self.closed = True


@pytest.fixture
def client(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(main, "connect", lambda: conn)
    monkeypatch.setattr(main, "limiter", main.RateLimiter(0))
    test_client = TestClient(main.app)
    test_client.conn = conn
    return test_client


def hits():
    return [Hit(1, "a.md", "A > Part", "https://x/a/", "text a", 0.9)]


def test_health_does_not_need_the_database(monkeypatch):
    def boom():
        raise AssertionError("health must not connect")

    monkeypatch.setattr(main, "connect", boom)
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_results_without_any_key(client, monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda conn, q, mode, limit: hits())
    response = client.post("/search", json={"query": "upload a file", "k": 3})
    assert response.status_code == 200
    first = response.json()["results"][0]
    assert first["page_path"] == "a.md"
    assert first["url"] == "https://x/a/"
    assert client.conn.closed


def test_search_rejects_bad_input(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "k": 99}).status_code == 422
    assert client.post("/search", json={"query": "x", "mode": "magic"}).status_code == 422
    assert client.post("/search", json={"query": "x" * 501}).status_code == 422


def test_ask_passes_the_visitor_key_through_and_never_echoes_it(client, monkeypatch):
    seen = {}

    def fake_ask(conn, question, api_key, model, k):
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


def test_ask_without_a_key_is_a_clear_401(client, monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: hits())
    response = client.post("/ask", json={"question": "How?"})
    assert response.status_code == 401
    assert "API key" in response.json()["detail"]


def test_ask_maps_provider_errors_to_friendly_messages(client, monkeypatch):
    def fake_ask(*args, **kwargs):
        raise llm.LLMError("Your OpenRouter account has no credits left.", 402)

    monkeypatch.setattr(main.answer_module, "answer_question", fake_ask)
    response = client.post("/ask", json={"question": "How?"}, headers={"X-OpenRouter-Key": SECRET})
    assert response.status_code == 402
    assert SECRET not in response.text


def test_ask_rejects_odd_model_names(client):
    response = client.post("/ask", json={"question": "How?", "model": "bad name; drop"})
    assert response.status_code == 422


def test_stream_sends_events_in_order(client, monkeypatch):
    def fake_stream(conn, question, api_key, model, k):
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": "Hi"}
        yield {"type": "done", "answer": "Hi", "found": True}

    monkeypatch.setattr(main.answer_module, "answer_stream", fake_stream)
    response = client.post(
        "/ask/stream", json={"question": "How?"}, headers={"X-OpenRouter-Key": "k"}
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")
    ]
    assert [e["type"] for e in events] == ["sources", "token", "done"]


def test_stream_turns_errors_into_an_error_event(client, monkeypatch):
    def fake_stream(*args, **kwargs):
        raise llm.LLMError("OpenRouter rejected the API key.", 401)
        yield

    monkeypatch.setattr(main.answer_module, "answer_stream", fake_stream)
    response = client.post(
        "/ask/stream", json={"question": "How?"}, headers={"X-OpenRouter-Key": SECRET}
    )
    assert response.status_code == 200
    event = json.loads(response.text.split("data: ", 1)[1].split("\n\n")[0])
    assert event == {"type": "error", "message": "OpenRouter rejected the API key."}
    assert SECRET not in response.text


def test_stream_hides_unexpected_error_details(client, monkeypatch):
    def fake_stream(*args, **kwargs):
        raise RuntimeError("secret internal detail")
        yield

    monkeypatch.setattr(main.answer_module, "answer_stream", fake_stream)
    response = client.post("/ask/stream", json={"question": "How?"})
    assert "secret internal detail" not in response.text
    assert "went wrong" in response.text


def test_page_found_and_not_found(monkeypatch):
    conn = FakeConn(("a.md", "Title", "https://x/a/", "body text"))
    monkeypatch.setattr(main, "connect", lambda: conn)
    monkeypatch.setattr(main, "limiter", main.RateLimiter(0))
    client = TestClient(main.app)
    assert client.get("/pages/a.md").json()["title"] == "Title"
    conn.row = None
    assert client.get("/pages/missing.md").status_code == 404


def test_page_rejects_path_tricks(client):
    assert client.get("/pages/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/pages/a..b").status_code == 404


def test_metrics_with_data_and_when_empty(monkeypatch):
    monkeypatch.setattr(main, "limiter", main.RateLimiter(0))
    conn = FakeConn((12, 0.75, 1200.0, 3400.0, 0.000321))
    monkeypatch.setattr(main, "connect", lambda: conn)
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


def test_rate_limit_blocks_after_the_limit(monkeypatch):
    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "limiter", main.RateLimiter(2))
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: [])
    client = TestClient(main.app)
    codes = [client.post("/search", json={"query": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
