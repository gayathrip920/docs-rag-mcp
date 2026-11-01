from rag import answer, llm
from rag.retrieve import Hit


def hits(n=3):
    return [
        Hit(i, f"page{i}.md", f"Page {i} > Part", f"https://x/{i}/", f"text {i}")
        for i in range(1, n + 1)
    ]


class FakeConn:
    def __init__(self, fail=False):
        self.rows = []
        self.fail = fail
        self.rolled_back = False

    def execute(self, sql, params=()):
        if self.fail:
            raise RuntimeError("db down")
        self.rows.append(params)

    def commit(self):
        pass

    def rollback(self):
        self.rolled_back = True


def reply(text, cost=0.001):
    return llm.LLMResult(text, "m", 10, 5, cost, 12)


def test_messages_number_the_sources_and_include_the_question():
    messages = answer.build_messages("How?", hits(2))
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "[1] https://x/1/" in user
    assert "[2] https://x/2/" in user
    assert user.endswith("Question: How?")
    assert "exactly: Not found in the docs." in messages[0]["content"]


def test_parse_citations_handles_groups_and_ranges():
    valid, invalid = answer.parse_citations("A [1] and B [2, 3] and C [1][9].", 3)
    assert valid == [1, 2, 3]
    assert invalid == [9]


def test_not_found_variants_are_recognised():
    assert answer.is_not_found("Not found in the docs.")
    assert answer.is_not_found("  not found in the docs  ")
    assert not answer.is_not_found("It was not found in the docs I read.")


def test_good_answer_marks_cited_sources():
    result = answer.finalize("Use Depends [1]. Also see [3].", hits(3), reply("x"))
    assert result.found is True
    assert result.citations_valid is True
    assert [s["cited"] for s in result.sources] == [True, False, True]
    assert result.cost_usd == 0.001


def test_made_up_citation_is_removed_and_flagged():
    result = answer.finalize("Do this [1] and that [7].", hits(3))
    assert result.citations_valid is False
    assert "[7]" not in result.answer
    assert "[1]" in result.answer
    assert "that." in result.answer


def test_answer_without_any_citation_is_flagged():
    result = answer.finalize("Just trust me.", hits(3))
    assert result.found is True
    assert result.citations_valid is False


def test_not_found_has_nothing_to_cite():
    result = answer.finalize("Not found in the docs.", hits(3))
    assert result.found is False
    assert result.citations_valid is None
    assert not any(s["cited"] for s in result.sources)


def test_no_hits_skips_the_model_and_still_logs(monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(
        llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call"))
    )
    conn = FakeConn()
    result = answer.answer_question(conn, "What is Paris?", api_key="k")
    assert result.answer == answer.NOT_FOUND
    assert len(conn.rows) == 1
    assert conn.rows[0][0] == "What is Paris?"
    assert conn.rows[0][4] is False


def test_answer_question_uses_the_key_passed_in_and_logs_without_it(monkeypatch):
    seen = {}

    def fake_chat(messages, model, api_key, **kwargs):
        seen["key"], seen["model"] = api_key, model
        return reply("Use it [2].", cost=0.5)

    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: hits(3))
    monkeypatch.setattr(llm, "chat", fake_chat)
    conn = FakeConn()
    result = answer.answer_question(
        conn, "How?", api_key="visitor-key", model="some/model", source="mcp"
    )
    assert seen == {"key": "visitor-key", "model": "some/model"}
    assert result.found and result.citations_valid
    question, source, _latency, cost, found = conn.rows[0]
    assert (question, source, cost, found) == ("How?", "mcp", 0.5, True)
    assert "visitor-key" not in repr(conn.rows)


def test_logging_failure_does_not_break_the_answer(monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: hits(1))
    monkeypatch.setattr(llm, "chat", lambda *a, **k: reply("Yes [1]."))
    conn = FakeConn(fail=True)
    result = answer.answer_question(conn, "How?", api_key="k")
    assert result.answer == "Yes [1]."
    assert conn.rolled_back


def test_stream_sends_sources_then_tokens_then_checked_done(monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: hits(2))

    def fake_stream(*a, **k):
        yield "Use "
        yield "it [1] [5]."
        yield reply("Use it [1] [5].")

    monkeypatch.setattr(llm, "stream_chat", fake_stream)
    events = list(answer.answer_stream(FakeConn(), "How?", api_key="k"))
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    done = events[-1]
    assert done["citations_valid"] is False
    assert "[5]" not in done["answer"]
    assert done["sources"][0]["cited"] is True
