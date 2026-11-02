import json

import httpx
import pytest

from rag import llm

MESSAGES = [{"role": "user", "content": "hi"}]
SECRET = "sk-or-test-secret-123"


def client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_chat_returns_text_cost_and_tokens():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "openai/test",
                "choices": [{"message": {"content": "  Hello [1]  "}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.0002},
            },
        )

    result = llm.chat(MESSAGES, "openai/test", SECRET, client=client_with(handler))
    assert result.text == "Hello [1]"
    assert result.cost_usd == 0.0002
    assert (result.prompt_tokens, result.completion_tokens) == (10, 4)
    assert seen["auth"] == f"Bearer {SECRET}"
    assert seen["body"]["usage"] == {"include": True}
    assert seen["body"]["temperature"] == 0


def test_chat_cost_is_none_when_not_reported():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    result = llm.chat(MESSAGES, "m", SECRET, client=client_with(handler))
    assert result.cost_usd is None


@pytest.mark.parametrize("status", [401, 402, 429, 400, 500])
def test_errors_are_friendly_and_never_contain_the_key(status):
    def handler(request):
        return httpx.Response(status, text=f"bad request, key was {SECRET}")

    with pytest.raises(llm.LLMError) as caught:
        llm.chat(MESSAGES, "m", SECRET, client=client_with(handler))
    assert SECRET not in str(caught.value)
    assert caught.value.status_code in (400, 401, 402, 429, 502)


def test_missing_or_odd_key_is_rejected_before_any_request():
    def handler(request):
        raise AssertionError("no request should be sent")

    for bad in ["", "   ", None, "has space", "line\nbreak"]:
        with pytest.raises(llm.LLMError) as caught:
            llm.chat(MESSAGES, "m", bad, client=client_with(handler))
        assert caught.value.status_code == 401


def test_network_failure_becomes_llm_error():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(llm.LLMError):
        llm.chat(MESSAGES, "m", SECRET, client=client_with(handler))


def test_weird_reply_shape_becomes_llm_error():
    def handler(request):
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(llm.LLMError):
        llm.chat(MESSAGES, "m", SECRET, client=client_with(handler))


def sse(*events: str) -> str:
    return "".join(f"data: {e}\n\n" for e in events)


def test_stream_yields_pieces_then_final_result_with_cost():
    body = ": OPENROUTER PROCESSING\n\n" + sse(
        json.dumps({"model": "m", "choices": [{"delta": {"content": "Hel"}}]}),
        json.dumps({"choices": [{"delta": {"content": "lo [1]"}}]}),
        json.dumps(
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "cost": 0.1}}
        ),
        "[DONE]",
    )

    def handler(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=body)

    items = list(llm.stream_chat(MESSAGES, "m", SECRET, client=client_with(handler)))
    assert items[:2] == ["Hel", "lo [1]"]
    final = items[-1]
    assert isinstance(final, llm.LLMResult)
    assert final.text == "Hello [1]"
    assert final.cost_usd == 0.1
    assert final.prompt_tokens == 7


def test_stream_error_status_is_friendly():
    def handler(request):
        return httpx.Response(401, text=SECRET)

    with pytest.raises(llm.LLMError) as caught:
        list(llm.stream_chat(MESSAGES, "m", SECRET, client=client_with(handler)))
    assert SECRET not in str(caught.value)
    assert caught.value.status_code == 401


def test_error_text_comes_only_from_the_fixed_list():
    odd = llm.LLMError("anything else, with a secret sk-or-123")
    assert odd.reason == "provider_error"
    assert odd.status_code == 502
    assert llm.public_message(odd) == llm.MESSAGES["provider_error"]
    assert "sk-or" not in llm.public_message(odd)
