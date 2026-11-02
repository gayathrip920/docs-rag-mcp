"""Talk to OpenRouter and keep track of time and cost.

The API key is passed in on every call. It is never stored, logged, or put
into an error message.
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS_EXTRA = {
    "HTTP-Referer": "https://github.com/gayathrip920/docs-rag-mcp",
    "X-Title": "docs-rag-mcp",
}


class LLMError(Exception):
    """A problem talking to OpenRouter. The message is safe to show to a user."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    latency_ms: int


def _clean_key(api_key: str | None) -> str:
    key = (api_key or "").strip()
    if not key:
        raise LLMError("Please enter your OpenRouter API key.", 401)
    if any(ch in key for ch in "\r\n\t "):
        raise LLMError("That API key does not look right.", 401)
    return key


def _payload(messages: list[dict], model: str, stream: bool) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "usage": {"include": True},  # asks OpenRouter to report the cost
    }
    if stream:
        body["stream"] = True
    return body


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    code = response.status_code
    if code == 401:
        raise LLMError("OpenRouter rejected the API key.", 401)
    if code == 402:
        raise LLMError("Your OpenRouter account has no credits left.", 402)
    if code == 429:
        raise LLMError("OpenRouter says there are too many requests. Try again soon.", 429)
    if code in (400, 404):
        raise LLMError("OpenRouter did not accept the request. Check the model name.", 400)
    raise LLMError(f"OpenRouter returned an error (status {code}).", 502)


def _usage_fields(usage: dict | None) -> tuple[int | None, int | None, float | None]:
    usage = usage or {}
    cost = usage.get("cost")
    return (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        float(cost) if cost is not None else None,
    )


def chat(
    messages: list[dict],
    model: str,
    api_key: str,
    timeout: float = 60,
    client: httpx.Client | None = None,
) -> LLMResult:
    key = _clean_key(api_key)
    headers = {"Authorization": f"Bearer {key}", **HEADERS_EXTRA}
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    started = time.perf_counter()
    try:
        response = client.post(
            OPENROUTER_URL, headers=headers, json=_payload(messages, model, False)
        )
    except httpx.HTTPError as exc:
        raise LLMError("Could not reach OpenRouter.", 502) from exc
    finally:
        if owns_client:
            client.close()
    latency_ms = int((time.perf_counter() - started) * 1000)
    _raise_for_status(response)

    data = response.json()
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("OpenRouter sent a reply in an unexpected shape.", 502) from exc
    prompt_tokens, completion_tokens, cost = _usage_fields(data.get("usage"))
    return LLMResult(
        text=text.strip(),
        model=data.get("model", model),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
    )


def stream_chat(
    messages: list[dict],
    model: str,
    api_key: str,
    timeout: float = 60,
    client: httpx.Client | None = None,
) -> Iterator[str | LLMResult]:
    """Yield text pieces as they arrive, then one final LLMResult."""
    key = _clean_key(api_key)
    headers = {"Authorization": f"Bearer {key}", **HEADERS_EXTRA}
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    started = time.perf_counter()
    pieces: list[str] = []
    usage = None
    served_model = model
    try:
        with client.stream(
            "POST", OPENROUTER_URL, headers=headers, json=_payload(messages, model, True)
        ) as response:
            if response.status_code >= 400:
                response.read()
                _raise_for_status(response)
            for line in response.iter_lines():
                # Lines starting with ":" are keep-alive comments.
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    raise LLMError("OpenRouter stopped the answer with an error.", 502)
                served_model = event.get("model", served_model)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        pieces.append(piece)
                        yield piece
    except httpx.HTTPError as exc:
        raise LLMError("Could not reach OpenRouter.", 502) from exc
    finally:
        if owns_client:
            client.close()

    prompt_tokens, completion_tokens, cost = _usage_fields(usage)
    yield LLMResult(
        text="".join(pieces).strip(),
        model=served_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
