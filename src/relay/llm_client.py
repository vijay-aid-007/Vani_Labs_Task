"""One-attempt chat client for an OpenAI-compatible endpoint (Groq or stand-in).

Deliberately dumb: no retries, no queueing, no rate limiting. Those belong to
the load layer (step 5) so they can be measured, swapped and switched off.
Switching real <-> stand-in means changing ``base_url`` only.
The API key is passed in by the caller and is never stored or logged here.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv

import httpx

from relay import config


@dataclass(frozen=True)
class ChatResult:
    """Everything observed about one HTTP attempt; nothing is interpreted."""

    status: int | None  # None = transport error before any HTTP response
    start_time: float  # epoch seconds when the request was sent
    ttft_ms: float | None  # send -> first non-empty content delta
    total_ms: float  # send -> stream end or error
    headers: dict[str, str]  # response headers, lower-case keys
    text: str  # concatenated streamed content
    usage: dict[str, Any] | None  # as reported by the server, else None
    error_body: str | None  # raw non-200 body, untouched
    error_type: str | None  # exception class name for transport/parse errors
    raw_events: list[str] = field(default_factory=list)  # raw SSE data payloads
    first_token_any_ms: float | None = None  # send -> first reasoning OR content delta

load_dotenv(config.ROOT / ".env")  # does not override a variable already set in the environment
def get_api_key() -> str:
    """Read the key from the environment; refuse placeholders without echoing the key."""
    key = os.environ.get(config.GROQ_API_KEY_ENV_VAR, "")
    if not key.startswith(config.API_KEY_PREFIX) or any(c in key for c in " <>"):
        raise RuntimeError(
            f"{config.GROQ_API_KEY_ENV_VAR} is unset or not a real key (length {len(key)}; "
            f"expected prefix {config.API_KEY_PREFIX!r} and no spaces or angle brackets)")
    return key

async def stream_chat(
    client: httpx.AsyncClient,
    messages: list[dict[str, str]],
    *,
    api_key: str,
    max_tokens: int,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
) -> ChatResult:
    """Send ONE streaming chat request and report what happened. Never retries."""
    model = model or config.GROQ_MODEL
    if model is None:
        raise ValueError("no model configured (config.GROQ_MODEL is None)")
    url = f"{base_url or config.GROQ_BASE_URL}/chat/completions"
    payload: dict[str, Any] = {"model": model, "messages": messages,
                               "max_tokens": max_tokens, "stream": True}
    if config.LLM_TEMPERATURE is not None:
        payload["temperature"] = config.LLM_TEMPERATURE

    start_time, t0 = time.time(), time.perf_counter()

    def ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    status: int | None = None
    headers: dict[str, str] = {}
    events: list[str] = []
    parts: list[str] = []
    ttft: float | None = None
    first_any: float | None = None
    usage: dict[str, Any] | None = None
    try:
        async with client.stream(
            "POST", url, json=payload, headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s or config.LLM_TIMEOUT_S,
        ) as resp:
            status = resp.status_code
            headers = {k.lower(): v for k, v in resp.headers.items()}
            if status != 200:
                body = (await resp.aread()).decode("utf-8", "replace")
                return ChatResult(status, start_time, None, ms(), headers, "", None, body, None)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                events.append(data)
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if (piece or delta.get("reasoning")) and first_any is None:
                        first_any = ms()
                    if piece:
                            if ttft is None:
                                ttft = ms()
                            parts.append(piece)
                usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return ChatResult(status, start_time, ttft, ms(), headers, "".join(parts), usage,
                          None, type(exc).__name__, events, first_any)
    return ChatResult(status, start_time, ttft, ms(), headers, "".join(parts), usage,
                      None, None, events, first_any)