"""FastAPI stand-in for the Groq chat endpoint (OpenAI wire format; swap base_url only)."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from relay import config
from relay.standin.model import (Controls, Limiter, Sampler, estimate_prompt_tokens,
                                 fmt_duration, load_calibration, require)

log = logging.getLogger("standin.server")

KNOWN_GAPS = [
    "Latency fitted from one client under light load (small n); no overload data, tail is a lognormal extrapolation.",
    "Load slowdown and in-flight cap: not measured (real service showed none up to 8 concurrent); off unless enabled as a what-if.",
    "Only the TPM 429 body/headers were recorded; RPD, RPM and too-large bodies are extrapolated (flagged by x-standin-extrapolated).",
    "Whether the real service refunds unused max_tokens, or counts failed requests: not measured.",
    "Answers are placeholder words: stand-in rows cannot measure answer quality.",
    "No real network/TLS/DNS; fitted waits are client-observed on the real path, so local overhead is ~0 extra.",
    "Windows timer granularity makes sub-15 ms chunk gaps bunch up; totals stay correct (deadline scheduling).",
    "Prompt tokens are a words->tokens line fitted on two prompt sizes, not a tokenizer.",
]
CONTROL_FIELDS: dict[str, type] = {
    "fail_rate": float, "fail_status": int, "hang_rate": float, "slow_factor": float,
    "load_slowdown_per_inflight": float, "tpm_limit": int, "rpm_limit": int,
    "rpm_enforced": bool, "rpd_limit": int, "refund_unused": bool}


class AppState:
    """Mutable server state: controls, limiter, counters, calibration."""

    def __init__(self, cal: dict[str, Any]) -> None:
        self.cal = cal
        self.sampler = Sampler(cal, config.SEED)
        self.warned_headers = False
        self.reset()

    def reset(self) -> None:
        tier = (self.cal.get("rate_limit_429") or {}).get("tier", "on_demand")
        self.controls = Controls()
        self.limiter = Limiter(self.controls, require(config.GROQ_MODEL, "GROQ_MODEL"), tier)  
        self.inflight = self.peak = self.arrivals = self.rejected = 0

    def apply(self, body: dict[str, Any]) -> None:
        c, lim = self.controls, self.limiter
        if body.get("reset"):
            self.reset()
            return
        for k, typ in CONTROL_FIELDS.items():
            if k in body:
                setattr(c, k, None if body[k] is None else typ(body[k]))
        if "tpm_limit" in body:
            lim.tpm.resize(c.tpm_limit, c.tpm_limit / 60)
        if "rpm_limit" in body:
            lim.rpm.resize(c.rpm_limit, c.rpm_limit / 60)
        if "rpd_limit" in body and not c.daily_frozen:
            lim.rpd.resize(c.rpd_limit, c.rpd_limit / 86400)
        if "rpd_exhausted" in body:
            c.daily_frozen = bool(body["rpd_exhausted"])
            lim.rpd.resize(c.rpd_limit, 0.0 if c.daily_frozen else c.rpd_limit / 86400)
            lim.rpd.set_level(0.0 if c.daily_frozen else c.rpd_limit)

    def reject_latency_s(self) -> float:
        lat = (self.cal.get("rate_limit_429") or {}).get("latency_ms") or []
        return statistics.median(lat) / 1000 if lat else 0.0

    def log_event(self, **ev: Any) -> None:
        path = Path(config.STANDIN_SERVER_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **ev}) + "\n")


def _rl_headers(st: AppState, kind: str, retry_after: int | None = None) -> dict[str, str]:
    lim = st.limiter
    rpd_reset = lim.rpd.reset_s
    if math.isinf(rpd_reset):      # daily budget frozen: no refill, use the chosen constant
        rpd_reset = config.STANDIN_DAILY_FROZEN_RETRY_AFTER_S
    vals = {"x-ratelimit-limit-requests": str(int(lim.rpd.capacity)),
            "x-ratelimit-limit-tokens": str(int(lim.tpm.capacity)),
            "x-ratelimit-remaining-requests": str(int(lim.rpd.level)),
            "x-ratelimit-remaining-tokens": str(int(lim.tpm.level)),
            "x-ratelimit-reset-requests": fmt_duration(rpd_reset),
            "x-ratelimit-reset-tokens": fmt_duration(lim.tpm.reset_s)}
    if retry_after is not None:
        vals["retry-after"] = str(retry_after)
    names = (st.cal.get("headers") or {}).get(kind, {}).get("names") or list(vals)
    missing = [n for n in names if n not in vals]
    if missing and not st.warned_headers:
        log.warning("recorded header names not emulated: %s", missing)
        st.warned_headers = True
    out = {n: vals[n] for n in names if n in vals}
    out["x-standin-service"] = "relay-standin"
    return out


def _error(status: int, msg: str, typ: str, code: str,
           headers: dict[str, str] | None = None) -> JSONResponse:
    h = {"x-standin-service": "relay-standin", **(headers or {})}
    return JSONResponse({"error": {"message": msg, "type": typ, "code": code}},
                        status_code=status, headers=h)


def _chunk(cid: str, created: int, model: str, delta: dict[str, Any],
           finish: str | None = None, extra: dict[str, Any] | None = None) -> str:
    d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
         "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}
    d.update(extra or {})
    return f"data: {json.dumps(d)}\n\n"


async def _sleep_until(t0: float, ms: float) -> None:
    """Deadline scheduling: sleep until t0+ms so timer jitter does not accumulate."""
    d = ms / 1000 - (time.perf_counter() - t0)
    if d > 0:
        await asyncio.sleep(d)


def create_app(cal_path: str | Path | None = None) -> FastAPI:
    cal = load_calibration(cal_path or config.CALIBRATION_FILE)
    st = AppState(cal)
    default_model: str = require(config.GROQ_MODEL, "GROQ_MODEL")
    shape = cal.get("stream_shape") or {}
    emit_x, emit_top = bool(shape.get("usage_in_x_groq")), bool(shape.get("usage_top_level"))
    if not (emit_x or emit_top):
        emit_x = emit_top = True
        log.warning("usage location not detected in probe data: emitting both")
    rkey = shape.get("reasoning_key") or "reasoning"
    if not shape.get("reasoning_key"):
        log.warning("reasoning delta key not detected: falling back to 'reasoning'")
    app = FastAPI(title="relay-standin")

    async def chat(request: Request) -> Any:
        t0 = time.perf_counter()
        if not request.headers.get("authorization"):
            return _error(401, "Invalid API Key", "invalid_request_error", "invalid_api_key")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "body is not JSON", "invalid_request_error", "bad_body")
        msgs = body.get("messages")
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
        if not isinstance(msgs, list) or not isinstance(max_tokens, int) or max_tokens < 1:
            return _error(400, "need messages and integer max_tokens", "invalid_request_error", "bad_body")
        c = st.controls
        st.arrivals += 1
        plan = st.sampler.plan(max_tokens, st.inflight, c)
        if st.sampler.rng.random() < c.hang_rate:
            await asyncio.sleep(config.STANDIN_HANG_S)
        if st.sampler.rng.random() < c.fail_rate:
            await _sleep_until(t0, plan.wait_ms)
            st.log_event(event="injected_failure", status=c.fail_status)
            return _error(c.fail_status, "stand-in injected failure", "server_error", "injected_failure")
        p_tok = estimate_prompt_tokens(msgs, cal["fits"]["prompt_tokens"])
        charge = p_tok + max_tokens
        rej = st.limiter.admit(charge)
        if rej:
            st.rejected += 1
            await asyncio.sleep(st.reject_latency_s())
            ra = math.ceil(rej.retry_after_s) if rej.retry_after_s > 0 else None
            h = _rl_headers(st, "429", ra)
            if rej.extrapolated:
                h["x-standin-extrapolated"] = rej.kind
            st.log_event(event="rejected", kind=rej.kind, charge=charge, inflight=st.inflight,
                         tpm_level=st.limiter.tpm.level)
            return _error(rej.status, rej.message, rej.type, rej.code, h)
        st.inflight += 1
        st.peak = max(st.peak, st.inflight)
        at_arrival = st.inflight
        usage = st.sampler.usage(plan, p_tok)
        headers = _rl_headers(st, "success")
        cid, created = f"chatcmpl-{uuid.uuid4()}", int(time.time())
        model: str = body.get("model") or default_model

        def finish(done: bool) -> None:
            st.inflight -= 1
            if done and c.refund_unused:
                st.limiter.tpm.give(max(0, max_tokens - plan.completion_tokens))
            st.log_event(event="completed" if done else "aborted", charge=charge,
                         inflight_at_arrival=at_arrival, wait_ms=plan.wait_ms,
                         reasoning_tokens=plan.reasoning_tokens, content_tokens=plan.content_tokens,
                         total_ms=plan.total_ms, tpm_level=st.limiter.tpm.level)

        if body.get("stream"):
            async def gen() -> AsyncIterator[str]:
                done = False
                ck = config.STANDIN_CHUNK_TOKENS
                try:
                    await _sleep_until(t0, plan.wait_ms)
                    yield _chunk(cid, created, model, {"role": "assistant", "content": ""})
                    rt, ct = plan.reasoning_tokens, plan.content_tokens
                    for s in range(0, rt, ck):
                        await _sleep_until(t0, plan.wait_ms + plan.reasoning_ms * s / rt)
                        yield _chunk(cid, created, model, {rkey: " r" * min(ck, rt - s)})
                    c0 = plan.wait_ms + plan.reasoning_ms
                    for s in range(0, ct, ck):
                        await _sleep_until(t0, c0 + plan.content_ms * s / ct)
                        yield _chunk(cid, created, model, {"content": " w" * min(ck, ct - s)})
                    await _sleep_until(t0, plan.total_ms)
                    extra: dict[str, Any] = {}
                    if emit_x:
                        extra["x_groq"] = {"id": cid.replace("chatcmpl", "req"), "usage": usage}
                    if emit_top:
                        extra["usage"] = usage
                    yield _chunk(cid, created, model, {}, plan.finish_reason, extra)
                    yield "data: [DONE]\n\n"
                    done = True
                finally:
                    finish(done)
            return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

        done = False
        try:
            await _sleep_until(t0, plan.total_ms)
            done = True
            msg = {"role": "assistant", "content": " w" * plan.content_tokens,
                   rkey: " r" * plan.reasoning_tokens}
            return JSONResponse({"id": cid, "object": "chat.completion", "created": created,
                                 "model": model, "usage": usage,
                                 "choices": [{"index": 0, "message": msg,
                                              "finish_reason": plan.finish_reason}]},
                                headers=headers)
        finally:
            finish(done)

    async def control(request: Request) -> Any:
        body = await request.json()
        unknown = set(body) - set(CONTROL_FIELDS) - {"reset", "rpd_exhausted"}
        if unknown:
            return _error(400, f"unknown control fields {sorted(unknown)}", "invalid_request_error", "bad_control")
        st.apply(body)
        return await state(request)

    async def state(_: Request) -> Any:
        lim = st.limiter
        return {"controls": vars(st.controls), "inflight": st.inflight, "peak_inflight": st.peak,
                "arrivals": st.arrivals, "rejected": st.rejected,
                "tpm_level": lim.tpm.level, "rpd_level": lim.rpd.level, "rpm_level": lim.rpm.level,
                "known_gaps": KNOWN_GAPS}

    for path in ("/v1/chat/completions", "/openai/v1/chat/completions"):
        app.add_api_route(path, chat, methods=["POST"])
    app.add_api_route("/_control", control, methods=["POST"])
    app.add_api_route("/_state", state, methods=["GET"])
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if config.STANDIN_HOST is None or config.STANDIN_PORT is None:
        raise SystemExit("set STANDIN_HOST / STANDIN_PORT in config.py")
    uvicorn.run(create_app(), host=config.STANDIN_HOST, port=config.STANDIN_PORT, log_level="warning")


if __name__ == "__main__":
    main()