"""Shared helpers for the real-service probe: prompts, pacing, plan, gating, raw writer."""
from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay import config  # noqa: E402
from relay.llm_client import ChatResult, stream_chat  # noqa: E402

log = logging.getLogger("probe")
_WORDS = ("the", "model", "reads", "some", "text", "about", "queues", "and")
_FIXED_WORDS = 12  # words in the fixed wrapper text; used for estimates only


def make_prompt(n_words: int) -> list[dict[str, str]]:
    """Deterministic prompt: ~n_words of padding + a question that elicits a long answer."""
    padding = " ".join(_WORDS[i % len(_WORDS)] for i in range(n_words))
    question = "Explain in detail how a transformer neural network works."
    return [{"role": "user", "content": f"Ignore this padding: {padding}\n\n{question}"}]


def est_tokens(n_words: int, max_tokens: int) -> int:
    """PLANNING estimate only (input guess + output cap); real counts come from usage."""
    return int((n_words + _FIXED_WORDS) * config.TOKENS_PER_WORD_ESTIMATE) + max_tokens


def b_cells() -> list[tuple[int, int]]:
    """Stage B grid of (prompt_words, max_tokens), repeated, shuffled with the fixed seed."""
    cells = [(w, m) for w in config.PROBE_B_PROMPT_WORDS for m in config.PROBE_B_MAX_TOKENS]
    cells *= config.PROBE_B_REPS
    random.Random(config.SEED).shuffle(cells)
    return cells


def pace_s(tokens: int) -> float:
    """Seconds between request starts that stays under the safety share of RPM and TPM."""
    assert config.RPM_LIMIT and config.TPM_LIMIT
    f = config.PROBE_SAFETY_FRACTION
    return max(60.0 / (config.RPM_LIMIT * f), tokens * 60.0 / (config.TPM_LIMIT * f))


def print_plan() -> None:
    """Log request/token estimates per stage and the quota facts we do not have."""
    rpm = config.RPM_LIMIT
    nd = (f"<= {rpm + config.PROBE_D_EXTRA_REQUESTS} burst + <= {config.PROBE_D_MAX_POLLS} polls"
          if rpm else "not measured (needs RPM_LIMIT)")
    nc = sum(config.PROBE_C_LEVELS)
    log.info("model=%s base_url=%s", config.GROQ_MODEL, config.GROQ_BASE_URL)
    log.info("A: 1 request, ~%d tokens (estimate)", est_tokens(20, config.PROBE_A_MAX_TOKENS))
    log.info("B: %d requests, ~%d tokens (estimate), paced",
             len(b_cells()), sum(est_tokens(w, m) for w, m in b_cells()))
    log.info("C: %d requests in bursts %s, ~%d tokens (estimate)", nc, config.PROBE_C_LEVELS,
             nc * est_tokens(20, config.PROBE_C_MAX_TOKENS))
    log.info("D: %s; tiny prompts", nd)
    log.info("Remaining daily budget: not measured (see x-ratelimit-remaining-requests in stage A)")
    log.info("RPM=%s TPM=%s RPD=%s TPD=%s (None = not measured)", config.RPM_LIMIT,
             config.TPM_LIMIT, config.RPD_LIMIT, config.TPD_LIMIT)


def require_limits(stage: str) -> None:
    """Refuse stages B-D unless the limits they pace against are known."""
    if stage == "A":
        return
    missing = [n for n in ("RPM_LIMIT", "TPM_LIMIT") if getattr(config, n) is None]
    if missing:
        raise ValueError(f"{missing} not set in config.py. Fill them from the Groq docs or "
                         "the stage-A headers; limits are never guessed.")
    assert config.RPM_LIMIT and config.TPM_LIMIT
    f = config.PROBE_SAFETY_FRACTION
    if stage == "B" and max(est_tokens(w, m) for w, m in b_cells()) > config.TPM_LIMIT * f:
        raise ValueError("largest stage-B request exceeds the TPM safety share; shrink PROBE_B_*")
    if stage == "C" and sum(config.PROBE_C_LEVELS) > config.RPM_LIMIT * f:
        raise ValueError("stage-C bursts exceed the RPM safety share; shrink PROBE_C_LEVELS")


class RawWriter:
    """Append-only JSONL writer: one record per HTTP attempt, written immediately."""

    def __init__(self, stage: str) -> None:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = config.RAW_DIR / f"probe_{stage}_{stamp}.jsonl"
        self.write(record="meta", stage=stage, model=config.GROQ_MODEL,
                   base_url=config.GROQ_BASE_URL, started_utc=stamp)

    def write(self, **fields: Any) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fields, default=str) + "\n")


def brief(res: ChatResult) -> str:
    """One-line log summary: status, timings, usage, rate-limit headers."""
    rl = {k: v for k, v in res.headers.items() if k.startswith("x-ratelimit") or k == "retry-after"}
    return f"status={res.status} ttft_ms={res.ttft_ms} total_ms={res.total_ms:.1f} usage={res.usage} {rl}"


def retry_after_s(res: ChatResult) -> float | None:
    """Parse the retry-after header (seconds) if present and numeric."""
    try:
        return float(res.headers["retry-after"])
    except (KeyError, ValueError):
        return None


async def one(client: httpx.AsyncClient, key: str, w: RawWriter, words: int,
              max_tokens: int, **tags: Any) -> ChatResult:
    """Send one request, write its raw record, log a summary."""
    res = await stream_chat(client, make_prompt(words), api_key=key, max_tokens=max_tokens)
    w.write(record="request", prompt_words=words, max_tokens=max_tokens, **tags, **asdict(res))
    log.info("%s %s", tags, brief(res))
    return res