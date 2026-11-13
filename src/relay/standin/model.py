"""Pure stand-in logic (no FastAPI): calibration, latency sampling, buckets, 429 shaping."""
from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from relay import config


log = logging.getLogger("standin.model")
T = TypeVar("T")
def require(value: T | None, name: str) -> T:
    """Return a config value, or fail loudly if it is None (= not measured/set)."""
    if value is None:
        raise RuntimeError(f"config.{name} is None (not measured/set)")
    return value

ORG = "org_STANDIN"
TAIL = "Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing"
REQUIRED_FITS = ("queue_ms", "overhead_ms", "reasoning_phase", "reasoning_tokens",
                 "content_tokens", "decode_tps", "prompt_tokens")


def fmt_duration(s: float) -> str:
    """Go-style like the recorded headers: '56.64s', '7m12s' (trailing zeros dropped)."""
    def num(x: float) -> str:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    if s < 60:
        return f"{num(s)}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m{num(sec)}s"


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Load calibration.json; refuse to run if any needed fit is missing (not measured)."""
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_FITS if not (cal.get("fits") or {}).get(k)]
    if missing:
        raise RuntimeError(f"calibration lacks fits for {missing}: not measured")
    return cal


def estimate_prompt_tokens(messages: list[Any], fit: dict[str, float]) -> int:
    """Words -> tokens via the fitted line (no tokenizer; see known gaps)."""
    words = sum(len(str(m.get("content", "")).split()) for m in messages if isinstance(m, dict))
    return max(1, round(fit["intercept"] + fit["slope"] * words))


@dataclass
class Controls:
    """Runtime knobs; scenarios (limit halved, daily exhausted, faults) change these."""
    fail_rate: float = config.STANDIN_FAIL_RATE
    fail_status: int = config.STANDIN_FAIL_STATUS
    hang_rate: float = config.STANDIN_HANG_RATE
    slow_factor: float = config.STANDIN_SLOW_FACTOR
    load_slowdown_per_inflight: float | None = config.STANDIN_LOAD_SLOWDOWN_PER_INFLIGHT
    tpm_limit: int = require(config.TPM_LIMIT, "TPM_LIMIT")
    rpm_limit: int = require(config.RPM_LIMIT, "RPM_LIMIT")
    rpm_enforced: bool = config.STANDIN_RPM_ENFORCED
    rpd_limit: int = require(config.RPD_LIMIT, "RPD_LIMIT")
    daily_frozen: bool = False
    refund_unused: bool = config.STANDIN_REFUND_UNUSED


class Bucket:
    """Continuous token bucket: level refills at per_second up to capacity."""

    def __init__(self, capacity: float, per_second: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.capacity, self.per_second, self._clock = float(capacity), float(per_second), clock
        self._level, self._t = float(capacity), clock()

    def _refill(self) -> None:
        now = self._clock()
        self._level = min(self.capacity, self._level + (now - self._t) * self.per_second)
        self._t = now

    @property
    def level(self) -> float:
        self._refill()
        return self._level

    @property
    def reset_s(self) -> float:
        """Seconds until the bucket is full again (matches reset-tokens = (cap-level)/refill)."""
        self._refill()
        return (self.capacity - self._level) / self.per_second if self.per_second > 0 else math.inf

    def wait_for(self, amount: float) -> float:
        self._refill()
        if amount <= self._level:
            return 0.0
        return (amount - self._level) / self.per_second if self.per_second > 0 else math.inf

    def take(self, amount: float) -> None:
        self._refill()
        self._level -= amount

    def give(self, amount: float) -> None:
        self._refill()
        self._level = min(self.capacity, self._level + amount)

    def set_level(self, level: float) -> None:
        self._refill()
        self._level = min(self.capacity, level)

    def resize(self, capacity: float, per_second: float) -> None:
        self._refill()
        self.capacity, self.per_second = float(capacity), float(per_second)
        self._level = min(self._level, self.capacity)


@dataclass(frozen=True)
class Reject:
    kind: str            # tpm | rpm | rpd | too_large
    status: int
    code: str
    type: str
    message: str
    retry_after_s: float
    extrapolated: bool   # True = body/behavior NOT observed on the real service


class Limiter:
    """TPM + RPD (always) and RPM (optional) buckets; rejected requests charge nothing."""

    def __init__(self, c: Controls, model: str, tier: str,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.c, self.model, self.tier = c, model, tier
        self.tpm = Bucket(c.tpm_limit, c.tpm_limit / 60, clock)
        self.rpm = Bucket(c.rpm_limit, c.rpm_limit / 60, clock)
        self.rpd = Bucket(c.rpd_limit, c.rpd_limit / 86400, clock)

    def _head(self, unit: str) -> str:
        return (f"Rate limit reached for model `{self.model}` in organization `{ORG}` "
                f"service tier `{self.tier}` on {unit}")

    def admit(self, charge: int) -> Reject | None:
        """Charge prompt+max_tokens at admission, or return why not (nothing charged)."""
        c = self.c
        if charge > self.tpm.capacity:
            msg = (f"Request too large for model `{self.model}` in organization `{ORG}` on tokens "
                   f"per minute (TPM): Limit {int(self.tpm.capacity)}, Requested {charge}.")
            return Reject("too_large", 413, "request_too_large", "tokens", msg, 0.0, True)
        w_rpd = self.rpd.wait_for(1)
        if w_rpd > 0:
            secs = config.STANDIN_DAILY_FROZEN_RETRY_AFTER_S if math.isinf(w_rpd) else w_rpd
            used = round(self.rpd.capacity - self.rpd.level)
            msg = (f"{self._head('requests per day (RPD)')}: Limit {int(self.rpd.capacity)}, "
                   f"Used {used}, Requested 1. Please try again in {fmt_duration(secs)}. {TAIL}")
            return Reject("rpd", 429, "rate_limit_exceeded", "requests", msg, secs, True)
        w_tpm = self.tpm.wait_for(charge)
        if w_tpm > 0:
            used = round(self.tpm.capacity - self.tpm.level)
            msg = (f"{self._head('tokens per minute (TPM)')}: Limit {int(self.tpm.capacity)}, "
                   f"Used {used}, Requested {charge}. Please try again in {w_tpm:.9f}s. {TAIL}")
            return Reject("tpm", 429, "rate_limit_exceeded", "tokens", msg, w_tpm, False)
        w_rpm = self.rpm.wait_for(1) if c.rpm_enforced else 0.0
        if w_rpm > 0:
            used = round(self.rpm.capacity - self.rpm.level)
            msg = (f"{self._head('requests per minute (RPM)')}: Limit {int(self.rpm.capacity)}, "
                   f"Used {used}, Requested 1. Please try again in {w_rpm:.3f}s. {TAIL}")
            return Reject("rpm", 429, "rate_limit_exceeded", "requests", msg, w_rpm, True)
        self.tpm.take(charge)
        self.rpd.take(1)
        if c.rpm_enforced:
            self.rpm.take(1)
        return None


@dataclass(frozen=True)
class Plan:
    """Sampled timeline for one admitted request (all ms already include slow_factor)."""
    queue_ms: float
    overhead_ms: float
    reasoning_tokens: int
    reasoning_ms: float
    content_tokens: int
    content_ms: float
    completion_tokens: int
    finish_reason: str

    @property
    def wait_ms(self) -> float:
        """Arrival -> first reasoning-or-content delta (first_token_any)."""
        return self.queue_ms + self.overhead_ms

    @property
    def total_ms(self) -> float:
        return self.wait_ms + self.reasoning_ms + self.content_ms


class Sampler:
    """Draws per-request timelines from the fitted distributions (seeded)."""

    def __init__(self, cal: dict[str, Any], seed: int) -> None:
        self.f, self.rng = cal["fits"], random.Random(seed)
        self.eb, self.ut = cal.get("empty_behavior") or {}, cal.get("usage_timing") or {}

    def _ln(self, key: str) -> float:
        return self.rng.lognormvariate(self.f[key]["mu"], self.f[key]["sigma"])

    def plan(self, max_tokens: int, inflight: int, c: Controls) -> Plan:
        k = c.slow_factor
        q = self._ln("queue_ms")
        if c.load_slowdown_per_inflight is not None:     # what-if only; not measured
            q *= 1 + c.load_slowdown_per_inflight * max(0, inflight - 1)
        ov = self._ln("overhead_ms")
        budget = max(0, max_tokens - self.eb.get("reasoning_offset", 0))
        r_raw = max(1, round(self._ln("reasoning_tokens")))
        if r_raw >= budget:                              # reasoning ate the cap -> empty answer
            reasoning, content = budget, 0
            comp, finish = max(0, max_tokens - self.eb.get("completion_gap", 0)), "length"
        else:
            reasoning = r_raw
            content = min(max(1, round(self._ln("content_tokens"))), max_tokens - reasoning)
            comp = reasoning + content
            finish = "length" if comp >= max_tokens else "stop"
        rp, d = self.f["reasoning_phase"], self.f["decode_tps"]
        r_ms = max(0.0, rp["intercept"] + rp["slope"] * reasoning
                   + self.rng.gauss(0, rp["resid_sd"]))
        tps = min(max(self.rng.gauss(d["mean"], d["sd"]), d["min"]), d["max"])
        return Plan(q * k, ov * k, reasoning, r_ms * k, content, content / tps * 1000 * k,
                    comp, finish)

    def usage(self, p: Plan, prompt_tokens: int) -> dict[str, Any]:
        pst = self.ut.get("prompt_s_per_token", 0.0)
        ctps = self.ut.get("server_completion_tps") or 1.0
        comp_t = p.completion_tokens / ctps
        return {"queue_time": p.queue_ms / 1000, "prompt_tokens": prompt_tokens,
                "prompt_time": prompt_tokens * pst, "completion_tokens": p.completion_tokens,
                "completion_time": comp_t, "total_tokens": prompt_tokens + p.completion_tokens,
                "total_time": prompt_tokens * pst + comp_t,
                "completion_tokens_details": {"reasoning_tokens": p.reasoning_tokens}}