"""Fit stand-in parameters from real probe output (no network, no quota).

Reads results/raw/probe_*.jsonl and writes data/calibration.json with each
fitted parameter, its sample size n, and the source files.
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from relay import config

log = logging.getLogger("calibrate")
Rec = dict[str, Any]

TPM_RE = re.compile(
    r"Rate limit reached for model `(?P<model>[^`]+)` in organization `[^`]+` "
    r"service tier `(?P<tier>[^`]+)` on tokens per minute \(TPM\): "
    r"Limit (?P<limit>\d+), Used (?P<used>\d+), Requested (?P<req>\d+)\. "
    r"Please try again in (?P<secs>[\d.]+)s\."
)


def load_records(raw_dir: Path) -> tuple[list[Rec], list[str]]:
    """Return all 'request' records from probe files plus the file names read."""
    recs: list[Rec] = []
    files: list[str] = []
    for path in sorted(raw_dir.glob("probe_*.jsonl")):
        files.append(path.name)
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                log.warning("bad json at %s:%d (skipped)", path.name, i)
                continue
            if obj.get("record") == "request":
                recs.append(obj)
    return recs, files


def rtok(r: Rec) -> int | None:
    d = (r.get("usage") or {}).get("completion_tokens_details") or {}
    return d.get("reasoning_tokens")


def is_empty(r: Rec) -> bool:
    return not (r.get("text") or "").strip()


def pct(vals: list[float], p: float) -> float:
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

def lognorm(vals: list[float], what: str) -> dict[str, Any] | None:
# def lognorm(vals: list[float], what: str) -> dict[str, float] | None:
    """Lognormal fit (mu, sigma of ln x). None if fewer than STANDIN_FIT_MIN_N."""
    v = [x for x in vals if x is not None and x > 0]
    if len(v) < config.STANDIN_FIT_MIN_N:
        log.warning("%s: n=%d below minimum, NOT fitted (not measured)", what, len(v))
        return None
    logs = [math.log(x) for x in v]
    return {"mu": statistics.fmean(logs), "sigma": statistics.stdev(logs), "n": len(v),
            "min": min(v), "median": statistics.median(v), "max": max(v)}


# def ols(pairs: list[tuple[float, float]], what: str) -> dict[str, float] | None:
def ols(pairs: list[tuple[float, float]], what: str) -> dict[str, Any] | None:
    """Least squares y = intercept + slope*x with residual sd and Pearson r."""
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    n = len(pairs)
    if n < config.STANDIN_FIT_MIN_N or len(set(xs)) < 2:
        log.warning("%s: n=%d / distinct x=%d too small, NOT fitted", what, n, len(set(xs)))
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    icpt = my - slope * mx
    sse = sum((y - (icpt + slope * x)) ** 2 for x, y in pairs)
    return {"slope": slope, "intercept": icpt, "n": n, "distinct_x": len(set(xs)),
            "resid_sd": math.sqrt(sse / (n - 2)) if n > 2 else 0.0,
            "r": sxy / math.sqrt(sxx * syy) if syy > 0 else None}


def fit_components(ok: list[Rec]) -> dict[str, Any]:
    """Fit every latency/token component from successful requests."""
    q_ms: list[float] = []
    ov_ms: list[float] = []
    phase: list[tuple[float, float]] = []
    reas: list[float] = []
    content: list[float] = []
    tps: list[float] = []
    words: list[tuple[float, float]] = []
    for r in ok:
        u = r["usage"]
        fta, ttft, rt = r.get("first_token_any_ms"), r.get("ttft_ms"), rtok(r)
        if fta is not None and u.get("queue_time") is not None:
            q = u["queue_time"] * 1000
            q_ms.append(q)
            ov_ms.append(fta - q)
        if fta is not None and ttft is not None and rt:
            phase.append((rt, ttft - fta))
        if rt is not None and not is_empty(r):
            reas.append(rt)
        if not is_empty(r) and rt is not None and u.get("completion_tokens") is not None:
            c = u["completion_tokens"] - rt
            content.append(c)
            if (ttft is not None and c >= config.STANDIN_FIT_MIN_CONTENT_TOKENS
                    and r["total_ms"] > ttft):
                tps.append(c / ((r["total_ms"] - ttft) / 1000))
        if r.get("prompt_words") and u.get("prompt_tokens"):
            words.append((r["prompt_words"], u["prompt_tokens"]))
    decode = None
    if len(tps) >= config.STANDIN_FIT_MIN_N:
        decode = {"mean": statistics.fmean(tps), "sd": statistics.stdev(tps), "n": len(tps),
                  "min": min(tps), "median": statistics.median(tps), "max": max(tps)}
    else:
        log.warning("decode_tps: n=%d below minimum, NOT fitted", len(tps))
    fits = {"queue_ms": lognorm(q_ms, "queue_ms"), "overhead_ms": lognorm(ov_ms, "overhead_ms"),
            "reasoning_phase": ols(phase, "reasoning_phase"),
            "reasoning_tokens": lognorm(reas, "reasoning_tokens"),
            "content_tokens": lognorm(content, "content_tokens"),
            "decode_tps": decode, "prompt_tokens": ols(words, "prompt_tokens")}
    if fits["queue_ms"] and fits["overhead_ms"]:
        waits = [a + b for a, b in zip(q_ms, ov_ms)]
        rng = random.Random(config.SEED)
        sim = [rng.lognormvariate(fits["queue_ms"]["mu"], fits["queue_ms"]["sigma"])
               + rng.lognormvariate(fits["overhead_ms"]["mu"], fits["overhead_ms"]["sigma"])
               for _ in range(5000)]
        log.info("wait check n=%d observed p50/p90 = %.0f/%.0f ms; simulated = %.0f/%.0f ms",
                 len(waits), pct(waits, 50), pct(waits, 90), pct(sim, 50), pct(sim, 90))
    return fits


def usage_timing(ok: list[Rec]) -> dict[str, float]:
    us = [r["usage"] for r in ok]
    pst = [u["prompt_time"] / u["prompt_tokens"] for u in us
           if u.get("prompt_time") and u.get("prompt_tokens")]
    ctps = [u["completion_tokens"] / u["completion_time"] for u in us
            if u.get("completion_time") and u.get("completion_tokens")]
    return {"prompt_s_per_token": statistics.median(pst) if pst else 0.0,
            "server_completion_tps": statistics.median(ctps) if ctps else 0.0,
            "n": min(len(pst), len(ctps))}


def empty_behavior(ok: list[Rec]) -> dict[str, Any] | None:
    emp = [r for r in ok if is_empty(r) and rtok(r) is not None and r.get("max_tokens")]
    if not emp:
        return None
    offs = [r["max_tokens"] - rtok(r) for r in emp]
    gaps = [r["max_tokens"] - r["usage"]["completion_tokens"] for r in emp]
    return {"n": len(emp), "reasoning_offset": Counter(offs).most_common(1)[0][0],
            "completion_gap": Counter(gaps).most_common(1)[0][0],
            "offsets_seen": sorted(set(offs)), "gaps_seen": sorted(set(gaps))}


def stream_shape(ok: list[Rec]) -> dict[str, Any]:
    """Learn the SSE delta keys and where usage appears from recorded raw_events."""
    delta_keys: Counter[str] = Counter()
    top_keys: set[str] = set()
    usage_x = usage_top = False
    parsed = 0
    for r in ok:
        for ev in r.get("raw_events") or []:
            s = ev.strip()
            if s.startswith("data:"):
                s = s[5:].strip()
            if not s or s == "[DONE]":
                continue
            try:
                ch = json.loads(s)
            except json.JSONDecodeError:
                continue
            if not isinstance(ch, dict):
                continue
            parsed += 1
            top_keys.update(ch.keys())
            usage_top |= bool(ch.get("usage"))
            xg = ch.get("x_groq")
            usage_x |= isinstance(xg, dict) and bool(xg.get("usage"))
            for choice in ch.get("choices") or []:
                delta_keys.update((choice.get("delta") or {}).keys())
    extra = [k for k in delta_keys if k not in ("role", "content")]
    if not (usage_x or usage_top) or not extra:
        log.warning("stream shape incomplete (usage_x=%s top=%s reasoning_key=%s); "
                    "server will fall back and say so", usage_x, usage_top, extra)
    return {"events_parsed": parsed, "delta_keys": sorted(delta_keys),
            "reasoning_key": extra[0] if extra else None, "top_level_keys": sorted(top_keys),
            "usage_in_x_groq": usage_x, "usage_top_level": usage_top}


def header_info(recs: list[Rec]) -> dict[str, Any]:
    out: dict[str, dict[str, str]] = {"success": {}, "429": {}}
    for r in recs:
        status = r.get("status")
        kind = "success" if status == 200 else "429" if status == 429 else None
        if not kind:
            continue
        for k, v in (r.get("headers") or {}).items():
            if k.startswith("x-ratelimit") or k == "retry-after":
                out[kind].setdefault(k, v)
    return {k: {"names": sorted(v), "examples": v} for k, v in out.items()}


def fit_429(recs: list[Rec]) -> dict[str, Any] | None:
    rs = [r for r in recs if r.get("status") == 429]
    if not rs:
        return None
    r = rs[0]
    body = r.get("error_body") or ""
    try:
        msg = json.loads(body)["error"]["message"]
    except (json.JSONDecodeError, KeyError, TypeError):
        msg = body
    hdr = {k: v for k, v in (r.get("headers") or {}).items()
           if k.startswith("x-ratelimit") or k == "retry-after"}
    m = TPM_RE.search(msg)
    out: dict[str, Any] = {"n": len(rs), "recorded_body": body, "recorded_headers": hdr,
                           "message_pattern_matches": bool(m),
                           "latency_ms": [x["total_ms"] for x in rs if x.get("total_ms") is not None]}
    if m:
        g = m.groupdict()
        secs = float(g["secs"])
        ra = hdr.get("retry-after")
        out.update(tier=g["tier"], limit=int(g["limit"]), used=int(g["used"]),
                   requested=int(g["req"]), body_secs=secs,
                   retry_after_rule_ok=ra is not None and math.ceil(secs) == int(float(ra)))
        try:
            rem = float(hdr["x-ratelimit-remaining-tokens"])
            reset = float(str(hdr["x-ratelimit-reset-tokens"]).rstrip("s"))
            out["reset_tokens_formula_err_s"] = reset - (int(g["limit"]) - rem) * 60 / int(g["limit"])
        except (KeyError, ValueError):
            pass
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    recs, files = load_records(Path(config.RAW_DIR))
    ok = [r for r in recs if r.get("status") == 200 and isinstance(r.get("usage"), dict)]
    log.info("files=%d request_records=%d ok_with_usage=%d", len(files), len(recs), len(ok))
    cal = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
           "model": config.GROQ_MODEL, "seed": config.SEED, "source_files": files,
           "n_request_records": len(recs), "n_ok": len(ok),
           "fits": fit_components(ok), "usage_timing": usage_timing(ok),
           "empty_behavior": empty_behavior(ok), "stream_shape": stream_shape(ok),
           "headers": header_info(recs), "rate_limit_429": fit_429(recs)}
    out = Path(config.CALIBRATION_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    for name, f in cal["fits"].items():
        log.info("%-16s %s", name, "NOT FITTED" if f is None else json.dumps(f))
    log.info("empty_behavior=%s", cal["empty_behavior"])
    log.info("429: %s", {k: v for k, v in (cal["rate_limit_429"] or {}).items() if k != "recorded_body"})
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()