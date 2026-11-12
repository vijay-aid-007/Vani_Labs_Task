"""Real-service probe (step 1). THIS SPENDS REAL GROQ QUOTA: read the plan first.

    python scripts/probe_groq.py --plan            # no network: request estimates
    python scripts/probe_groq.py --stage A --yes   # 1 request: headers, TTFT, usage
    python scripts/probe_groq.py --stage B --yes   # paced latency grid
    python scripts/probe_groq.py --stage C --yes   # concurrency ramp (in-flight behaviour)
    python scripts/probe_groq.py --stage D --yes   # provoke a 429, measure recovery

Stages B-D refuse to run until RPM_LIMIT and TPM_LIMIT are set in config.py
(from the Groq docs or the stage-A headers). Limits are never guessed.
Raw per-request records go to results/raw/probe_<stage>_<utc>.jsonl.
The API key comes from the environment and is never logged or written.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from probe_common import (RawWriter, b_cells, brief, est_tokens, make_prompt, one,  # noqa: E402
                          pace_s, print_plan, require_limits, retry_after_s)
from relay import config  # noqa: E402
from relay.llm_client import get_api_key, stream_chat  # noqa: E402

log = logging.getLogger("probe")


async def stage_a(client: httpx.AsyncClient, key: str, w: RawWriter) -> None:
    """One minimal request: headers, TTFT, whether streaming reports usage."""
    await one(client, key, w, 20, config.PROBE_A_MAX_TOKENS, stage="A")
    


async def stage_b(client: httpx.AsyncClient, key: str, w: RawWriter) -> None:
    """Paced grid over prompt size x output cap, to calibrate latency. Stops on any non-200."""
    for i, (words, mt) in enumerate(b_cells()):
        gap, t0 = pace_s(est_tokens(words, mt)), time.perf_counter()
        res = await one(client, key, w, words, mt, stage="B", seq=i)
        if res.status != 200:
            log.error("stage B stopped at seq=%d (status=%s); record kept", i, res.status)
            return
        await asyncio.sleep(max(0.0, gap - (time.perf_counter() - t0)))


async def stage_c(client: httpx.AsyncClient, key: str, w: RawWriter) -> None:
    """Concurrency ramp: bursts of k simultaneous requests; see how latency/status change."""
    for k in config.PROBE_C_LEVELS:
        results = await asyncio.gather(*(
            stream_chat(client, make_prompt(20), api_key=key, max_tokens=config.PROBE_C_MAX_TOKENS)
            for _ in range(k)))
        for idx, res in enumerate(results):
            w.write(record="request", stage="C", level=k, idx=idx, prompt_words=20,
                    max_tokens=config.PROBE_C_MAX_TOKENS, **asdict(res))
            log.info("C level=%d idx=%d %s", k, idx, brief(res))
        if any(r.status != 200 for r in results):
            log.warning("stage C stopped after level %d: non-200 seen; records kept", k)
            return
        await asyncio.sleep(config.PROBE_C_COOLDOWN_S)


async def stage_d(client: httpx.AsyncClient, key: str, w: RawWriter) -> None:
    """Burst until the first 429 (exact body kept), then poll to measure recovery time."""
    assert config.RPM_LIMIT
    first_429: float | None = None
    wait: float | None = None
    for i in range(config.RPM_LIMIT + config.PROBE_D_EXTRA_REQUESTS):
        res = await one(client, key, w, 20, config.PROBE_D_MAX_TOKENS, stage="D", phase="burst", seq=i)
        if res.status == 429:
            first_429, wait = time.perf_counter(), retry_after_s(res)
            break
    if first_429 is None:
        log.warning("no 429 within the burst cap; the RPM limit was not reached")
        return
    await asyncio.sleep(wait if wait is not None else config.PROBE_D_POLL_INTERVAL_S)
    for poll in range(config.PROBE_D_MAX_POLLS):
        res = await one(client, key, w, 20, config.PROBE_D_MAX_TOKENS, stage="D", phase="recovery",
                        poll=poll, s_since_first_429=round(time.perf_counter() - first_429, 3))
        if res.status == 200:
            log.info("recovered %.1fs after the first 429 (retry-after=%s)",
                     time.perf_counter() - first_429, wait)
            return
        await asyncio.sleep(config.PROBE_D_POLL_INTERVAL_S)
    log.error("not recovered after %d polls", config.PROBE_D_MAX_POLLS)


async def run(stage: str, key: str) -> None:
    """Run one stage and report where the raw records went."""
    w = RawWriter(stage)
    stages = {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d}
    async with httpx.AsyncClient() as client:
        await stages[stage](client, key, w)
    log.info("raw records: %s", w.path)


def main() -> int:
    """CLI entry point; returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print estimates, send nothing")
    ap.add_argument("--stage", choices=["A", "B", "C", "D"])
    ap.add_argument("--yes", action="store_true", help="confirm that real quota will be spent")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.plan or not args.stage:
        print_plan()
        return 0
    if not args.yes:
        print_plan()
        log.error("No requests sent. Re-run with --yes to spend quota.")
        return 2
    try:
        require_limits(args.stage)
        key = get_api_key()
    except (ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    asyncio.run(run(args.stage, key))
    return 0


if __name__ == "__main__":
    sys.exit(main())