"""Redact account identifiers from raw probe files in place. Idempotent.

Replaces the Groq organization id and the Cloudflare bot-management cookie value
(both appear in saved error bodies / headers). Everything else is left untouched,
byte for byte, including non-ASCII model output.

Usage (from the repo root):  python scripts/redact_raw.py
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay import config  # noqa: E402

log = logging.getLogger("redact_raw")
PATTERNS = (
    (re.compile(r"org_[0-9a-z]+"), "org_REDACTED"),
    (re.compile(r'__cf_bm=[^;"\\]+'), "__cf_bm=REDACTED"),
)


def main() -> int:
    """Redact every results/raw/*.jsonl file; return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    changed = 0
    for path in sorted(config.RAW_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8", newline="") as fh:
            text = fh.read()
        new = text
        for pattern, replacement in PATTERNS:
            new = pattern.sub(replacement, new)
        if new != text:
            with path.open("w", encoding="utf-8", newline="") as fh:
                fh.write(new)
            changed += 1
            log.info("redacted %s", path.name)
    log.info("done: %d file(s) changed", changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())