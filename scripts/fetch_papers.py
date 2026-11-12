"""Download the three library papers at exact arXiv versions; record SHA-256.

Usage (from the repo root):  python scripts/fetch_papers.py

Safety properties:
  * Never overwrites an existing PDF. A file whose hash differs from the one
    recorded in CHECKSUMS.txt is an error, not something to "fix".
  * A download is validated BEFORE it touches disk and is written atomically
    (temp file + rename), so a failed run leaves no partial PDF behind.
  * Re-running with everything present downloads nothing.
  * Exit code is non-zero on any failure.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay import config  # noqa: E402

log = logging.getLogger("fetch_papers")


class FetchError(RuntimeError):
    """Any condition that must stop the run loudly."""


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def load_checksums(path: Path) -> dict[str, str]:
    """Parse sha256sum-style lines ('<hex>  <filename>') into {filename: hex}."""
    sums: dict[str, str] = {}
    if not path.exists():
        return sums
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise FetchError(f"{path}:{lineno}: malformed checksum line {line!r}")
        sums[parts[1]] = parts[0].lower()
    return sums


def save_checksums(path: Path, sums: dict[str, str]) -> None:
    """Atomically write checksums, sorted by filename."""
    tmp = path.with_name(path.name + ".part")
    body = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    tmp.write_text(body, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_bytes_atomic(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` via a temp file so no partial file remains."""
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def validate_pdf(data: bytes, paper: config.Paper) -> int:
    """Raise FetchError unless ``data`` is a complete, openable PDF.

    Returns the page count. The arXiv version stamp on page 1 is checked as
    evidence of the exact version, but only warns: it is a text heuristic.
    """
    name = paper.filename
    if len(data) < config.MIN_PDF_BYTES:
        raise FetchError(f"{name}: only {len(data)} bytes (< {config.MIN_PDF_BYTES}); not a paper PDF")
    if not data.startswith(b"%PDF-"):
        raise FetchError(f"{name}: not a PDF (starts with {data[:20]!r})")
    if b"%%EOF" not in data[-config.PDF_EOF_SCAN_BYTES:]:
        raise FetchError(f"{name}: no %%EOF near the end; download looks truncated")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = doc.page_count
            first_page_text = str(doc[0].get_text("text")) if pages else ""
    except Exception as exc:  # PyMuPDF raises several unrelated types
        raise FetchError(f"{name}: PyMuPDF cannot open it: {exc}") from exc
    if pages < 1:
        raise FetchError(f"{name}: PDF has zero pages")
    stamp = f"arXiv:{paper.arxiv_id}v{paper.version}"
    if stamp in "".join(first_page_text.split()):
        log.info("%s: valid PDF, %d pages, version stamp %s found", name, pages, stamp)
    else:
        log.warning("%s: valid PDF, %d pages, but stamp %s NOT found on page 1; "
                    "check the version by hand", name, pages, stamp)
    return pages


def download(client: httpx.Client, paper: config.Paper) -> bytes:
    """GET the paper PDF. HTTP errors raise; content is validated by the caller."""
    log.info("GET %s", paper.url)
    resp = client.get(paper.url)
    resp.raise_for_status()
    log.info("%s: HTTP %d, content-type=%s, %d bytes", paper.filename,
             resp.status_code, resp.headers.get("content-type", "?"), len(resp.content))
    return resp.content


def process(client: httpx.Client, paper: config.Paper, sums: dict[str, str]) -> bool:
    """Ensure one paper is present and recorded. Returns True if it downloaded."""
    dest = config.PAPERS_DIR / paper.filename
    recorded = sums.get(paper.filename)

    if dest.exists():
        data = dest.read_bytes()
        actual = sha256_hex(data)
        if recorded is not None and actual != recorded:
            raise FetchError(
                f"{dest} exists but sha256 {actual} != recorded {recorded}. Refusing to "
                "overwrite. Investigate; to refetch, delete the file AND its CHECKSUMS line.")
        validate_pdf(data, paper)
        if recorded is None:
            log.warning("%s: present but unrecorded; recording its checksum", paper.filename)
            sums[paper.filename] = actual
            save_checksums(config.CHECKSUMS_FILE, sums)
        else:
            log.info("%s: already present, checksum matches; skipping", paper.filename)
        return False

    data = download(client, paper)
    validate_pdf(data, paper)
    actual = sha256_hex(data)
    if recorded is not None and actual != recorded:
        raise FetchError(
            f"{paper.filename}: arXiv served different bytes (sha256 {actual}) than the "
            f"recorded {recorded}. Not saving. Decide by hand whether to re-record.")
    write_bytes_atomic(dest, data)
    sums[paper.filename] = actual
    save_checksums(config.CHECKSUMS_FILE, sums)
    log.info("%s: saved, sha256=%s", paper.filename, actual)
    return True


def main() -> int:
    """Fetch all papers; return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        sums = load_checksums(config.CHECKSUMS_FILE)
        with httpx.Client(follow_redirects=True, timeout=config.DOWNLOAD_TIMEOUT_S,
                          headers={"User-Agent": config.USER_AGENT}) as client:
            for paper in config.PAPERS:
                if process(client, paper, sums):
                    time.sleep(config.DOWNLOAD_DELAY_S)
    except (FetchError, httpx.HTTPError) as exc:
        log.error("FAILED: %s", exc)
        return 1
    for paper in config.PAPERS:
        log.info("%s  %s", sums[paper.filename], paper.filename)
    log.info("OK: %d papers verified; checksums in %s", len(config.PAPERS), config.CHECKSUMS_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())