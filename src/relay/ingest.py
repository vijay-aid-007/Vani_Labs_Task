"""PDF ingestion: extract text per page, chunk per page, attach location metadata.

Chunking deliberately never crosses a page boundary, so `page` in rule 8's
chunk schema is always exact. Trade-off: a sentence that straddles a page
break gets cut across two chunks instead of kept whole — accepted for
location accuracy, flagged in WRITE-UP NOTES below.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import fitz  # PyMuPDF

from relay import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of text, per rule 8's schema (minus score)."""

    chunk_id: str
    text: str
    paper: str          # arxiv_id, e.g. "1706.03762"
    version: int
    page: int            # 1-indexed
    section: str | None  # left unset: heading detection not attempted


def extract_pages(pdf_path) -> list[tuple[int, str]]:
    """Return [(page_number_1_indexed, page_text), ...]."""
    doc = fitz.open(pdf_path)
    try:
        pages: list[tuple[int, str]] = []
        for i in range(doc.page_count):
            page_text = doc[i].get_text("text")
            pages.append((i + 1, str(page_text)))
    finally:
        doc.close()
    logger.info("extracted %d pages from %s", len(pages), pdf_path)
    return pages


def chunk_page_text(text: str, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    """Slide a char window over `text`, snapping to whitespace so words aren't split.

    Guarantees `start` strictly advances every iteration: if the whitespace
    snap would make the window so short that overlap subtraction stalls or
    reverses progress, the snap is rejected and a fixed-size step is used
    instead. Without this guard, a word-boundary near `start` can produce an
    infinite/near-infinite loop of near-duplicate tiny chunks (MemoryError).
    """
    if overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    min_advance = chunk_size_chars - overlap_chars  # guaranteed step when no snap applies
    while start < n:
        end = min(start + chunk_size_chars, n)
        if end < n:
            snap = text.rfind(" ", start, end)
            # Only accept the snap if it still leaves a sane-sized window;
            # otherwise keep the unsnapped `end` (word may split, rare, acceptable).
            if snap > start and (snap - start) >= min_advance:
                end = snap
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - overlap_chars
        if next_start <= start:  # safety net: never stall or go backward
            next_start = start + min_advance
        start = next_start
    return chunks


def ingest_papers(
    papers: tuple[config.Paper, ...] = config.PAPERS,
    chunk_size_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[Chunk]:
    """Extract + chunk all papers. Raises if a PDF is missing or chunking isn't configured."""
    chunk_size_chars = config.CHUNK_SIZE_CHARS if chunk_size_chars is None else chunk_size_chars
    overlap_chars = config.CHUNK_OVERLAP_CHARS if overlap_chars is None else overlap_chars
    if chunk_size_chars is None or overlap_chars is None:
        raise ValueError("CHUNK_SIZE_CHARS / CHUNK_OVERLAP_CHARS not configured in config.py")

    all_chunks: list[Chunk] = []
    for paper in papers:
        pdf_path = config.PAPERS_DIR / paper.filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"missing {pdf_path}; re-run step 0 download")
        for page_num, page_text in extract_pages(pdf_path):
            pieces = chunk_page_text(page_text, chunk_size_chars, overlap_chars)
            for idx, piece in enumerate(pieces):
                all_chunks.append(
                    Chunk(
                        chunk_id=f"{paper.arxiv_id}v{paper.version}_p{page_num}_{idx}",
                        text=piece,
                        paper=paper.arxiv_id,
                        version=paper.version,
                        page=page_num,
                        section=None,
                    )
                )
    logger.info("ingested %d chunks across %d papers", len(all_chunks), len(papers))
    return all_chunks