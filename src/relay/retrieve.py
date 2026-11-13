"""Hybrid BM25 + embedding retrieval, with optional cross-encoder rerank.

Scores are min-max normalized within the candidate pool before combining —
NOT calibrated against any absolute threshold. SCORE_THRESHOLD (refusal
cutoff) stays undecided here; that's step 4's job, done in code per rule 9.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from relay.ingest import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """Rule 8's schema, with score filled in at query time."""

    chunk_id: str
    text: str
    paper: str
    version: int
    page: int
    section: str | None
    score: float


class Index:
    """Holds a fixed chunk set plus its BM25 and embedding indexes."""

    def __init__(self, chunks: list[Chunk], embedding_model: str) -> None:
        if not chunks:
            raise ValueError("cannot build an index over zero chunks")
        self.chunks = chunks
        self._bm25 = BM25Okapi([c.text.lower().split() for c in chunks])
        self._encoder = SentenceTransformer(embedding_model)
        self._embeddings = self._encoder.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        )
        logger.info("built index: %d chunks, embedding model=%s", len(chunks), embedding_model)

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(query.lower().split()))

    def _embedding_scores(self, query: str) -> np.ndarray:
        q_vec = self._encoder.encode([query], normalize_embeddings=True)[0]
        return self._embeddings @ q_vec  # cosine sim, since both are normalized


def _min_max(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def retrieve(
    query: str,
    index: Index,
    top_k: int,
    hybrid_alpha: float,
    rerank_enabled: bool,
    cross_encoder_model: str | None = None,
    rerank_pool_multiplier: int = 3,
) -> list[RetrievalResult]:
    """Return the top_k chunks for `query`, per the given retrieval setting."""
    bm25_raw = index._bm25_scores(query)
    embed_raw = index._embedding_scores(query)
    hybrid = hybrid_alpha * _min_max(embed_raw) + (1 - hybrid_alpha) * _min_max(bm25_raw)

    pool_size = top_k * rerank_pool_multiplier if rerank_enabled else top_k
    pool_size = min(pool_size, len(index.chunks))
    pool_idx = np.argsort(-hybrid)[:pool_size]

    if rerank_enabled:
        if not cross_encoder_model:
            raise ValueError("rerank_enabled requires cross_encoder_model")
        cross_encoder = CrossEncoder(cross_encoder_model)
        pairs = [(query, index.chunks[i].text) for i in pool_idx]
        rerank_scores = np.asarray(cross_encoder.predict(pairs))
        order = np.argsort(-rerank_scores)[:top_k]
        final_idx = pool_idx[order]
        final_scores = rerank_scores[order]
    else:
        final_idx = pool_idx[:top_k]
        final_scores = hybrid[final_idx]

    results = []
    for i, score in zip(final_idx, final_scores):
        c = index.chunks[i]
        results.append(
            RetrievalResult(
                chunk_id=c.chunk_id, text=c.text, paper=c.paper,
                version=c.version, page=c.page, section=c.section,
                score=float(score),
            )
        )
    return results