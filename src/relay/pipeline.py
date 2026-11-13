"""5-step pipeline: understand/rewrite -> retrieve -> decide (branch) ->
draft (streamed) -> verify grounding (loop: redraft once, then refuse).
No retries, no rate-limit handling — that's step 5, wrapping this function.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from relay import config
from relay.ingest import Chunk
from relay.llm_client import ChatResult, stream_chat
from relay.retrieve import Index, RetrievalResult, retrieve
from relay.tools import ToolError, calculator, extract_lookup_target, lookup_section

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.]+_p\d+_\d+)\]")
CALC_EXPR_RE = re.compile(r"\d[\d.\s()+\-*/]{2,}[\d)]")
AMBIGUOUS_STARTS = ("it ", "this ", "that ", "they ", "its ")


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float | None = None


@dataclass
class Trace:
    spans: list[Span] = field(default_factory=list)

    def start(self, name: str, clock: float) -> Span:
        s = Span(name, clock)
        self.spans.append(s)
        return s

    def end(self, span: Span, clock: float) -> None:
        span.end_ms = clock


@dataclass
class PipelineResult:
    request_id: str
    scenario: str
    source: str
    start_time: float
    first_token_ms: float | None
    total_ms: float
    status: str
    retries: int
    tokens_in: int | None
    tokens_out: int | None
    retrieval_setting: str
    answer_grade: str | None
    answer_text: str
    citations: list[str]
    refused: bool
    refusal_reason: str | None
    redraft_count: int
    branch_taken: str
    trace: Trace

    def csv_row(self) -> dict[str, object]:
        return {
            "request_id": self.request_id, "scenario": self.scenario, "source": self.source,
            "start_time": self.start_time, "first_token_ms": self.first_token_ms,
            "total_ms": self.total_ms, "status": self.status, "retries": self.retries,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "retrieval_setting": self.retrieval_setting,
            "answer_grade": self.answer_grade or "not_graded",
        }


def needs_rewrite(query: str) -> bool:
    """Rules first (brief). LLM only if this is True."""
    q = query.strip().lower()
    return len(q.split()) <= 2 or q.startswith(AMBIGUOUS_STARTS)


def _chat_status(result: ChatResult) -> str:
    if result.error_type in ("ReadTimeout", "ConnectTimeout", "PoolTimeout", "WriteTimeout"):
        return "timeout"
    if result.status == 429:
        return "rate_limited"
    if result.status == 200:
        return "ok"
    return "error"


async def _rewrite_query(client, query, api_key, base_url, model) -> str:
    messages = [
        {"role": "system", "content": "Rewrite the user's question into one clear, "
         "self-contained question. Do not answer it. Output only the rewritten "
         "question."},
        {"role": "user", "content": query},
    ]
    result = await stream_chat(client, messages, api_key=api_key, max_tokens=64,
                                base_url=base_url, model=model)
    if _chat_status(result) != "ok" or not result.text.strip():
        return query  # degrade: never block on a failed rewrite
    return result.text.strip()


def _decide_branch(query: str, retrieved: list[RetrievalResult]) -> str:
    """Code-decided per rule 9 — never the LLM's self-confidence."""
    if not retrieved or retrieved[0].score < (config.SCORE_THRESHOLD or 0.0):
        return "refuse"
    if CALC_EXPR_RE.search(query) and any(op in query for op in "+-*/"):
        return "tool:calculator"
    if extract_lookup_target(query) is not None:
        return "tool:lookup_section"
    return "answer"


def _build_answer_messages(query, context, extra_note: str | None = None) -> list[dict[str, str]]:
    context_block = "\n\n".join(
        f"[{c.chunk_id}] (paper {c.paper}v{c.version}, page {c.page}): {c.text}"
        for c in context
    )
    system = (
        "Answer ONLY using the context chunks below. Cite every claim with its "
        "exact chunk_id in brackets, e.g. [1706.03762v7_p4_0]. If the context "
        "does not answer the question, say exactly: 'Not found in the papers.' "
        "Never use outside knowledge.\n\nCONTEXT:\n" + context_block
    )
    if extra_note:
        system += "\n\n" + extra_note
    return [{"role": "system", "content": system}, {"role": "user", "content": query}]


def _grounding_ok(citations: list[str], allowed_ids: set[str]) -> bool:
    return bool(citations) and all(c in allowed_ids for c in citations)


async def run_pipeline(
    query: str, *, index: Index, all_chunks: list[Chunk],
    retrieval_setting_name: str, retrieval_setting: dict[str, Any],
    client: httpx.AsyncClient, api_key: str, base_url: str, model: str,
    source: str, scenario: str,
) -> PipelineResult:
    request_id = str(uuid.uuid4())
    start_time = time.time()
    t0 = time.perf_counter()
    trace = Trace()
    clock = lambda: (time.perf_counter() - t0) * 1000.0

    span = trace.start("understand", clock())
    eq = query
    if needs_rewrite(query):
        eq = await _rewrite_query(client, query, api_key, base_url, model)
    trace.end(span, clock())

    span = trace.start("retrieve", clock())
    retrieved = retrieve(eq, index, top_k=retrieval_setting["top_k"],
                          hybrid_alpha=config.HYBRID_ALPHA,
                          rerank_enabled=retrieval_setting["rerank_enabled"],
                          cross_encoder_model=config.CROSS_ENCODER_MODEL)
    trace.end(span, clock())

    span = trace.start("decide", clock())
    branch = _decide_branch(eq, retrieved)
    tool_context: list[Chunk] | None = None
    tool_note: str | None = None
    if branch == "tool:calculator":
        m = CALC_EXPR_RE.search(eq)
        tspan = trace.start("tool", clock())
        try:
            value = await asyncio.wait_for(asyncio.to_thread(calculator, m.group(0)),
                                            timeout=config.TOOL_TIMEOUT_S)
            tool_note = f"Calculator computed: {m.group(0).strip()} = {value}"
        except (ToolError, asyncio.TimeoutError) as exc:
            logger.warning("calculator failed (%s); degrading to answer", exc)
            branch = "answer"
        trace.end(tspan, clock())
    elif branch == "tool:lookup_section":
        target = extract_lookup_target(eq)
        tspan = trace.start("tool", clock())
        try:
            tool_context = await asyncio.wait_for(
                asyncio.to_thread(lookup_section, target, all_chunks),
                timeout=config.TOOL_TIMEOUT_S)
        except (ToolError, asyncio.TimeoutError) as exc:
            logger.warning("lookup_section failed (%s); degrading to answer", exc)
            branch = "answer"
        trace.end(tspan, clock())
    trace.end(span, clock())

    if branch == "refuse":
        return PipelineResult(
            request_id=request_id, scenario=scenario, source=source,
            start_time=start_time, first_token_ms=None, total_ms=clock(),
            status="ok", retries=0, tokens_in=None, tokens_out=None,
            retrieval_setting=retrieval_setting_name, answer_grade=None,
            answer_text="Not found in the papers.", citations=[], refused=True,
            refusal_reason="top retrieval score below SCORE_THRESHOLD",
            redraft_count=0, branch_taken="refuse", trace=trace,
        )

    context = tool_context if tool_context is not None else retrieved
    allowed_ids = {c.chunk_id for c in context}
    messages = _build_answer_messages(eq, context, extra_note=tool_note)

    span = trace.start("draft", clock())
    chat_result = await stream_chat(client, messages, api_key=api_key, max_tokens=512,
                                     base_url=base_url, model=model)
    trace.end(span, clock())

    redraft_count = 0
    citations = CITATION_RE.findall(chat_result.text)
    while not _grounding_ok(citations, allowed_ids) and redraft_count < config.MAX_REDRAFTS:
        redraft_count += 1
        span = trace.start(f"verify_redraft_{redraft_count}", clock())
        correction = ("Your previous answer cited a chunk_id not in the context, or "
                       "none at all. Cite ONLY the exact chunk_id(s) shown above, or "
                       "say 'Not found in the papers.'")
        rd_messages = _build_answer_messages(eq, context, extra_note=correction)
        chat_result = await stream_chat(client, rd_messages, api_key=api_key, max_tokens=512,
                                         base_url=base_url, model=model)
        citations = CITATION_RE.findall(chat_result.text)
        trace.end(span, clock())

    status = _chat_status(chat_result)
    refused, refusal_reason, answer_text = False, None, chat_result.text
    if status == "ok" and not _grounding_ok(citations, allowed_ids):
        refused = True
        refusal_reason = f"ungrounded after {redraft_count} redraft(s); hard cap reached"
        answer_text, citations = "Not found in the papers.", []

    usage = chat_result.usage or {}
    return PipelineResult(
        request_id=request_id, scenario=scenario, source=source, start_time=start_time,
        first_token_ms=chat_result.ttft_ms, total_ms=clock(), status=status, retries=0,
        tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"),
        retrieval_setting=retrieval_setting_name, answer_grade=None,
        answer_text=answer_text, citations=citations, refused=refused,
        refusal_reason=refusal_reason, redraft_count=redraft_count,
        branch_taken=branch, trace=trace,
    )