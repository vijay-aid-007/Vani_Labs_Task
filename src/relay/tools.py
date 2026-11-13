"""Tools the pipeline's branch step can call: lookup_section and calculator.
Both are pure, local, synchronous — the caller wraps them with asyncio.wait_for
for the timeout the brief requires; the timeout here is about protecting
against malformed input, not network latency.
"""
from __future__ import annotations

import ast
import operator
import re

from relay.ingest import Chunk

SECTION_RE = re.compile(r"section\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
PAGE_RE = re.compile(r"page\s+(\d+)", re.IGNORECASE)

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


class ToolError(Exception):
    """Raised on bad input; caller decides what degrade-path follows."""


def extract_lookup_target(query: str) -> dict[str, int] | None:
    sec = SECTION_RE.search(query)
    if sec:
        return {"section": sec.group(1)}
    page = PAGE_RE.search(query)
    if page:
        return {"page": int(page.group(1))}
    return None


def lookup_section(target: dict[str, object], chunks: list[Chunk]) -> list[Chunk]:
    """Page lookup only. Section-number lookup is unsupported: `Chunk.section`
    is always None (step 3), so rather than guess, a 'section X' request
    degrades to 'no match found' — handled by the caller, not here."""
    if "page" not in target:
        raise ToolError("section-number lookup unsupported (Chunk.section is always None)")
    matches = [c for c in chunks if c.page == target["page"]]
    if not matches:
        raise ToolError(f"no chunks found for page {target['page']}")
    return matches


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.operand))
    raise ToolError(f"unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> float:
    """+-*/**, parens only. No names, no calls, no attributes — cannot
    execute arbitrary code, unlike a bare eval()."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse {expression!r}: {exc}") from exc
    return _safe_eval_node(tree.body)