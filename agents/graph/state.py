"""Shared types, constants, and small helpers for the refinement graph."""

from __future__ import annotations

import re
from typing import Literal, TypedDict

MAX_ITERS = 3
LOCAL_K = 4
WEB_K = 4
WEB_QUERIES = 3  # number of search queries the web-query planner generates per iteration
MAX_REFINE_RETRIES = 1  # how many times refine may rerun on ungrounded output before reverting
GPT_MODEL = "gpt-5.4-nano"

Side = Literal["A", "B"]
RetrievalMode = Literal["local", "web", "none"]


class GraphState(TypedDict, total=False):
    topic: str
    original_post: str

    arg_a: str
    arg_b: str
    arg_a_prev: str
    arg_b_prev: str

    iter_a: int
    iter_b: int
    converged_a: bool
    converged_b: bool

    active_side: Side
    retrieval_mode: RetrievalMode
    web_queries: list[str]
    retrieved: list[str]
    critique: str
    critique_a: str
    critique_b: str

    grounded: bool
    hallucination_issues: list[str]
    refine_retries_a: int
    refine_retries_b: int

    history: list[dict]
    winner: Literal["A", "B"]
    final_scores: dict


def active_view(state: GraphState) -> tuple[Side, str, str, str]:
    """Return (side, current_arg, prev_arg, critique) for the active side."""
    side = state.get("active_side", "A")
    if side == "A":
        return "A", state["arg_a"], state.get("arg_a_prev", ""), state.get("critique_a", "")
    return "B", state["arg_b"], state.get("arg_b_prev", ""), state.get("critique_b", "")


_CITE_MARKER_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
_SOURCES_BLOCK_RE = re.compile(r"\n+#+\s*Sources.*\Z", re.IGNORECASE | re.DOTALL)


def strip_citations(text: str) -> str:
    """Remove inline [n] markers and a trailing '### Sources' block.

    The Qwen pairwise ranker was trained on uncited arguments, so we feed it
    plain prose to avoid the surface form of citations confounding the score.
    """
    if not text:
        return text
    cleaned = _SOURCES_BLOCK_RE.sub("", text)
    cleaned = _CITE_MARKER_RE.sub("", cleaned)
    return cleaned.strip()