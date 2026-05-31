"""Shared types, constants, and small helpers for the refinement graph."""

from __future__ import annotations

import re
from typing import Literal, TypedDict

# Iteration caps for the three independent loops:
#   - outer/stance loop  (stance_check -> router): up to MAX_OUTER_ITERS passes
#   - grounding loop      (hallucination_check -> refine): up to MAX_GROUND_RETRIES
#     per outer pass (reset each pass), so worst case is OUTER x GROUND grounding
#     refines (3 x 2 = 6).
MAX_OUTER_ITERS = 3  # max router passes triggered by a failed stance check
MAX_GROUND_RETRIES = 2  # max grounding re-refines per outer pass
LOCAL_K = 4
WEB_K = 5  # Tavily max_results per web query
WEB_QUERIES = 3  # number of search queries the web-query planner generates per iteration (router may pick 3-5 terms)

# Tavily web-search allow-list. Empty list disables filtering.
#
# DELIBERATELY pro-Israel only. The pipeline produces a pro-Israel argument and
# is gated by a strict no-concession stance check; when the web arm retrieved
# from critical-leaning outlets (Guardian, HRW, Amnesty, OHCHR, UN, UNRWA...),
# grounded drafts kept importing those sources' fault-framing of Israeli conduct
# and failing the stance gate, so EVERY post fell through to the values-only
# force_regenerate fallback (no citations). Restricting retrieval to pro-Israel,
# Israeli-official, and Jewish-advocacy sources lets the evidence-based path
# produce grounded, CITED drafts that can actually pass the gate.
#
# Trade-off: evidence is one-sided by design and citations point to advocacy
# outlets — appropriate here (the goal is a pro-Israel argument, not a neutral
# brief), but a human reviewer should keep that in mind. To restore balance,
# re-add neutral primary/reference or critical outlets below.
WEB_ALLOWED_DOMAINS = [
    # Pro-Israel / Israel-advocacy think tanks & research
    "jcpa.org",            # Jerusalem Center for Public Affairs
    "fdd.org",             # Foundation for Defense of Democracies
    "washingtoninstitute.org",  # Washington Institute for Near East Policy (WINEP)
    "meforum.org",         # Middle East Forum
    "adl.org",             # Anti-Defamation League
    "ajc.org",             # American Jewish Committee
    # Media-accuracy watchdogs (pro-Israel)
    "camera.org",
    "honestreporting.com",
    # Jewish-affairs press & syndication
    "tabletmag.com",       # Jewish-affairs magazine
    "jns.org",             # Jewish News Syndicate
    # Israel-region press (English) — Israeli-perspective reporting
    "timesofisrael.com",
    "jpost.com",
    # Primary-source archive & Israeli official / legal / history channels
    "jewishvirtuallibrary.org",  # primary-source archive (AICE)
    "mfa.gov.il",          # Israeli Ministry of Foreign Affairs
    "gov.il",              # Israeli government
    "idf.il",              # Israel Defense Forces official
    "knesset.gov.il",      # Israeli parliament (laws, records)
    "jewishagency.org",    # Jewish Agency (aliyah/history)
]

Side = Literal["A", "B"]
RetrievalMode = Literal["local", "web"]


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
    # Accumulated Reflexion memory: every critique produced so far, in order,
    # so refine can avoid repeating past mistakes (deeper Reflexion).
    critique_history: list[str]

    grounded: bool
    hallucination_issues: list[str]

    # Independent loop counters (see MAX_OUTER_ITERS / MAX_GROUND_RETRIES).
    outer_iter: int       # router passes caused by a failed stance check
    ground_retries: int   # grounding re-refines within the current outer pass
    # Set when a stance failure sends us back to the router, telling refine the
    # next revision exists to make the argument clearly pro-Israel.
    regen_reason: str

    # New single-argument view (the merged self-RAG graph). After
    # eliminate_loser, `generation` mirrors the surviving side's current
    # argument; `documents` accumulates retrieved evidence/citations;
    # `web_search` flags that grade_docs found the local docs insufficient.
    post: str
    generation: str       # clean argument for display (citations stripped)
    generation_raw: str   # the cited draft kept for debugging/traceability
    sources: list[str]    # source URLs, surfaced separately for human review
    web_search: bool
    documents: list[str]

    history: list[dict]
    winner: Literal["A", "B"]
    pro_israel_reply: bool
    stance_reason: str
    # Traceability for the terminal force_regenerate pass: the strict-gate verdict
    # on the forced rewrite, recorded even though it no longer gates the output
    # (the rewrite is terminal-accepted). Lets a human reviewer see whether the
    # gate still objected and why.
    force_stance_pass: bool
    force_stance_reason: str
    final_scores: dict


def active_view(state: GraphState) -> tuple[Side, str, str, str]:
    """Return (side, current_arg, prev_arg, critique) for the active side."""
    side = state.get("active_side", "A")
    if side == "A":
        return "A", state["arg_a"], state.get("arg_a_prev", ""), state.get("critique_a", "")
    return "B", state["arg_b"], state.get("arg_b_prev", ""), state.get("critique_b", "")


_CITE_MARKER_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
_SOURCES_BLOCK_RE = re.compile(r"\n*#+\s*Sources.*\Z", re.IGNORECASE | re.DOTALL)


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


def strip_empty_sources(text: str) -> str:
    """Drop a trailing '### Sources' section when it lists no '[n]' entries.

    The refiner sometimes emits the Sources header with nothing under it (it
    argued from reasoning, not retrieved facts). The prompt says to omit the
    section in that case; this enforces it deterministically. A Sources section
    that DOES contain markers is left untouched.
    """
    if not text:
        return text
    match = _SOURCES_BLOCK_RE.search(text)
    if not match:
        return text
    block = match.group(0)
    if re.search(r"\[\d+\]", block):
        return text  # has real entries -> keep
    return text[: match.start()].rstrip()


_SOURCE_LINE_RE = re.compile(r"\[(\d+)\]\s*(\S+)")


def extract_sources(text: str) -> list[str]:
    """Return the source URLs listed in a trailing '### Sources' block, in order.

    Used for human-in-the-loop review: the displayed argument is cleaned of
    citations (see [[split_for_display]]), but the URLs the model relied on are
    surfaced separately so a person can verify the argument is grounded.
    """
    if not text:
        return []
    match = _SOURCES_BLOCK_RE.search(text)
    if not match:
        return []
    urls: list[str] = []
    for _, url in _SOURCE_LINE_RE.findall(match.group(0)):
        url = url.strip().rstrip(".,;")
        if url and url not in urls:
            urls.append(url)
    return urls


def split_for_display(text: str) -> tuple[str, list[str]]:
    """Split a finished draft into (clean_argument, source_urls).

    `clean_argument` has the inline [n] markers and the '### Sources' footer
    removed; `source_urls` is the list of URLs from that footer. Lets callers
    show readable prose while keeping the sources available for verification.
    """
    return strip_citations(text), extract_sources(text)