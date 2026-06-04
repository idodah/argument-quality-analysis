"""Shared types, constants, and small helpers for the refinement graph."""

from __future__ import annotations

import re
from typing import Literal, TypedDict

# Iteration caps. Four independent loops, each with its own budget:
#
#   - EARLY regeneration loop (early_stance_check -> generate_initial): the
#     cheap pre-refinement gate may regenerate up to MAX_EARLY_REGEN_ITERS
#     times PER GENERATION. Once that budget is spent the early gate stops
#     short-circuiting and hands the draft to the refinement path (the late
#     stance gate then decides what to do). The budget is reset whenever the
#     late gate starts a fresh generation, so the two regen loops compose
#     multiplicatively (see the LATE regeneration loop below).
#
#   - REFINEMENT loop (late stance_check -> router): up to MAX_REFINE_ITERS
#     passes per generation. Triggered when the argument is on-topic but
#     doesn't yet make a pro-Israel case. When this budget is spent, the late
#     stance gate escalates the verdict to off_topic_or_anti, which routes to
#     generate_initial via the LATE regeneration loop.
#
#   - LATE regeneration loop (late stance_check -> generate_initial): up to
#     MAX_LATE_REGEN_ITERS fresh starts driven by the late gate (either an
#     off_topic_or_anti verdict, or a refinement-budget-exhaustion escalation).
#     Each fresh start RESETS the early gate's budget, so every generation gets
#     its own full set of early regenerations; the loops compose
#     multiplicatively (up to (1 + MAX_LATE_REGEN_ITERS) generations, each able
#     to be early-regenerated up to MAX_EARLY_REGEN_ITERS times). Once both
#     gates' budgets are spent the late gate gives up.
#
#   - GROUNDING loop (hallucination_check -> refine): up to MAX_GROUND_RETRIES
#     re-refines per refinement pass.
MAX_REFINE_ITERS = 3       # stance_check -> router: 2 retries (1 first pass + 2)
MAX_EARLY_REGEN_ITERS = 2  # early_stance_check -> generate_initial: 2 retries
MAX_LATE_REGEN_ITERS = 1   # stance_check -> generate_initial: 1 retry
MAX_GROUND_RETRIES = 2     # hallucination_check -> refine: 2 retries per pass
LOCAL_K = 4
WEB_K = 5  # Tavily max_results per web query
WEB_QUERIES = 3  # number of search queries the web-query planner generates per iteration (router may pick 3-5 terms)

# Tavily web-search allow-list. Empty list disables filtering.
#
# DELIBERATELY pro-Israel only. The pipeline produces a pro-Israel argument and
# is gated by a strict no-concession stance check; when the web arm retrieved
# from critical-leaning outlets (Guardian, HRW, Amnesty, OHCHR, UN, UNRWA...),
# grounded drafts kept importing those sources' fault-framing of Israeli conduct
# and failing the stance gate, exhausting both budgets and giving up on every
# post. Restricting retrieval to pro-Israel, Israeli-official, and
# Jewish-advocacy sources lets the evidence-based path produce grounded, CITED
# drafts that can actually pass the gate.
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
RetrievalMode = Literal["local", "web", "none"]
Stance = Literal["pro_israel", "neutral_needs_refine", "off_topic_or_anti"]


class GraphState(TypedDict, total=False):
    topic: str
    original_post: str

    # TRANSIENT handoff slots between `generate_initial` and `eliminate_loser`
    # ONLY. After eliminate_loser writes the survivor to `argument`, these are
    # never read again — they remain in state for traceability but no node
    # downstream of eliminate_loser branches on them.
    arg_a: str
    arg_b: str

    # Single-argument view. `eliminate_loser` writes the surviving raw draft
    # here; `refine` overwrites it on each pass. The Qwen pairwise comparison
    # happens once on the two initial drafts in `eliminate_loser` — after
    # that, only `argument` is read by downstream nodes.
    argument: str
    # Refinement-pass counter (number of times `refine` ran on the current
    # generation; reset on regeneration). Used for trajectory/logging only.
    iter: int
    # The side eliminate_loser picked ("A" or "B"). Cosmetic — preserved for
    # the legacy `winner` field in `final_scores`; nothing reads it for control
    # flow.
    winner: Side

    # Routing decision stamped by early_stance_check for early_stance_router to
    # read ("generate_initial" | "router"). Avoids re-deriving the branch from
    # early_regen_iter, which the node has already incremented.
    early_action: str
    retrieval_mode: RetrievalMode
    web_queries: list[str]
    critique: str
    # Accumulated Reflexion memory: every critique produced so far, in order,
    # so refine can avoid repeating past mistakes (deeper Reflexion).
    critique_history: list[str]

    grounded: bool
    # True when the hallucination grader actually ran against retrieved factual
    # evidence; False when grounding was ASSUMED because the pass had no such
    # evidence to check (local/none retrieval). Keeps "verified grounded" and
    # "assumed grounded" distinguishable in final_scores.
    grounding_verified: bool
    hallucination_issues: list[str]

    # Loop counters (see MAX_*_ITERS / MAX_GROUND_RETRIES).
    # refine_iter, ground_retries, AND early_regen_iter reset each generation
    # (the late gate resets all three when it starts a fresh generation);
    # late_regen_iter persists across the whole run.
    refine_iter: int        # stance_check -> router refinement passes
    early_regen_iter: int   # early_stance_check -> generate_initial restarts
    late_regen_iter: int    # stance_check -> generate_initial restarts
    ground_retries: int     # grounding re-refines within the current refinement pass
    # Set when a stance failure sends us back to the router (or to
    # generate_initial), telling the next pass why the previous attempt failed.
    regen_reason: str
    # Count of consecutive refinement PASSES whose refine output tripped the
    # critique-shaped guard. When this reaches 2, the next pass escalates to
    # regeneration rather than burning the rest of the refinement budget on a
    # stuck model. `noop_streak_pass` records the refine_iter the streak was
    # last bumped for, so the grounding loop's repeated refine calls within one
    # pass count that pass at most once. Both reset together on regeneration.
    consecutive_noop_refines: int
    noop_streak_pass: int

    # Post + retrieval/citation pool.
    post: str
    generation: str       # clean argument for display (citations stripped)
    generation_raw: str   # the cited draft kept for debugging/traceability
    sources: list[str]    # source URLs, surfaced separately for human review
    web_search: bool
    documents: list[str]

    history: list[dict]
    # Ternary stance + the legacy boolean (kept for run.py / tests / external
    # consumers that read pro_israel_reply directly). pro_israel_reply is just
    # `stance == "pro_israel"`.
    stance: Stance
    pro_israel_reply: bool
    stance_reason: str
    # Set when both budgets are exhausted and the pipeline gives up rather than
    # ship a non-pro-Israel argument as if it were one.
    gave_up: bool
    give_up_reason: str
    final_scores: dict


def current_argument(state: GraphState) -> str:
    """Return the current working draft. Single-side; no A/B branching."""
    return state.get("argument", "")


_CITE_MARKER_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
# `\Z` anchors to end-of-string and `.*` is greedy under DOTALL, so this matches
# the LAST occurrence of `### Sources` (or any heading-level) onward — exactly
# what we want when the prompt puts the Sources footer at the very end. If the
# model misbehaves and emits a stray `### Sources` mid-text, this still strips
# from that point on (no real footer to preserve), which is the safer default.
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