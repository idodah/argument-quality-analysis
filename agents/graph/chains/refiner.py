"""Refinement chain: rewrites a draft using critique + retrieved evidence."""

from __future__ import annotations

import re

from agents import prompts
from agents.llm import chat, creative_llm

# Phrases that mark the output as commentary ABOUT an argument rather than the
# argument itself — the failure mode where the refiner echoes the critique's
# editorial voice ("your draft should...", "a stronger approach is...").
_CRITIQUE_PHRASES = [
    r"\byour draft\b",
    r"\byour reply\b",
    r"\byour (?:next )?revision\b",
    r"\byour argument (?:should|needs|fails|doesn'?t)\b",
    r"\ba stronger (?:approach|argument|version) (?:is|would be)\b",
    r"\bthe (?:draft|argument|reply) (?:should|needs to|fails to|doesn'?t yet)\b",
    r"\byou(?:'re| are) missing\b",
    r"\bto change your view\b.{0,40}\btry\b",
    # The reflect prompt's planning voice talking ABOUT the response itself.
    # These only make sense if the writer is meta-commenting on an argument
    # under construction, not actually writing one.
    r"\bthe pro-israel (?:case|rebuttal|response|argument|reply) should\b",
    r"\b(?:israel|the pro-israel side) should be rebutted on\b",
    r"\b(?:section|paragraph) needs to be addressed in terms of\b",
]
_CRITIQUE_RE = re.compile("|".join(_CRITIQUE_PHRASES), re.IGNORECASE)

# Reflect prompt section headers that leaked verbatim into refine output. The
# reflect node uses '**Missing**' / '**Superfluous**' as headings for what to
# ADD vs CUT; if those headings appear at the start of a line in the refined
# output, the model treated its planning notes as part of the answer.
_REFLECT_HEADER_RE = re.compile(
    r"^\s*\*\*(?:Missing|Superfluous|Other)\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def looks_like_critique(text: str) -> bool:
    """True if the text reads as commentary about an argument, not the argument.

    Conservative: requires a clear editorial tell. A genuine rebuttal that
    happens to say 'you' is fine; this only fires on phrases that talk about
    the draft/reply/revision as an object to be improved, or on the reflect
    prompt's section headers ('**Missing**' / '**Superfluous**') appearing in
    the output (a leak of the planning structure into the answer).
    """
    if not text:
        return False
    if _REFLECT_HEADER_RE.search(text):
        return True
    return bool(_CRITIQUE_RE.search(text))


def refine_draft(
    topic: str,
    post: str,
    draft: str,
    critique_history: list[str],
    evidence_chunks: list[str],
    fix_notes: str = "",
) -> str:
    evidence = "\n\n---\n\n".join(evidence_chunks) if evidence_chunks else "(no retrieval this iteration)"
    history = "\n\n".join(critique_history) if critique_history else "(no prior critiques)"
    llm = creative_llm()
    user = prompts.REFINE_USER.format(
        topic=topic,
        post=post,
        draft=draft,
        critique_history=history,
        fix_notes=fix_notes or "(none — apply the critique history)",
        evidence=evidence,
    )
    # No retry on critique-shaped output: the refine node already discards such
    # output deterministically and keeps the current draft (see refine.py), so a
    # second LLM call here would be redundant.
    return chat(llm, prompts.REFINE_SYSTEM, user)