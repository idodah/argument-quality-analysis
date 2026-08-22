"""Initial-draft chain: produces two distinct candidate refutations from the post."""

from __future__ import annotations

from agents import prompts
from agents.llm import chat, creative_llm


def _parse_two_responses(text: str) -> tuple[str, str]:
    """Split the model output into its two drafts on the '### Response A/B'
    markers. If the markers are missing, fall back to a blank-line split, then
    to a halfway cut — so a stray format never loses a whole draft.
    """
    marker_a = "### Response A"
    marker_b = "### Response B"
    if marker_a not in text or marker_b not in text:
        halves = text.split("\n\n\n", 1)
        if len(halves) == 2:
            return halves[0].strip(), halves[1].strip()
        mid = len(text) // 2
        return text[:mid].strip(), text[mid:].strip()
    a_start = text.index(marker_a) + len(marker_a)
    b_start = text.index(marker_b)
    arg_a = text[a_start:b_start].strip()
    arg_b = text[b_start + len(marker_b):].strip()
    return arg_a, arg_b


def generate_initial_pair(topic: str, post: str, regen_reason: str = "") -> tuple[str, str]:
    """Draft two distinct candidate refutations of the trope in `post`.

    On regeneration, the previous attempt's failure reason is folded into the
    prompt so the new draft is informed rather than blind — the model knows
    what specifically the prior draft got wrong (e.g. "stayed neutral, didn't
    advocate" or "went off-topic onto unrelated points").
    """
    if regen_reason:
        prior_note = (
            "### IMPORTANT: Prior attempts on this post failed for this reason:\n"
            f"  {regen_reason}\n"
            "Avoid that failure mode in BOTH of the two new responses below.\n\n"
        )
    else:
        prior_note = ""
    llm = creative_llm()
    user = prompts.INITIAL_GEN_USER.format(topic=topic, post=post, prior_attempt_note=prior_note)
    text = chat(llm, prompts.INITIAL_GEN_SYSTEM, user)
    return _parse_two_responses(text)