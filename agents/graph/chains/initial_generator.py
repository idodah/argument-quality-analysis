"""Initial-draft chain: produces two distinct pro-Israel responses from the post."""

from __future__ import annotations

from agents import prompts
from agents.graph.llm import chat, creative_llm


def _parse_two_responses(text: str) -> tuple[str, str]:
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


def generate_initial_pair(topic: str, post: str) -> tuple[str, str]:
    """Draft two distinct pro-Israel responses to `post`."""
    llm = creative_llm()
    user = prompts.INITIAL_GEN_USER.format(topic=topic, post=post)
    text = chat(llm, prompts.INITIAL_GEN_SYSTEM, user)
    return _parse_two_responses(text)