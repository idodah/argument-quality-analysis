"""Chain: forcibly rewrite a draft into a clearly pro-Israel reply.

Last-resort step used when the stance check keeps failing at the iteration cap.
"""

from __future__ import annotations

from agents import prompts
from agents.llm import chat, creative_llm


def force_pro_israel(post: str, draft: str) -> str:
    """Rewrite `draft` as a pro-Israel reply to `post`."""
    llm = creative_llm()
    user = prompts.FORCE_PRO_ISRAEL_USER.format(post=post, draft=draft)
    return chat(llm, prompts.FORCE_PRO_ISRAEL_SYSTEM, user)
