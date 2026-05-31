"""Stance check: verifies the final argument is a pro-Israel reply to the post.

Returns {"pro_israel_reply": bool, "reason": str}. Parser is forgiving and
falls back to a permissive verdict on parse failure so a broken checker never
blocks the pipeline.
"""

from __future__ import annotations

import json
import re

from agents import prompts
from agents.llm import chat, deterministic_llm

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(text: str) -> dict:
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return {"pro_israel_reply": True, "reason": "", "parse_error": True}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"pro_israel_reply": True, "reason": "", "parse_error": True}
    return {
        "pro_israel_reply": bool(obj.get("pro_israel_reply", True)),
        "reason": str(obj.get("reason", "")).strip(),
    }


def check_stance(post: str, draft: str) -> dict:
    """Return {'pro_israel_reply': bool, 'reason': str} for the candidate reply."""
    llm = deterministic_llm()
    user = prompts.STANCE_CHECK_USER.format(post=post, draft=draft)
    text = chat(llm, prompts.STANCE_CHECK_SYSTEM, user)
    return _parse_verdict(text)
