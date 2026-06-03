"""Stance check: classify a candidate reply as pro_israel / neutral / off-topic.

Returns {"stance": str, "reason": str}. Parser is forgiving and falls back to
the most permissive verdict ("pro_israel") on parse failure so a broken
checker never blocks the pipeline. When uncertain (verdict unrecognized but
the response is non-empty), the parser falls back to "neutral_needs_refine"
— a soft fail that triggers a cheap refinement pass rather than an expensive
regeneration.
"""

from __future__ import annotations

import json
import re

from agents import prompts
from agents.llm import chat, deterministic_llm

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_VALID_STANCES = {"pro_israel", "neutral_needs_refine", "off_topic_or_anti"}


def _parse_verdict(text: str) -> dict:
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return {"stance": "pro_israel", "reason": "", "parse_error": True}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"stance": "pro_israel", "reason": "", "parse_error": True}
    raw_stance = str(obj.get("stance", "")).strip().lower()
    if raw_stance in _VALID_STANCES:
        stance = raw_stance
    elif raw_stance:
        # Recognized something but not a known stance — soft-fail to refinement.
        stance = "neutral_needs_refine"
    else:
        stance = "pro_israel"
    return {
        "stance": stance,
        "reason": str(obj.get("reason", "")).strip(),
    }


def check_stance(post: str, draft: str) -> dict:
    """Return {'stance': str, 'reason': str} for the candidate reply."""
    llm = deterministic_llm()
    user = prompts.STANCE_CHECK_USER.format(post=post, draft=draft)
    text = chat(llm, prompts.STANCE_CHECK_SYSTEM, user)
    return _parse_verdict(text)
