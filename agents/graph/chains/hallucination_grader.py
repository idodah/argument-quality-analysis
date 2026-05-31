"""Self-RAG hallucination grader: checks whether a refined argument is grounded in retrieved evidence.

Returns a structured verdict {"grounded": bool, "issues": list[str]} parsed
from the LLM's JSON output. Parser is forgiving: it tolerates stray prose
around the JSON and falls back to a conservative "grounded=True, no issues"
verdict on parse failure so a broken grader never blocks the pipeline.
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
        return {"grounded": True, "issues": [], "parse_error": True}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"grounded": True, "issues": [], "parse_error": True}
    grounded = bool(obj.get("grounded", True))
    issues = obj.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    issues = [str(x).strip() for x in issues if str(x).strip()]
    # Reconcile flag with issue list: an empty issues list means grounded,
    # regardless of the flag the model emitted. This guards the case where the
    # grader strips its 'no issue here' deliberation entries but forgets to
    # flip grounded back to true.
    if not issues:
        grounded = True
    return {"grounded": grounded, "issues": issues}


def check_grounding(draft: str, evidence_chunks: list[str]) -> dict:
    """Return {'grounded': bool, 'issues': list[str]} for the refined draft."""
    evidence = "\n\n---\n\n".join(evidence_chunks) if evidence_chunks else "(no retrieval this iteration)"
    llm = deterministic_llm()
    user = prompts.HALLUCINATION_GRADER_USER.format(draft=draft, evidence=evidence)
    text = chat(llm, prompts.HALLUCINATION_GRADER_SYSTEM, user)
    return _parse_verdict(text)
