"""Self-RAG relevance grader: a binary yes/no verdict over the retrieved docs.

Returns True if the documents are relevant to refining the draft, False
otherwise. Parser is forgiving and defaults to True (keep / no web search) on
parse failure so a broken grader never derails the pipeline.
"""

from __future__ import annotations

import json
import re

from agents import prompts
from agents.llm import chat, deterministic_llm

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_relevant(text: str) -> bool:
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return True
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return True
    return str(obj.get("relevant", "yes")).lower().strip().startswith("y")


def grade_docs_relevant(draft: str, chunks: list[str]) -> bool:
    """True if the retrieved documents are relevant for refining `draft`."""
    if not chunks:
        return False
    numbered = "\n\n".join(f"[{i + 1}]\n{c[:4000]}" for i, c in enumerate(chunks))
    llm = deterministic_llm()
    user = prompts.GRADER_USER.format(draft=draft[:4000], chunks=numbered)
    text = chat(llm, prompts.GRADER_SYSTEM, user)
    return _parse_relevant(text)
