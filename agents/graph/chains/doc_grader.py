"""Self-RAG relevance grader: a per-chunk yes/no verdict over retrieved docs.

`grade_docs_relevant` returns the SUBSET of chunks the model judged relevant to
refining the draft, in their original order. This is the canonical Self-RAG
move: keep the good chunks, drop only the noise — rather than collapsing the
whole pool to a single keep/drop verdict.

Parser is forgiving and defaults to keeping a chunk on parse failure, so a
broken grader never silently empties the citation pool.
"""

from __future__ import annotations

import json
import re

from agents import prompts
from agents.llm import chat, deterministic_llm

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_relevant_indices(text: str, n: int) -> list[int]:
    """Parse {"relevant": [1, 3, ...]} -> zero-based indices (1-based in JSON).

    On any parse failure, default to keeping ALL chunks (return every index):
    a broken grader should not quietly discard retrieved evidence.
    """
    all_indices = list(range(n))
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return all_indices
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return all_indices
    raw = obj.get("relevant")
    if not isinstance(raw, list):
        return all_indices
    kept: list[int] = []
    for v in raw:
        try:
            i = int(v) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in kept:
            kept.append(i)
    return kept


def grade_docs_relevant(draft: str, chunks: list[str]) -> list[str]:
    """Return the relevant subset of `chunks` (original order) for `draft`."""
    if not chunks:
        return []
    numbered = "\n\n".join(f"[{i + 1}]\n{c[:4000]}" for i, c in enumerate(chunks))
    llm = deterministic_llm()
    user = prompts.GRADER_USER.format(draft=draft[:4000], chunks=numbered)
    text = chat(llm, prompts.GRADER_SYSTEM, user)
    kept = _parse_relevant_indices(text, len(chunks))
    return [chunks[i] for i in kept]
