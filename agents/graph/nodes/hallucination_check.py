"""Node: Self-RAG hallucination check on the just-refined draft (binary).

Grades the refined draft yes/no for grounding in the retrieved evidence and
writes `grounded` to the state. When ungrounded, it increments `ground_retries`
(the per-outer-pass grounding budget). Routing (route_after_hallucination)
decides what happens next:
  - grounded=False & budget left -> loop back to `refine` (re-refine)
  - grounded=False & budget spent -> give up grounding, go to stance check
  - grounded=True                 -> continue to the stance check

Leniency ("less strict"):
  - Local CMV retrieval is SKIPPED: those chunks are persuasive example
    arguments (other people's deltas), not factual source documents, so the
    grader would always claim the cited facts aren't present. Treated as grounded.
  - Citation-marker problems (mis-pointed / unsupported [n] markers) are treated
    as grounded. Only a genuinely fabricated FACT counts as ungrounded.
"""

from __future__ import annotations

import re

from agents.graph.chains.hallucination_grader import check_grounding
from agents.graph.state import GraphState

_MARKER_RE = re.compile(r"\[\d+\]")


def _is_citation_shaped(issue: str) -> bool:
    """True if the issue is about a citation marker, not a fabricated fact."""
    if _MARKER_RE.search(issue):
        return True
    lowered = issue.lower()
    return "wrong-pointer" in lowered or "wrong pointer" in lowered or "citation" in lowered


def hallucination_check(state: GraphState) -> GraphState:
    side = state["active_side"]
    draft = state["arg_a"] if side == "A" else state["arg_b"]
    history = state.get("history", [])
    pass_n = state.get("outer_iter", 0)

    # Local retrieval: evidence is example arguments, not source docs -> grounded.
    if state.get("retrieval_mode") == "local":
        print(f"[hallucination_check] side={side} grounded=True (local retrieval skipped)")
        history = history + [{"side": side, "outer_iter": pass_n, "stage": "hallucination_check",
                              "grounded": True, "skipped": "local_retrieval"}]
        return {**state, "grounded": True, "hallucination_issues": [], "history": history}

    evidence = state.get("documents", []) or state.get("retrieved", []) or []
    verdict = check_grounding(draft, evidence)
    issues = verdict.get("issues", [])

    # Lenient: only fabricated FACTS (non-citation issues) count as ungrounded.
    fact_issues = [i for i in issues if not _is_citation_shaped(i)]
    grounded = not fact_issues

    # Spend one grounding retry per ungrounded verdict (router resets to 0 each
    # outer pass), so route_after_hallucination can cap the grounding loop.
    ground_retries = state.get("ground_retries", 0) + (0 if grounded else 1)

    print(f"[hallucination_check] side={side} grounded={grounded} "
          f"(fact_issues={len(fact_issues)}, citation_issues={len(issues) - len(fact_issues)}, "
          f"ground_retries={ground_retries})")
    for i, issue in enumerate(fact_issues, 1):
        print(f"  [fact issue {i}] {issue}")

    history = history + [{"side": side, "outer_iter": pass_n, "stage": "hallucination_check",
                          "grounded": grounded, "issues": fact_issues, "ground_retries": ground_retries}]
    return {**state, "grounded": grounded, "hallucination_issues": fact_issues,
            "ground_retries": ground_retries, "history": history}
