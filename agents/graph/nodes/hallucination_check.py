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
from agents.graph.state import GraphState, current_argument

_MARKER_RE = re.compile(r"\[\d+\]")


def _is_citation_shaped(issue: str) -> bool:
    """True if the issue is about a citation marker, not a fabricated fact."""
    if _MARKER_RE.search(issue):
        return True
    lowered = issue.lower()
    return "wrong-pointer" in lowered or "wrong pointer" in lowered or "citation" in lowered


def hallucination_check(state: GraphState) -> GraphState:
    draft = current_argument(state)
    history = state.get("history", [])
    pass_n = state.get("refine_iter", 0)

    # 'none' retrieval: no fresh factual evidence to ground against this pass.
    #
    # 'local' retrieval USED to be skipped too, on the reasoning that the local
    # corpus held only delta-awarded CMV comments — example arguments, not
    # source documents, so there was nothing to verify a factual claim against.
    # That is no longer true: the collection also holds authoritative reference
    # articles (USHMM / Wikipedia) that carry real URLs and ARE citable. Skipping
    # the grader whenever mode=='local' would bypass groundedness checking on
    # exactly the evidence the reference corpus exists to supply, so a local pass
    # is skipped only when it produced no citable (url-bearing) evidence.
    mode = state.get("retrieval_mode")
    documents = state.get("documents") or []
    has_citable_evidence = any(d.lstrip().startswith("[url] ") for d in documents)
    if mode == "none" or (mode == "local" and not has_citable_evidence):
        print(f"[hallucination_check] grounded=True ({mode} retrieval skipped — ASSUMED, not verified)")
        history = history + [{"refine_iter": pass_n, "stage": "hallucination_check",
                              "grounded": True, "grounding_verified": False, "skipped": f"{mode}_retrieval"}]
        # grounding_verified=False: 'grounded' here means "assumed grounded
        # because there was no factual evidence to check against", NOT "the
        # grader confirmed it". Downstream (finalize/final_scores) keeps the two
        # apart so an analysis can tell a verified pass from a skipped one.
        return {"grounded": True, "grounding_verified": False,
                "hallucination_issues": [], "history": history}

    evidence = state.get("documents", []) or []
    verdict = check_grounding(draft, evidence)
    issues = verdict.get("issues", [])

    # Lenient: only fabricated FACTS (non-citation issues) count as ungrounded.
    fact_issues = [i for i in issues if not _is_citation_shaped(i)]
    grounded = not fact_issues

    # Spend one grounding retry per ungrounded verdict (router resets to 0 each
    # outer pass), so route_after_hallucination can cap the grounding loop.
    ground_retries = state.get("ground_retries", 0) + (0 if grounded else 1)

    print(f"[hallucination_check] grounded={grounded} "
          f"(fact_issues={len(fact_issues)}, citation_issues={len(issues) - len(fact_issues)}, "
          f"ground_retries={ground_retries})")
    for i, issue in enumerate(fact_issues, 1):
        print(f"  [fact issue {i}] {issue}")

    history = history + [{"refine_iter": pass_n, "stage": "hallucination_check",
                          "grounded": grounded, "grounding_verified": True,
                          "issues": fact_issues, "ground_retries": ground_retries}]
    # grounding_verified=True: the grader actually ran against retrieved evidence.
    return {"grounded": grounded, "grounding_verified": True,
            "hallucination_issues": fact_issues,
            "ground_retries": ground_retries, "history": history}
