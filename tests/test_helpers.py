"""Unit tests for the deterministic helper functions.

These are the pure string/parse functions that the offline graph test
(test_graph_offline.py) stubs out: citation/source extraction, the
critique-shape guard, the router's JSON parsing, chunk dedup, and the ranker's
prompt trimming. They have real branching logic but no LLM/network/model
dependency, so they are cheap to test exhaustively and a good regression net.

Run: `uv run python tests/test_helpers.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph.chains.refiner import looks_like_critique
from agents.graph.chains.router import _parse_decision, _parse_queries
from agents.graph.nodes.hallucination_check import _is_citation_shaped
from agents.graph.nodes.retrieve import _dedupe
from agents.graph.state import (
    extract_sources,
    split_for_display,
    strip_citations,
    strip_empty_sources,
)

_failures: list[str] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def _eq(name: str, got, want) -> None:
    _check(name, got == want, f"got {got!r}, want {want!r}")


# --------------------------------------------------------------------------- #
# state.py: citation / sources handling
# --------------------------------------------------------------------------- #
def test_strip_citations() -> None:
    print("strip_citations")
    _eq("removes inline single marker", strip_citations("Israel withdrew in 2005 [1]."),
        "Israel withdrew in 2005.")
    _eq("removes multi-number marker", strip_citations("A claim [1, 2] stands."),
        "A claim stands.")
    _eq("removes trailing Sources block",
        strip_citations("Body text [1].\n\n### Sources\n[1] http://x.com"),
        "Body text.")
    _eq("empty string is passed through", strip_citations(""), "")
    _eq("text without markers is unchanged", strip_citations("Plain prose."), "Plain prose.")


def test_strip_empty_sources() -> None:
    print("strip_empty_sources")
    _eq("drops a Sources header with no entries",
        strip_empty_sources("Argument body.\n\n### Sources\n"),
        "Argument body.")
    _eq("keeps a Sources block that has [n] entries",
        strip_empty_sources("Argument body.\n\n### Sources\n[1] http://x.com"),
        "Argument body.\n\n### Sources\n[1] http://x.com")
    _eq("no Sources block -> unchanged", strip_empty_sources("Just prose."), "Just prose.")


def test_extract_sources() -> None:
    print("extract_sources")
    _eq("pulls URLs in order",
        extract_sources("Body [1][2].\n\n### Sources\n[1] http://a.com\n[2] http://b.com"),
        ["http://a.com", "http://b.com"])
    _eq("dedups repeated URLs",
        extract_sources("### Sources\n[1] http://a.com\n[2] http://a.com"),
        ["http://a.com"])
    _eq("strips trailing punctuation",
        extract_sources("### Sources\n[1] http://a.com."),
        ["http://a.com"])
    _eq("no Sources block -> empty list", extract_sources("Body [1] with no footer."), [])


def test_split_for_display() -> None:
    print("split_for_display")
    clean, urls = split_for_display("Claim [1].\n\n### Sources\n[1] http://a.com")
    _eq("clean text has no markers/footer", clean, "Claim.")
    _eq("urls extracted alongside", urls, ["http://a.com"])


# --------------------------------------------------------------------------- #
# refiner.py: critique-shape guard
# --------------------------------------------------------------------------- #
def test_looks_like_critique() -> None:
    print("looks_like_critique")
    _check("flags 'your draft should' editorial voice",
           looks_like_critique("Your draft should cite a source here."))
    _check("flags a leaked reflect header (**Missing**)",
           looks_like_critique("**Missing**\n- add the 1947 partition context"))
    _check("does NOT flag a genuine rebuttal that says 'you'",
           not looks_like_critique("You claim Israel controls Gaza's water, but in fact..."))
    _check("empty string is not critique-shaped", not looks_like_critique(""))


# --------------------------------------------------------------------------- #
# hallucination_check.py: citation-vs-fact issue classifier
# --------------------------------------------------------------------------- #
def test_is_citation_shaped() -> None:
    print("_is_citation_shaped")
    _check("marker like [2] -> citation issue", _is_citation_shaped("marker [2] points to the wrong chunk"))
    _check("'wrong-pointer' phrasing -> citation issue", _is_citation_shaped("wrong-pointer on the second claim"))
    _check("the word 'citation' -> citation issue", _is_citation_shaped("citation does not match the source"))
    _check("a fabricated fact -> NOT citation-shaped",
           not _is_citation_shaped("the 34,000 casualty figure is unsupported"))


# --------------------------------------------------------------------------- #
# router.py: LLM-output JSON parsing (with fallbacks)
# --------------------------------------------------------------------------- #
def test_parse_decision() -> None:
    print("_parse_decision")
    _eq("web mode keeps queries (capped at n)",
        _parse_decision('{"mode": "web", "queries": ["a", "b", "c", "d"]}', 3),
        ("web", ["a", "b", "c"]))
    _eq("local mode drops queries",
        _parse_decision('{"mode": "local", "queries": ["a", "b"]}', 3),
        ("local", []))
    _eq("unparseable text -> safe default (local, [])",
        _parse_decision("the model rambled with no json", 3),
        ("local", []))
    _eq("malformed json -> safe default",
        _parse_decision('{"mode": "web", queries: [}', 3),
        ("local", []))
    _eq("web with non-list queries -> empty",
        _parse_decision('{"mode": "web", "queries": "oops"}', 3),
        ("web", []))


def test_parse_queries() -> None:
    print("_parse_queries")
    _eq("extracts and caps queries",
        _parse_queries('{"queries": ["x", "y", "z", "w"]}', 2), ["x", "y"])
    _eq("drops blank entries", _parse_queries('{"queries": ["a", "  ", "b"]}', 5), ["a", "b"])
    _eq("no json -> empty", _parse_queries("nope", 3), [])


# --------------------------------------------------------------------------- #
# retrieve.py: chunk dedup
# --------------------------------------------------------------------------- #
def test_dedupe() -> None:
    print("_dedupe")
    a = "[url] http://x.com\nfirst copy"
    b = "[url] http://x.com\nsecond copy (same url)"
    c = "[url] http://y.com\nother"
    _eq("dedups by [url] header, keeps first occurrence", _dedupe([a, b, c]), [a, c])
    _eq("dedups headerless chunks by full content",
        _dedupe(["same", "same", "diff"]), ["same", "diff"])
    _eq("drops empty chunks", _dedupe(["", "keep", ""]), ["keep"])


def main() -> int:
    for t in (
        test_strip_citations,
        test_strip_empty_sources,
        test_extract_sources,
        test_split_for_display,
        test_looks_like_critique,
        test_is_citation_shaped,
        test_parse_decision,
        test_parse_queries,
        test_dedupe,
    ):
        t()

    print("\n" + ("ALL CHECKS PASSED" if not _failures else f"{len(_failures)} CHECK(S) FAILED"))
    return 0 if not _failures else 1


if __name__ == "__main__":
    sys.exit(main())
