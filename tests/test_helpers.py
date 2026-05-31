"""Unit tests for the deterministic helper functions.

These are the pure string/parse functions that the offline graph test
(test_graph_offline.py) stubs out: citation/source extraction, the
critique-shape guard, the router's JSON parsing, chunk dedup, and the issue
classifier. They have real branching logic but no LLM/network/model
dependency, so they are cheap to test exhaustively and a good regression net.

Run: `uv run pytest tests/test_helpers.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


# --------------------------------------------------------------------------- #
# state.py: citation / sources handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, want", [
    ("Israel withdrew in 2005 [1].", "Israel withdrew in 2005."),
    ("A claim [1, 2] stands.", "A claim stands."),
    ("Body text [1].\n\n### Sources\n[1] http://x.com", "Body text."),
    ("", ""),
    ("Plain prose.", "Plain prose."),
])
def test_strip_citations(text, want):
    assert strip_citations(text) == want


@pytest.mark.parametrize("text, want", [
    ("Argument body.\n\n### Sources\n", "Argument body."),
    # A Sources block that has [n] entries is left untouched.
    ("Argument body.\n\n### Sources\n[1] http://x.com",
     "Argument body.\n\n### Sources\n[1] http://x.com"),
    ("Just prose.", "Just prose."),
])
def test_strip_empty_sources(text, want):
    assert strip_empty_sources(text) == want


@pytest.mark.parametrize("text, want", [
    ("Body [1][2].\n\n### Sources\n[1] http://a.com\n[2] http://b.com",
     ["http://a.com", "http://b.com"]),
    # Repeated URLs are deduped, in first-seen order.
    ("### Sources\n[1] http://a.com\n[2] http://a.com", ["http://a.com"]),
    # Trailing punctuation is stripped.
    ("### Sources\n[1] http://a.com.", ["http://a.com"]),
    # A Sources block at the very start of the text is still detected
    # (regression: the regex once required a leading newline).
    ("### Sources\n[1] http://a.com", ["http://a.com"]),
    ("Body [1] with no footer.", []),
])
def test_extract_sources(text, want):
    assert extract_sources(text) == want


def test_split_for_display():
    clean, urls = split_for_display("Claim [1].\n\n### Sources\n[1] http://a.com")
    assert clean == "Claim."
    assert urls == ["http://a.com"]


# --------------------------------------------------------------------------- #
# refiner.py: critique-shape guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, expected", [
    ("Your draft should cite a source here.", True),
    ("**Missing**\n- add the 1947 partition context", True),   # leaked reflect header
    ("You claim Israel controls Gaza's water, but in fact...", False),  # genuine rebuttal
    ("", False),
])
def test_looks_like_critique(text, expected):
    assert looks_like_critique(text) is expected


# --------------------------------------------------------------------------- #
# hallucination_check.py: citation-vs-fact issue classifier
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("issue, expected", [
    ("marker [2] points to the wrong chunk", True),
    ("wrong-pointer on the second claim", True),
    ("citation does not match the source", True),
    ("the 34,000 casualty figure is unsupported", False),  # fabricated fact
])
def test_is_citation_shaped(issue, expected):
    assert _is_citation_shaped(issue) is expected


# --------------------------------------------------------------------------- #
# router.py: LLM-output JSON parsing (with fallbacks)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, n, want", [
    ('{"mode": "web", "queries": ["a", "b", "c", "d"]}', 3, ("web", ["a", "b", "c"])),
    ('{"mode": "local", "queries": ["a", "b"]}', 3, ("local", [])),  # local drops queries
    ("the model rambled with no json", 3, ("local", [])),           # unparseable -> safe default
    ('{"mode": "web", queries: [}', 3, ("local", [])),              # malformed json -> safe default
    ('{"mode": "web", "queries": "oops"}', 3, ("web", [])),         # non-list queries -> empty
])
def test_parse_decision(text, n, want):
    assert _parse_decision(text, n) == want


@pytest.mark.parametrize("text, n, want", [
    ('{"queries": ["x", "y", "z", "w"]}', 2, ["x", "y"]),
    ('{"queries": ["a", "  ", "b"]}', 5, ["a", "b"]),  # blanks dropped
    ("nope", 3, []),
])
def test_parse_queries(text, n, want):
    assert _parse_queries(text, n) == want


# --------------------------------------------------------------------------- #
# retrieve.py: chunk dedup
# --------------------------------------------------------------------------- #
def test_dedupe_by_url_header():
    a = "[url] http://x.com\nfirst copy"
    b = "[url] http://x.com\nsecond copy (same url)"
    c = "[url] http://y.com\nother"
    assert _dedupe([a, b, c]) == [a, c]  # dedup by [url], keep first


def test_dedupe_headerless_by_content():
    assert _dedupe(["same", "same", "diff"]) == ["same", "diff"]


def test_dedupe_drops_empty():
    assert _dedupe(["", "keep", ""]) == ["keep"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
