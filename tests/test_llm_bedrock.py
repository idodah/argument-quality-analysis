"""Offline tests for the Bedrock LLM wrapper's response flattening.

`agents.llm._as_text` is the single chokepoint every chain's output passes
through, and `ChatBedrockConverse` can return `.content` as a string OR a list of
content blocks. A wrong flatten would silently corrupt every chain, so pin the
shapes here. No AWS — `_as_text` is pure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm import _as_text


def test_plain_string_passthrough():
    assert _as_text("hello") == "hello"


def test_list_of_text_blocks_concatenated():
    content = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
    assert _as_text(content) == "foobar"


def test_list_of_raw_strings():
    assert _as_text(["a", "b", "c"]) == "abc"


def test_non_text_blocks_dropped():
    content = [
        {"type": "reasoning_content", "reasoning": "thinking..."},  # no 'text' key
        {"type": "text", "text": "answer"},
    ]
    assert _as_text(content) == "answer"


def test_empty_and_none():
    assert _as_text([]) == ""
    assert _as_text(None) == ""
