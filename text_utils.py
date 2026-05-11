"""
Text cleaning and comment combination utilities for the Webis-CMV-20 dataset.
"""

import re

import tiktoken

_OPENAI_EMBED_MODEL = "text-embedding-3-small"

_tiktoken_enc = tiktoken.encoding_for_model(_OPENAI_EMBED_MODEL)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_ENTITY_RE = re.compile(r"&\w+;")
_MARKDOWN_BOLD_ITALIC_RE = re.compile(r"\*{1,3}(.*?)\*{1,3}")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\n{3,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_EXCEL_MAX_CHARS = 32767
_MIN_SENTENCE_WORDS = 4


def count_tokens(text: str) -> int:
    return len(_tiktoken_enc.encode(text))


def clean_text(text: str) -> str:
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    text = _HTML_ENTITY_RE.sub(" ", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("", text)
    text = _MARKDOWN_BOLD_ITALIC_RE.sub(r"\1", text)
    text = _MARKDOWN_HEADER_RE.sub("", text)

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(>\s*)+", stripped):
            continue
        stripped = re.sub(
            r"[^.!?]*(?:!delta|Δ|∆|deltabot|\bdelta\b)[^.!?]*[.!?]?",
            "",
            stripped,
            flags=re.IGNORECASE,
        ).strip()
        if stripped:
            cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    text = _WHITESPACE_RE.sub("\n\n", text)
    return text.strip()
