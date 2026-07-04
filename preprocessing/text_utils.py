"""Text-cleaning helpers for the preprocessing pipeline.

These turn raw Reddit comment/post bodies and thread titles into the plain text
stored in the dataset: Markdown and HTML are unwrapped, links and quotes dropped,
and CMV-specific noise (delta markers, "Edit:" addenda, "CMV:" title tags) removed.
"""

import re

# Markdown / HTML / URL noise stripped from comment bodies.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_ENTITY_RE = re.compile(r"&\w+;")
_MARKDOWN_BOLD_ITALIC_RE = re.compile(r"\*{1,3}(.*?)\*{1,3}")   # **bold** / *italic* -> inner text
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")        # [label](url) -> label
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)   # leading "# " on a line
_WHITESPACE_RE = re.compile(r"\n{3,}")                          # 3+ blank lines -> one blank line

# A trailing "Edit: ..." paragraph the author appended after the fact.
_EDIT_PARAGRAPH_RE = re.compile(r"^\s*edit\b.*?(?=\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL | re.MULTILINE)

# The "CMV:" tag CMV titles carry, at the start or (occasionally) the end.
_TOPIC_PREFIX_RE = re.compile(r"^CMV:\s*", re.IGNORECASE)
_TOPIC_SUFFIX_RE = re.compile(r"[.\s]*cmv\s*\.?\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalize a raw Reddit comment/post body into plain text.

    Decodes HTML entities, unwraps Markdown (links, bold/italic, headers), drops
    URLs and blockquote (``>``) lines, and removes sentences mentioning the CMV
    delta mechanic (``!delta``, ``Δ``, ``deltabot``) so the delta signal can't
    leak into the argument text. Runs of blank lines are collapsed.
    """
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
    text = re.sub(r"_{3,}", "", text)
    return text.strip()


def strip_edit_paragraphs(text: str) -> str:
    """Remove trailing 'Edit:' paragraphs and collapse the whitespace they leave behind."""
    text = _EDIT_PARAGRAPH_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_topic(topic: str) -> str:
    """Strip the leading 'CMV:' prefix and any trailing 'cmv' marker from a thread title."""
    topic = _TOPIC_PREFIX_RE.sub("", topic).strip()
    topic = _TOPIC_SUFFIX_RE.sub("", topic).strip()
    return topic
