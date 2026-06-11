"""Read r/ChangeMyView posts from the public Atom feed.

`/r/changemyview/new/.rss` is served to unauthenticated clients (unlike the
`.json` API and the data API, which are blocked / approval-gated), so it's the
way to read the REAL r/changemyview from any machine with no credentials. It
carries the newest ~25 posts; poll it on a schedule (see harvester/poll.py).

`fetch_from_rss()` yields `Post`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterator

SUBREDDIT = "changemyview"
RSS_URL = f"https://www.reddit.com/r/{SUBREDDIT}/new/.rss"
_ATOM = "{http://www.w3.org/2005/Atom}"
# Reddit blocks empty/generic agents; a descriptive UA is also just polite.
_DEFAULT_UA = "python:argument-quality-harvester:1.0 (CMV rss reader)"


@dataclass
class Post:
    id: str
    title: str
    body: str
    url: str
    created_utc: float


class _HTMLToText(HTMLParser):
    """Minimal HTML -> plain text. Reddit's RSS `content` is rendered HTML; we
    only need readable text for the classifier and notifications, so we drop tags
    and turn block elements into newlines. Stdlib only — no bs4/html2text dep."""

    _BLOCKS = {"p", "div", "br", "li", "blockquote", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._BLOCKS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._BLOCKS:
            self._parts.append("\n")

    def handle_data(self, data):
        self._parts.append(data)

    def text(self) -> str:
        # Collapse runs of blank lines and trim.
        raw = "".join(self._parts)
        lines = [ln.strip() for ln in raw.splitlines()]
        out: list[str] = []
        for ln in lines:
            if ln or (out and out[-1]):
                out.append(ln)
        return "\n".join(out).strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLToText()
    parser.feed(html or "")
    return parser.text()


def _post_id_from_link(link: str) -> str:
    """Extract the reddit base36 post id from a permalink
    (.../comments/<id>/slug/). Falls back to the whole link if not matched."""
    parts = [p for p in (link or "").split("/") if p]
    if "comments" in parts:
        i = parts.index("comments")
        if i + 1 < len(parts):
            return f"t3_{parts[i + 1]}"
    return link or ""


def fetch_from_rss(limit: int = 25, url: str = RSS_URL) -> Iterator[Post]:
    """Yield recent r/changemyview posts from the public Atom feed.

    No credentials needed — Reddit serves `.rss` to unauthenticated clients. The
    feed carries the newest ~25 posts; `limit` caps how many we yield. Skips
    entries with an empty body (CMV requires one).
    """
    import xml.etree.ElementTree as ET
    from datetime import datetime

    import requests

    from harvester.fediverse.base import assert_safe_url

    assert_safe_url(url)  # SSRF guard (blocks internal/link-local hosts on AWS)
    headers = {"User-Agent": os.environ.get("REDDIT_USER_AGENT") or _DEFAULT_UA}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"RSS fetch failed: {e}") from e

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        raise RuntimeError(f"RSS feed was not valid XML: {e}") from e

    yielded = 0
    for entry in root.findall(f"{_ATOM}entry"):
        if yielded >= limit:
            break
        title_el = entry.find(f"{_ATOM}title")
        content_el = entry.find(f"{_ATOM}content")
        link_el = entry.find(f"{_ATOM}link")
        updated_el = entry.find(f"{_ATOM}updated")

        title = (title_el.text or "").strip() if title_el is not None else ""
        body = _html_to_text(content_el.text if content_el is not None else "")
        if not body:
            continue  # removed / link-only post; CMV needs a body
        link = link_el.get("href") if link_el is not None else ""

        created = 0.0
        if updated_el is not None and updated_el.text:
            try:
                created = datetime.fromisoformat(updated_el.text).timestamp()
            except ValueError:
                created = 0.0

        yield Post(
            id=_post_id_from_link(link),
            title=title,
            body=body,
            url=link,
            created_utc=created,
        )
        yielded += 1

