"""Notifier: send one ntfy push per generated response.

Set `NTFY_TOPIC` in `.env` and subscribe to it in the ntfy app / web — no bot,
no chat IDs. Optionally `NTFY_SERVER` (default https://ntfy.sh) and `NTFY_TOKEN`
(Bearer auth for protected topics). Note: ntfy.sh is a public relay; pick an
unguessable topic.
"""

from __future__ import annotations

import os

import requests

_MAX = 4000


def configured() -> bool:
    """True if a usable ntfy topic is configured in the environment."""
    return bool(os.environ.get("NTFY_TOPIC"))


def _chunks(text: str, size: int = _MAX):
    for i in range(0, len(text), size):
        yield text[i : i + size]


def send(text: str) -> None:
    """Push `text` to the configured ntfy topic, chunked under the size limit."""
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        raise RuntimeError(
            "ntfy not configured: set NTFY_TOPIC (and optionally NTFY_SERVER, "
            "default https://ntfy.sh) in your .env, then subscribe to that topic "
            "in the ntfy app."
        )
    server = (os.environ.get("NTFY_SERVER") or "https://ntfy.sh").rstrip("/")
    url = f"{server}/{topic}"
    headers = {
        "Title": "New CMV response",
        "Tags": "speech_balloon",
    }
    token = os.environ.get("NTFY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for part in _chunks(text):
        resp = requests.post(url, data=part.encode("utf-8"), headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"ntfy send failed ({resp.status_code}): {resp.text}")


def format_result(title: str, url: str, post_body: str, argument: str,
                  grounded: bool, pro_israel: bool, sources: list[str] | None = None) -> str:
    """Build the per-post message: original post + clean argument + sources.

    The argument itself carries no citations; the sources are listed in a
    separate trailing section for human-in-the-loop verification."""
    flags = []
    if not grounded:
        flags.append("NOT grounded")
    if not pro_israel:
        flags.append("NOT pro-Israel")
    flag_line = f"\n[warnings: {', '.join(flags)}]" if flags else ""
    sources = sources or []
    sources_block = (
        "\n\n--- SOURCES (for your review) ---\n"
        + "\n".join(f"{i}. {u}" for i, u in enumerate(sources, 1))
        if sources else ""
    )
    return (
        f"NEW CMV RESPONSE\n"
        f"Title: {title}\n"
        f"URL: {url}{flag_line}\n\n"
        f"--- ORIGINAL POST ---\n{post_body}\n\n"
        f"--- GENERATED ARGUMENT ---\n{argument}"
        f"{sources_block}"
    )
