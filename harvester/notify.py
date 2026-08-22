"""Notifier: send one push per generated response.

Two backends, selected by `NOTIFY_BACKEND` in `.env`:

  - "telegram" (default when configured) — see `harvester.notify_telegram`.
    Handles long arguments properly: text up to 4096 chars, anything longer as
    a .md document, so nothing is cut. Delivers only to your chat id.
  - "ntfy" — set `NTFY_TOPIC` and subscribe in the ntfy app. Optionally
    `NTFY_SERVER` (default https://ntfy.sh) and `NTFY_TOKEN` (Bearer auth).

    Caveat that motivated the Telegram backend: ntfy caps a notification body
    at 4096 bytes, so a long argument is split into several independent pushes
    that arrive unordered and cut mid-sentence. ntfy.sh is also a public relay.

With no `NOTIFY_BACKEND` set, whichever backend is configured wins (Telegram
first); `send()` raises with setup instructions if neither is.
"""

from __future__ import annotations

import os

import requests

from harvester import notify_telegram

_MAX = 4000


def ntfy_configured() -> bool:
    """True if a usable ntfy topic is configured in the environment."""
    return bool(os.environ.get("NTFY_TOPIC"))


def active_backend() -> str | None:
    """Which backend send() would use: 'telegram', 'ntfy', or None.

    An explicit NOTIFY_BACKEND wins (and is honoured even if unconfigured, so a
    typo surfaces as that backend's setup error rather than silently falling
    back to the other one). Otherwise Telegram is preferred when configured,
    since it is the backend that delivers long arguments intact.
    """
    choice = (os.environ.get("NOTIFY_BACKEND") or "").strip().lower()
    if choice:
        return choice
    if notify_telegram.configured():
        return "telegram"
    if ntfy_configured():
        return "ntfy"
    return None


def configured() -> bool:
    """True if any notifier backend is usable."""
    return notify_telegram.configured() or ntfy_configured()


def send(text: str, click_url: str | None = None) -> None:
    """Push `text` via the active backend.

    If `click_url` is given the notification gains an "Open post" button
    pointing at the source post.
    """
    backend = active_backend()
    if backend == "telegram":
        notify_telegram.send(text, click_url=click_url)
        return
    if backend == "ntfy":
        _send_ntfy(text, click_url=click_url)
        return
    raise RuntimeError(
        "No notifier configured. Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID "
        "(recommended — long arguments arrive whole), or NTFY_TOPIC, in your "
        ".env. Select one explicitly with NOTIFY_BACKEND=telegram|ntfy."
    )


def _chunks(text: str, size: int = _MAX):
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _send_ntfy(text: str, click_url: str | None = None) -> None:
    """Push `text` to the configured ntfy topic, chunked under the size limit.

    NOTE: chunking splits a long argument across several independent
    notifications, which arrive unordered and cut mid-sentence. This is the
    limitation the telegram backend exists to avoid; prefer it for long output.
    """
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
        "Title": "New trope refutation",
        "Tags": "speech_balloon",
    }
    # Only http(s) links are valid ntfy click/action targets; skip anything else
    # (e.g. an empty url) so we never send a malformed header.
    if click_url and click_url.startswith(("http://", "https://")):
        headers["Click"] = click_url
        headers["Actions"] = f"view, Open post, {click_url}"
    token = os.environ.get("NTFY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for part in _chunks(text):
        resp = requests.post(url, data=part.encode("utf-8"), headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"ntfy send failed ({resp.status_code}): {resp.text}")


def format_result(title: str, url: str, post_body: str, argument: str,
                  grounded: bool, pro_israel: bool, sources: list[str] | None = None) -> str:
    """Build the per-post message: original post + argument + a trailing sources
    list (the argument carries no inline citations; sources are separated out for
    human review).

    `pro_israel` keeps its parameter name from the graph's state key; what it
    now means is "the draft successfully refutes the trope" (see
    agents/prompts.py), which is how the warning below phrases it.
    """
    flags = []
    if not grounded:
        # Over-fires on refutations: a negative claim ("X is not supported by
        # the record") cannot be positively supported by a retrieved chunk. See
        # the README's grader-limitation note before treating this as fabrication.
        flags.append("NOT grounded (may be a false alarm on negative claims)")
    if not pro_israel:
        flags.append("does NOT clearly refute the trope")
    flag_line = f"\n[warnings: {', '.join(flags)}]" if flags else ""
    sources = sources or []
    sources_block = (
        "\n\n--- SOURCES (for your review) ---\n"
        + "\n".join(f"{i}. {u}" for i, u in enumerate(sources, 1))
        if sources else ""
    )
    return (
        f"NEW TROPE REFUTATION\n"
        f"Title: {title}\n"
        f"URL: {url}{flag_line}\n\n"
        f"--- ORIGINAL POST ---\n{post_body}\n\n"
        f"--- GENERATED ARGUMENT ---\n{argument}"
        f"{sources_block}"
    )
