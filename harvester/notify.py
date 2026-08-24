"""Notifier: send one email per generated response.

The transport lives in `harvester.notify_email`; this module owns the message
*shape* (`format_result`) and re-exports `send` / `configured` so callers have
one import.

Email is the only backend. Three were tried before it:

  - **ntfy** capped a notification body at 4096 bytes and split anything longer
    into independent pushes that arrived unordered and cut mid-sentence; ntfy.sh
    is also a public relay serving attachments at guessable URLs.
  - **Telegram** handled long messages correctly, but `api.telegram.org` is
    unreachable from the cluster this runs on (TLS reset at handshake) — a
    Telegram-specific block no code change could work around.
  - **Discord** worked and was reachable, but still had to split a long
    refutation between an inline excerpt and a `.md` attachment.

Email has no length limit at all, so the whole refutation arrives untouched in
one message — and SMTP is permitted almost anywhere HTTPS is, which matters
after two backends were lost to reachability. It needs no third-party service
and no extra dependency (stdlib `smtplib`).
"""

from __future__ import annotations

from harvester.notify_email import configured, send

__all__ = ["configured", "send", "format_result"]


def format_result(title: str, url: str, post_body: str, argument: str,
                  grounded: bool, refutes_trope: bool, sources: list[str] | None = None) -> str:
    """Build the per-post message: original post + argument + a trailing sources
    list (the argument carries no inline citations; sources are separated out for
    human review).

    `refutes_trope` keeps its parameter name from the graph's state key; what it
    now means is "the draft successfully refutes the trope" (see
    agents/prompts.py), which is how the warning below phrases it.
    """
    flags = []
    if not grounded:
        # Over-fires on refutations: a negative claim ("X is not supported by
        # the record") cannot be positively supported by a retrieved chunk. See
        # the README's grader-limitation note before treating this as fabrication.
        flags.append("NOT grounded (may be a false alarm on negative claims)")
    if not refutes_trope:
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
