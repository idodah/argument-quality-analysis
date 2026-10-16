"""Notifier: send one ntfy push per generated response.

The transport lives in `harvester.notify_ntfy`; this module owns the message
*shape* (`format_result`) and re-exports `send` / `configured` so callers have
one import.

ntfy is the backend because it is the only one needing no credentials: pick a
topic name, subscribe to it in the app, done. The alternatives each required an
account and each hit a different external wall — Telegram is unreachable from
this cluster (TLS reset at handshake), Gmail SMTP needs 2FA plus an app
password, and Microsoft has disabled basic SMTP auth for consumer accounts.

Long messages are NOT split. ntfy stores any payload over 4096 bytes as a
`.txt` attachment and links to it from the notification, so the full refutation
survives in one push — the earlier version of this module chunked at 4000 bytes
instead, which is what produced several unordered, mid-sentence pushes.

PRIVACY: ntfy.sh is a public relay — anyone who knows the topic name can read
the messages. Use a long random topic, or self-host and set NTFY_SERVER.
"""

from __future__ import annotations

from harvester.notify_ntfy import configured, send

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
