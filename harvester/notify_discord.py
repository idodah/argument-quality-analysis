"""Discord webhook notifier: send one message per generated response.

Chosen after `api.telegram.org` turned out to be unreachable from the cluster
this runs on (TLS reset at handshake, while discord.com, slack, pushover and
the SMTP relays were all fine) — the block is Telegram-specific, not general
egress filtering.

Handles the long-argument case the same way the Telegram backend did: content
up to 2000 characters goes inline, anything longer is uploaded as a `.md`
attachment via multipart, so a long refutation arrives whole in one message and
is never split or truncated.

Setup:
  1. In your Discord server: Server Settings -> Integrations -> Webhooks ->
     New Webhook, pick a channel, "Copy Webhook URL".
  2. Put it in `.env`:
         DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/<id>/<token>

The webhook URL is a credential — anyone holding it can post to that channel.
It lives in `.env` (gitignored), never in code or a commit.
"""

from __future__ import annotations

import io
import json
import os

import requests

# Discord Execute Webhook limits.
MAX_CONTENT = 2000        # `content` field
MAX_FILE_BYTES = 8 * 1024 * 1024   # conservative; server default is 10MB+
REQUEST_TIMEOUT = 30

# Leave room for the "full text attached" pointer appended to the inline part.
_INLINE_BUDGET = MAX_CONTENT - 120

USERNAME = "CMV Harvester"


def configured() -> bool:
    """True if a webhook URL is present in the environment."""
    return bool(os.environ.get("DISCORD_WEBHOOK_URL"))


def _webhook_url() -> str:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        raise RuntimeError(
            "Discord not configured: set DISCORD_WEBHOOK_URL in your .env. "
            "Create one via Server Settings -> Integrations -> Webhooks -> "
            "New Webhook -> Copy Webhook URL."
        )
    return url


def _post(url: str, *, data=None, json_body=None, files=None) -> None:
    """POST to the webhook, raising with Discord's own error text on failure."""
    resp = requests.post(url, data=data, json=json_body, files=files,
                         timeout=REQUEST_TIMEOUT)
    # 204 No Content is the normal success for a webhook execute.
    if resp.status_code in (200, 204):
        return
    detail = resp.text[:300]
    try:
        payload = resp.json()
        detail = payload.get("message", detail)
    except ValueError:
        pass
    raise RuntimeError(f"discord webhook failed ({resp.status_code}): {detail}")


def send(text: str, click_url: str | None = None) -> None:
    """Deliver `text` to the configured channel, whole.

    Short messages go inline. Anything over the 2000-character limit is uploaded
    as a `.md` attachment instead of being split, so the argument is never cut
    mid-sentence and stays in the channel history as one artifact.
    """
    url = _webhook_url()

    # Discord has no "button" primitive for plain webhooks; a bare URL on its
    # own line is what renders as a clickable link + preview.
    link_line = ""
    if click_url and click_url.startswith(("http://", "https://")):
        link_line = f"\n{click_url}"

    if len(text) + len(link_line) <= MAX_CONTENT:
        _post(url, json_body={
            "username": USERNAME,
            "content": text + link_line,
            # Suppress link embeds: the Sources block would otherwise generate
            # a wall of previews that pushes the argument off-screen.
            "flags": 4,
        })
        return

    # Too long for one message: inline an excerpt, attach the full text.
    excerpt = text[:_INLINE_BUDGET].rstrip()
    payload = {
        "username": USERNAME,
        "content": f"{excerpt}\n\n[…full text attached]{link_line}",
        "flags": 4,
    }
    blob = text.encode("utf-8")
    if len(blob) > MAX_FILE_BYTES:
        blob = blob[:MAX_FILE_BYTES]
    buf = io.BytesIO(blob)
    _post(url,
          data={"payload_json": json.dumps(payload)},
          files={"files[0]": ("response.md", buf, "text/markdown")})
