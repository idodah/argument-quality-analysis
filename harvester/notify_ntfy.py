"""ntfy notifier: send one push per generated response.

Chosen for having no credentials to configure — subscribe to a topic name in
the app and that is the entire setup. Every alternative tried here needed an
account, a bot token, a webhook, or an app password, and each hit a different
external wall (Telegram unreachable from the cluster; Gmail requiring 2FA app
passwords; Microsoft having disabled basic SMTP auth for consumer accounts).

Long messages: ntfy caps a *notification body* at 4096 bytes, but a request
larger than that is not rejected or truncated — the server stores the whole
payload as a `.txt` attachment and the notification links to it. So this sends
the message in ONE request and lets ntfy do that, rather than the earlier
approach of splitting at 4000 bytes into several independent pushes that
arrived unordered and cut mid-sentence.

Setup (`.env`):
    NTFY_TOPIC=cmv-<something-unguessable>
    NTFY_SERVER=https://ntfy.sh    # optional, this is the default
    NTFY_TOKEN=tk_...              # optional, Bearer auth for a protected topic

Then subscribe to that topic in the ntfy app or at https://ntfy.sh/<topic>.

PRIVACY: ntfy.sh is a public relay. Anyone who knows (or guesses) the topic
name can read every message, and attachments are served at a public URL for
about 3 hours. Pick a long, random topic name — or self-host and point
NTFY_SERVER at it — since the drafts quote real people's posts.
"""

from __future__ import annotations

import os

import requests

REQUEST_TIMEOUT = 30
DEFAULT_SERVER = "https://ntfy.sh"

# ntfy's inline-body limit. Past this the server stores the payload as an
# attachment instead; we never split, so this is informational.
INLINE_LIMIT = 4096


def configured() -> bool:
    """True if a usable ntfy topic is configured in the environment."""
    return bool(os.environ.get("NTFY_TOPIC"))


def _settings() -> dict:
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        raise RuntimeError(
            "ntfy not configured: set NTFY_TOPIC in your .env (optionally "
            "NTFY_SERVER, default https://ntfy.sh), then subscribe to that "
            "topic in the ntfy app or at https://ntfy.sh/<topic>."
        )
    return {
        "url": f"{(os.environ.get('NTFY_SERVER') or DEFAULT_SERVER).rstrip('/')}/{topic}",
        "token": os.environ.get("NTFY_TOKEN"),
    }


def send(text: str, click_url: str | None = None) -> None:
    """Push `text` to the configured topic in a single request.

    Nothing is split or truncated: a body over INLINE_LIMIT is stored by ntfy
    as a `.txt` attachment that the notification links to.
    """
    cfg = _settings()
    headers = {
        "Title": "New trope refutation",
        "Tags": "speech_balloon",
    }
    # Only http(s) links are valid ntfy click/action targets; skip anything else
    # (e.g. an empty url) so we never send a malformed header.
    if click_url and click_url.startswith(("http://", "https://")):
        headers["Click"] = click_url
        headers["Actions"] = f"view, Open post, {click_url}"
    if cfg["token"]:
        headers["Authorization"] = f"Bearer {cfg['token']}"

    try:
        resp = requests.post(cfg["url"], data=text.encode("utf-8"),
                             headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        raise RuntimeError(f"ntfy send failed: {e}") from e
    if resp.status_code != 200:
        raise RuntimeError(f"ntfy send failed ({resp.status_code}): {resp.text[:300]}")
