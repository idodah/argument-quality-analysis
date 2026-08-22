"""Telegram notifier backend: send one message per generated response.

Chosen over ntfy for long arguments. ntfy.sh caps a notification body at 4096
bytes and the previous code worked around that by splitting a message into
several independent pushes, which arrive unordered and cut mid-sentence.
Telegram handles the long case properly: messages up to 4096 characters go out
as text, and anything longer is sent as a .md document (bots may upload up to
50 MB), so a long argument arrives whole, in one notification, and stays
readable in the chat history rather than expiring.

Setup:
  1. Talk to @BotFather, `/newbot`, copy the token.
  2. Message your new bot once (a bot cannot open a conversation with you).
  3. Get your chat id:  curl https://api.telegram.org/bot<TOKEN>/getUpdates
  4. Put both in `.env`:
         TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
         TELEGRAM_CHAT_ID=987654321

Unlike ntfy.sh (a public relay serving attachments at guessable URLs), a bot
delivers only to the chat id you configure.
"""

from __future__ import annotations

import io
import json
import os

import requests

API_BASE = "https://api.telegram.org/bot{token}/{method}"

# Telegram Bot API limits.
MAX_TEXT = 4096          # sendMessage `text`
MAX_CAPTION = 1024       # sendDocument `caption`
REQUEST_TIMEOUT = 30

# Room for the caption's own framing (title/url/flag lines) before the excerpt.
_CAPTION_BUDGET = MAX_CAPTION - 200


def configured() -> bool:
    """True if both bot token and chat id are present in the environment."""
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def _credentials() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "Telegram not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID "
            "in your .env. Create a bot with @BotFather, message it once, then "
            "read your chat id from "
            "https://api.telegram.org/bot<TOKEN>/getUpdates"
        )
    return token, chat_id


def _post(method: str, token: str, *, data: dict, files: dict | None = None) -> dict:
    """POST one Bot API call, raising with Telegram's own error text on failure."""
    resp = requests.post(
        API_BASE.format(token=token, method=method),
        data=data, files=files, timeout=REQUEST_TIMEOUT,
    )
    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"telegram {method} failed ({resp.status_code}): {resp.text[:300]}")
    if not payload.get("ok"):
        # description carries the actual reason (bad token, chat not found, ...)
        raise RuntimeError(
            f"telegram {method} failed ({resp.status_code}): "
            f"{payload.get('description', 'unknown error')}"
        )
    return payload


def _keyboard(click_url: str | None) -> dict | None:
    """An 'Open post' button, when the source url is a usable http(s) link."""
    if not click_url or not click_url.startswith(("http://", "https://")):
        return None
    return {"inline_keyboard": [[{"text": "Open post", "url": click_url}]]}


def send(text: str, click_url: str | None = None, markdown: bool = True) -> None:
    """Deliver `text` to the configured chat, whole.

    Short messages go as text. Anything over MAX_TEXT is uploaded as a .md
    document instead of being split, so the argument is never cut mid-sentence
    and stays in the chat history as one artifact.
    """
    token, chat_id = _credentials()
    markup = _keyboard(click_url)

    if len(text) <= MAX_TEXT:
        data = {
            "chat_id": chat_id,
            "text": text,
            # Long-form arguments carry urls in a Sources block; previews would
            # push the text itself off-screen.
            "link_preview_options": json.dumps({"is_disabled": True}),
        }
        if markdown:
            data["parse_mode"] = "Markdown"
        if markup:
            data["reply_markup"] = json.dumps(markup)
        try:
            _post("sendMessage", token, data=data)
            return
        except RuntimeError as e:
            # A stray '*' or '_' in generated prose can make Telegram reject the
            # whole message. Losing the notification to a formatting quirk is
            # worse than losing the formatting, so retry as plain text.
            if markdown and "parse" in str(e).lower():
                data.pop("parse_mode", None)
                _post("sendMessage", token, data=data)
                return
            raise

    # Too long for one message: send the full text as a document.
    caption = text[:_CAPTION_BUDGET].rstrip()
    if len(text) > _CAPTION_BUDGET:
        caption += "\n\n[…full text attached]"
    data = {"chat_id": chat_id, "caption": caption}
    if markup:
        data["reply_markup"] = json.dumps(markup)
    buf = io.BytesIO(text.encode("utf-8"))
    buf.name = "response.md"
    _post("sendDocument", token, data=data, files={"document": ("response.md", buf, "text/markdown")})
