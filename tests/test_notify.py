"""Offline tests for the notifier backends.

The bug these guard against is the one that motivated the Telegram backend: a
long argument being silently cut. ntfy caps a notification body at 4096 bytes,
so the ntfy path splits into several independent pushes; Telegram instead sends
anything over its limit as a single .md document, so the text arrives whole.

No network: `requests.post` is stubbed and the calls are inspected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harvester import notify, notify_telegram


@pytest.fixture
def tg_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.delenv("NOTIFY_BACKEND", raising=False)
    monkeypatch.delenv("NTFY_TOPIC", raising=False)


def _ok_response():
    resp = mock.Mock()
    resp.status_code = 200
    resp.json.return_value = {"ok": True, "result": {}}
    return resp


# --------------------------------------------------------------------------- #
# Telegram: short vs. long
# --------------------------------------------------------------------------- #
def test_short_message_goes_as_text(tg_env):
    with mock.patch.object(notify_telegram.requests, "post",
                           return_value=_ok_response()) as post:
        notify_telegram.send("a short argument")
    (url,), kwargs = post.call_args
    assert url.endswith("/sendMessage")
    assert kwargs["data"]["text"] == "a short argument"
    assert kwargs.get("files") is None


def test_long_message_is_sent_whole_as_a_document(tg_env):
    # The regression that matters: nothing may be truncated or split.
    body = "x" * (notify_telegram.MAX_TEXT + 5000)
    with mock.patch.object(notify_telegram.requests, "post",
                           return_value=_ok_response()) as post:
        notify_telegram.send(body)
    (url,), kwargs = post.call_args
    assert url.endswith("/sendDocument")
    _, buf, _ = kwargs["files"]["document"]
    assert buf.getvalue().decode() == body, "document must carry the FULL text"


def test_long_message_sends_exactly_one_request(tg_env):
    body = "y" * (notify_telegram.MAX_TEXT * 3)
    with mock.patch.object(notify_telegram.requests, "post",
                           return_value=_ok_response()) as post:
        notify_telegram.send(body)
    assert post.call_count == 1, "must not split into multiple notifications"


def test_document_caption_respects_the_api_limit(tg_env):
    body = "z" * (notify_telegram.MAX_TEXT + 100)
    with mock.patch.object(notify_telegram.requests, "post",
                           return_value=_ok_response()) as post:
        notify_telegram.send(body)
    caption = post.call_args.kwargs["data"]["caption"]
    assert len(caption) <= notify_telegram.MAX_CAPTION


def test_click_url_becomes_an_inline_button(tg_env):
    with mock.patch.object(notify_telegram.requests, "post",
                           return_value=_ok_response()) as post:
        notify_telegram.send("hi", click_url="https://reddit.com/r/x/1")
    markup = json.loads(post.call_args.kwargs["data"]["reply_markup"])
    assert markup["inline_keyboard"][0][0]["url"] == "https://reddit.com/r/x/1"


def test_non_http_click_url_is_dropped(tg_env):
    with mock.patch.object(notify_telegram.requests, "post",
                           return_value=_ok_response()) as post:
        notify_telegram.send("hi", click_url="javascript:alert(1)")
    assert "reply_markup" not in post.call_args.kwargs["data"]


def test_markdown_parse_failure_retries_as_plain_text(tg_env):
    # Generated prose can contain a stray '*' or '_'. Losing the notification to
    # a formatting quirk is worse than losing the formatting.
    bad = mock.Mock()
    bad.status_code = 400
    bad.json.return_value = {"ok": False, "description": "Bad Request: can't parse entities"}
    with mock.patch.object(notify_telegram.requests, "post",
                           side_effect=[bad, _ok_response()]) as post:
        notify_telegram.send("a *broken_ markdown message")
    assert post.call_count == 2
    assert "parse_mode" not in post.call_args.kwargs["data"]


def test_missing_credentials_raises_with_setup_help(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(RuntimeError, match="BotFather"):
        notify_telegram.send("hi")


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #
def test_telegram_preferred_when_both_configured(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:a")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "2")
    monkeypatch.setenv("NTFY_TOPIC", "t")
    monkeypatch.delenv("NOTIFY_BACKEND", raising=False)
    assert notify.active_backend() == "telegram"


def test_explicit_backend_env_wins(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:a")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "2")
    monkeypatch.setenv("NTFY_TOPIC", "t")
    monkeypatch.setenv("NOTIFY_BACKEND", "ntfy")
    assert notify.active_backend() == "ntfy"


def test_falls_back_to_ntfy_when_only_it_is_configured(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("NOTIFY_BACKEND", raising=False)
    monkeypatch.setenv("NTFY_TOPIC", "t")
    assert notify.active_backend() == "ntfy"


def test_no_backend_configured(monkeypatch):
    for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "NTFY_TOPIC", "NOTIFY_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    assert notify.active_backend() is None
    assert notify.configured() is False
    with pytest.raises(RuntimeError, match="No notifier configured"):
        notify.send("hi")


def test_send_routes_to_telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:a")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "2")
    monkeypatch.delenv("NOTIFY_BACKEND", raising=False)
    with mock.patch.object(notify_telegram, "send") as tg:
        notify.send("hi", click_url="https://x.com")
    tg.assert_called_once_with("hi", click_url="https://x.com")
