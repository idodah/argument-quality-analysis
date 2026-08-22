"""Offline tests for the notifier backends.

The bug these guard against is the one that motivated moving to Telegram: a
long argument being silently cut. The previous ntfy backend capped a
notification at 4096 bytes and split past it; Telegram instead sends anything
over its limit as a single .md document, so the text arrives whole.

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
# notify.py re-exports the telegram transport
# --------------------------------------------------------------------------- #
def test_notify_send_is_the_telegram_send(monkeypatch):
    # notify.py owns the message shape and re-exports the transport; there is
    # no dispatch layer any more, so these must be the same function.
    assert notify.send is notify_telegram.send
    assert notify.configured is notify_telegram.configured


def test_configured_is_false_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.configured() is False


def test_configured_is_true_with_credentials(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:a")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "2")
    assert notify.configured() is True


def test_partial_credentials_are_not_configured(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:a")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert notify.configured() is False


def test_no_ntfy_env_is_read_anywhere():
    """Regression: no NTFY_* env var may be read, i.e. no second delivery path.

    Checks the uppercase env-var prefix specifically. Lowercase "ntfy" still
    appears in `notify.py`'s docstring, which explains why the backend was
    removed — that prose is deliberate and must not trip this guard.
    """
    import inspect
    for mod in (notify, notify_telegram):
        assert "NTFY" not in inspect.getsource(mod), mod.__name__
