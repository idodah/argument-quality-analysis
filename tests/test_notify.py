"""Offline tests for the email notifier backend.

The bug these guard against drove three backend changes: a long argument being
silently cut. ntfy capped at 4096 bytes and split past it; Discord split a long
message between an inline excerpt and an attachment. Email has no length limit,
so the test that matters is that the body goes out byte-for-byte intact.

No network: smtplib is stubbed and the constructed message is inspected.
"""

from __future__ import annotations

import smtplib
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harvester import notify, notify_email


@pytest.fixture
def smtp_env(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "me@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("EMAIL_TO", "me@example.com")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("EMAIL_FROM", raising=False)


def _capture(monkeypatch, cls="SMTP"):
    """Patch smtplib and return the mock server whose send_message we inspect."""
    server = mock.MagicMock()
    server.__enter__.return_value = server
    monkeypatch.setattr(notify_email.smtplib, cls, mock.Mock(return_value=server))
    return server


def _sent_body(server):
    msg = server.send_message.call_args.args[0]
    return msg.get_content()


# --------------------------------------------------------------------------- #
# the whole point: no truncation, ever
# --------------------------------------------------------------------------- #
def test_long_body_is_sent_completely_intact(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    body = "x" * 50_000
    notify_email.send(body)
    # get_content() appends a trailing newline per RFC; compare on the content.
    assert _sent_body(server).rstrip("\n") == body


def test_long_body_sends_exactly_one_message(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    notify_email.send("y" * 50_000)
    assert server.send_message.call_count == 1


def test_short_body_is_unchanged(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    notify_email.send("a short refutation")
    assert _sent_body(server).rstrip("\n") == "a short refutation"


# --------------------------------------------------------------------------- #
# headers
# --------------------------------------------------------------------------- #
def test_subject_uses_the_post_title(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    notify_email.send("NEW TROPE REFUTATION\nTitle: [reddit] Rothschild myth\nURL: x")
    msg = server.send_message.call_args.args[0]
    assert "Rothschild myth" in msg["Subject"]
    assert msg["Subject"].startswith(notify_email.SUBJECT_PREFIX)


def test_subject_falls_back_when_no_title_line(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    notify_email.send("body with no title header")
    assert server.send_message.call_args.args[0]["Subject"] == \
        f"{notify_email.SUBJECT_PREFIX} new trope refutation"


def test_subject_is_bounded(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    notify_email.send("Title: " + "z" * 500)
    assert len(server.send_message.call_args.args[0]["Subject"]) <= notify_email.MAX_SUBJECT_CHARS


def test_from_defaults_to_smtp_user(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    notify_email.send("hi")
    msg = server.send_message.call_args.args[0]
    assert msg["From"] == "me@example.com" and msg["To"] == "me@example.com"


def test_explicit_from_is_honoured(smtp_env, monkeypatch):
    monkeypatch.setenv("EMAIL_FROM", "bot@example.com")
    server = _capture(monkeypatch)
    notify_email.send("hi")
    assert server.send_message.call_args.args[0]["From"] == "bot@example.com"


# --------------------------------------------------------------------------- #
# click url + transport modes
# --------------------------------------------------------------------------- #
def test_click_url_is_appended(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    notify_email.send("body", click_url="https://reddit.com/r/x/1")
    assert "https://reddit.com/r/x/1" in _sent_body(server)


def test_non_http_click_url_is_dropped(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    notify_email.send("body", click_url="javascript:alert(1)")
    assert "javascript" not in _sent_body(server)


def test_port_465_uses_implicit_ssl(smtp_env, monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "465")
    server = _capture(monkeypatch, cls="SMTP_SSL")
    notify_email.send("hi")
    assert server.send_message.called
    server.starttls.assert_not_called()


def test_port_587_uses_starttls(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    notify_email.send("hi")
    server.starttls.assert_called_once()


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
def test_missing_config_raises_with_setup_help(monkeypatch):
    for k in ("SMTP_USER", "SMTP_PASSWORD", "EMAIL_TO"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="App Password"):
        notify_email.send("hi")


def test_auth_error_explains_app_passwords(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"denied")
    with pytest.raises(RuntimeError, match="App Password"):
        notify_email.send("hi")


def test_transport_error_is_wrapped(smtp_env, monkeypatch):
    server = _capture(monkeypatch)
    server.send_message.side_effect = smtplib.SMTPException("boom")
    with pytest.raises(RuntimeError, match="email send failed"):
        notify_email.send("hi")


# --------------------------------------------------------------------------- #
# notify.py re-exports the email transport
# --------------------------------------------------------------------------- #
def test_notify_send_is_the_email_send():
    assert notify.send is notify_email.send
    assert notify.configured is notify_email.configured


def test_configured_requires_all_three_settings(monkeypatch):
    for k in ("SMTP_USER", "SMTP_PASSWORD", "EMAIL_TO"):
        monkeypatch.delenv(k, raising=False)
    assert notify.configured() is False
    monkeypatch.setenv("SMTP_USER", "me@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    assert notify.configured() is False   # EMAIL_TO still missing
    monkeypatch.setenv("EMAIL_TO", "me@example.com")
    assert notify.configured() is True


def test_no_removed_backend_env_is_read_anywhere():
    """Regression: no NTFY_*, TELEGRAM_* or DISCORD_* env var may be read.

    Uppercase prefixes only; lowercase prose in the docstrings explains why
    those backends were dropped and is deliberate.
    """
    import inspect
    for mod in (notify, notify_email):
        src = inspect.getsource(mod)
        for prefix in ("NTFY", "TELEGRAM_", "DISCORD_"):
            assert prefix not in src, f"{mod.__name__} references {prefix}"
