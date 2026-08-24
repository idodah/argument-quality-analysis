"""Email notifier: send one message per generated response over SMTP.

Chosen after two push backends failed for different reasons — ntfy truncated
long arguments, and api.telegram.org is unreachable from the cluster this runs
on. Email has no length limit at all, so the full refutation arrives intact as
ordinary mail: searchable, permanent, and readable from any device without an
app. It is also the delivery path least likely to be blocked, since SMTP is
allowed almost everywhere HTTPS is.

Uses only the standard library (`smtplib` + `email`), so it adds no dependency
to the deployed image.

Setup (`.env`):
    SMTP_HOST=smtp.gmail.com          # optional, this is the default
    SMTP_PORT=587                     # optional; 587 STARTTLS or 465 implicit SSL
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=<app password>      # NOT your account password
    EMAIL_TO=you@gmail.com
    EMAIL_FROM=you@gmail.com          # optional, defaults to SMTP_USER

For Gmail you must enable 2-Step Verification, then create an **App Password**
(Google Account -> Security -> App passwords). A normal account password is
rejected. Other providers work the same way; only host/port change.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate

DEFAULT_HOST = "smtp.gmail.com"
DEFAULT_PORT = 587
TIMEOUT = 30

SUBJECT_PREFIX = "[harvester]"
# Keep the subject to one readable line; the body carries everything.
MAX_SUBJECT_CHARS = 120


def configured() -> bool:
    """True if the minimum SMTP settings are present in the environment."""
    return bool(
        os.environ.get("SMTP_USER")
        and os.environ.get("SMTP_PASSWORD")
        and os.environ.get("EMAIL_TO")
    )


def _settings() -> dict:
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to_addr = os.environ.get("EMAIL_TO")
    if not (user and password and to_addr):
        raise RuntimeError(
            "Email not configured: set SMTP_USER, SMTP_PASSWORD and EMAIL_TO in "
            "your .env. For Gmail, SMTP_PASSWORD must be an App Password "
            "(Google Account -> Security -> App passwords), not your account "
            "password; 2-Step Verification has to be on to create one."
        )
    return {
        "host": os.environ.get("SMTP_HOST") or DEFAULT_HOST,
        "port": int(os.environ.get("SMTP_PORT") or DEFAULT_PORT),
        "user": user,
        "password": password,
        "to": to_addr,
        "from": os.environ.get("EMAIL_FROM") or user,
    }


def _subject(text: str) -> str:
    """Derive a one-line subject from the message's own header lines.

    format_result() puts a 'Title: ...' line near the top; use it so the inbox
    shows which post was answered without opening the mail.
    """
    for line in text.splitlines():
        if line.startswith("Title:"):
            title = line[len("Title:"):].strip()
            if title:
                subject = f"{SUBJECT_PREFIX} {title}"
                return subject[:MAX_SUBJECT_CHARS]
    return f"{SUBJECT_PREFIX} new trope refutation"


def send(text: str, click_url: str | None = None) -> None:
    """Deliver `text` as an email, whole.

    There is no size limit to work around here — unlike the push backends this
    replaced, the entire refutation goes in the body untouched.
    """
    cfg = _settings()

    body = text
    if click_url and click_url.startswith(("http://", "https://")):
        body = f"{text}\n\n--- SOURCE POST ---\n{click_url}\n"

    msg = EmailMessage()
    msg["Subject"] = _subject(text)
    msg["From"] = cfg["from"]
    msg["To"] = cfg["to"]
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(body)

    context = ssl.create_default_context()
    try:
        if cfg["port"] == 465:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=TIMEOUT,
                                  context=context) as server:
                server.login(cfg["user"], cfg["password"])
                server.send_message(msg)
        else:
            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=TIMEOUT) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(cfg["user"], cfg["password"])
                server.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        # By far the most common failure: a real password instead of an app one.
        raise RuntimeError(
            f"SMTP auth failed for {cfg['user']} ({e.smtp_code}). For Gmail, "
            "SMTP_PASSWORD must be a 16-character App Password, not your "
            "account password."
        ) from e
    except (smtplib.SMTPException, OSError) as e:
        raise RuntimeError(f"email send failed via {cfg['host']}:{cfg['port']}: {e}") from e
