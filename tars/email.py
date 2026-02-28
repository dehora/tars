"""Email channel for tars — IMAP polling + SMTP reply."""

import email
import email.utils
import imaplib
import os
import re
import smtplib
import sys
import time
from email.message import EmailMessage
from email.mime.text import MIMEText
from pathlib import Path

from tars.brief import build_brief_sections, format_brief_text
from tars.commands import dispatch
from tars.config import ModelConfig, model_summary
from tars.conversation import Conversation, process_message, save_session
from tars.sessions import _session_path


def _email_config() -> dict | None:
    """Load email config from env vars. Returns None if not configured."""
    address = os.environ.get("TARS_EMAIL_ADDRESS")
    password = os.environ.get("TARS_EMAIL_PASSWORD")
    allow = os.environ.get("TARS_EMAIL_ALLOW")
    to_addr = os.environ.get("TARS_EMAIL_TO")
    if not address or not password or not allow or not to_addr:
        return None
    interval = os.environ.get("TARS_EMAIL_POLL_INTERVAL", "60")
    try:
        interval_sec = int(interval)
    except ValueError:
        interval_sec = 60
    return {
        "address": address,
        "password": password,
        "allow": [a.strip().lower() for a in allow.split(",") if a.strip()],
        "poll_interval": interval_sec,
        "to": to_addr.strip(),
    }


def _extract_body(msg: email.message.Message) -> str:
    """Extract plain text body from email.

    Prefers text/plain from multipart/alternative.
    Strips quoted reply lines (starting with '>').
    """
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            # Skip attachment parts — only consider inline/body text.
            if part.get_content_disposition() == "attachment":
                continue
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")
                break
        # Fallback: try text/html if no text/plain found
        if not body:
            for part in msg.walk():
                if part.get_content_disposition() == "attachment":
                    continue
                ct = part.get_content_type()
                if ct == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        html = payload.decode(charset, errors="replace")
                        body = _strip_html(html)
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace")

    # Strip quoted reply lines
    lines = body.splitlines()
    cleaned = [line for line in lines if not line.startswith(">")]
    return "\n".join(cleaned).strip()


def _strip_html(html: str) -> str:
    """Basic HTML tag removal for fallback body extraction."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _thread_id(msg: email.message.Message) -> str:
    """Extract thread ID from email headers.

    Uses the root Message-ID from the References chain,
    or the message's own Message-ID for new threads.
    """
    refs = msg.get("References", "")
    if refs:
        # References is space-separated; first entry is the root
        ids = refs.strip().split()
        if ids:
            return ids[0]
    return msg.get("Message-ID", f"unknown-{time.time()}")


def _is_allowed_sender(msg: email.message.Message, allowed: list[str]) -> bool:
    """Check if the sender is in the whitelist."""
    from_header = msg.get("From", "")
    # Extract email address from "Name <addr>" format
    _, addr = email.utils.parseaddr(from_header)
    return addr.lower() in allowed


def _connect_imap(address: str, password: str) -> imaplib.IMAP4_SSL:
    """Connect to Gmail IMAP."""
    imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
    imap.login(address, password)
    return imap


def _fetch_unseen(
    imap: imaplib.IMAP4_SSL, allowed: list[str],
) -> list[tuple[bytes, email.message.Message]]:
    """Fetch unread emails, filtered to allowed senders.

    Uses BODY.PEEK[] so messages stay UNSEEN until explicitly marked
    after successful processing. Returns (msg_num, Message) tuples.
    """
    imap.select("INBOX")
    _, data = imap.search(None, "UNSEEN")
    if not data or not data[0]:
        return []

    messages = []
    for num in data[0].split():
        _, msg_data = imap.fetch(num, "(BODY.PEEK[])")
        if not msg_data or not msg_data[0]:
            continue
        raw = msg_data[0]
        if isinstance(raw, tuple) and len(raw) >= 2:
            msg = email.message_from_bytes(raw[1])
        else:
            continue
        if _is_allowed_sender(msg, allowed):
            messages.append((num, msg))
        else:
            imap.store(num, "+FLAGS", "\\Seen")

    return messages


def _send_reply(config: dict, original: email.message.Message, body: str) -> None:
    """Send reply threading correctly with original."""
    reply = MIMEText(body, "plain", "utf-8")

    reply["From"] = config["address"]
    # Reply to the sender
    _, from_addr = email.utils.parseaddr(original.get("From", ""))
    reply["To"] = from_addr

    # Threading headers
    orig_id = original.get("Message-ID", "")
    reply["In-Reply-To"] = orig_id
    orig_refs = original.get("References", "")
    if orig_refs:
        reply["References"] = f"{orig_refs} {orig_id}"
    else:
        reply["References"] = orig_id

    # Subject
    orig_subject = original.get("Subject", "")
    if orig_subject.lower().startswith("re:"):
        reply["Subject"] = orig_subject
    else:
        reply["Subject"] = f"Re: {orig_subject}"

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(config["address"], config["password"])
        smtp.send_message(reply)


def _send_message(config: dict, to_addr: str, subject: str, body: str) -> None:
    """Send a standalone email (non-reply)."""
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = config["address"]
    msg["To"] = to_addr
    msg["Subject"] = subject

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(config["address"], config["password"])
        smtp.send_message(msg)


def send_brief_email() -> None:
    """Send the daily brief to the configured recipient."""
    config = _email_config()
    if config is None:
        raise RuntimeError(
            "missing config — set TARS_EMAIL_ADDRESS, TARS_EMAIL_PASSWORD, "
            "TARS_EMAIL_ALLOW, TARS_EMAIL_TO"
        )
    sections = build_brief_sections()
    body = format_brief_text(sections)
    _send_message(config, config["to"], "tars brief", body)


# In-memory conversation state, keyed by thread ID
_conversations: dict[str, Conversation] = {}
_session_files: dict[str, Path | None] = {}

_MAX_RETRIES = 3
# msg_num → (attempt_count, cached_reply_text_or_None)
_failed: dict[bytes, tuple[int, str | None]] = {}


def run_email(model_config: ModelConfig) -> None:
    """Poll inbox and process messages. Runs until interrupted."""
    email_config = _email_config()
    if email_config is None:
        print(
            "email: missing config — set TARS_EMAIL_ADDRESS, "
            "TARS_EMAIL_PASSWORD, TARS_EMAIL_ALLOW, TARS_EMAIL_TO",
            file=sys.stderr,
        )
        return

    from tars.commands import set_task_runner
    from tars.taskrunner import TaskRunner

    summary = model_summary(model_config)
    print(
        f"email: polling {email_config['address']} every {email_config['poll_interval']}s "
        f"[{summary['primary']}]"
    )
    print(f"email: allowed senders: {', '.join(email_config['allow'])}")

    runner = TaskRunner(model_config.primary_provider, model_config.primary_model)
    runner.start()
    set_task_runner(runner)

    imap: imaplib.IMAP4_SSL | None = None

    try:
        while True:
            # Connect / reconnect
            if imap is None:
                try:
                    imap = _connect_imap(email_config["address"], email_config["password"])
                    print("email: connected to IMAP")
                except Exception as e:
                    print(f"email: IMAP connect failed: {e}", file=sys.stderr)
                    time.sleep(email_config["poll_interval"])
                    continue

            try:
                emails = _fetch_unseen(imap, email_config["allow"])
            except (imaplib.IMAP4.error, OSError) as e:
                print(f"email: fetch failed, reconnecting: {e}", file=sys.stderr)
                imap = None
                continue

            seen_nums: set[bytes] = set()

            for msg_num, msg in emails:
                seen_nums.add(msg_num)
                tid = _thread_id(msg)
                _, from_addr = email.utils.parseaddr(msg.get("From", ""))
                subject = msg.get("Subject", "(no subject)")

                attempts, cached_reply = _failed.get(msg_num, (0, None))

                if attempts >= _MAX_RETRIES:
                    print(f"email: max retries for [{from_addr}] {subject}", file=sys.stderr)
                    try:
                        _send_reply(
                            email_config, msg,
                            "I wasn't able to send a reply after multiple attempts. "
                            "Please resend your message.",
                        )
                    except Exception:
                        pass
                    imap.store(msg_num, "+FLAGS", "\\Seen")
                    _failed.pop(msg_num, None)
                    continue

                # Use cached reply from a previous send failure
                if cached_reply is not None:
                    reply_text = cached_reply
                    print(f"email: retrying send ({attempts + 1}/{_MAX_RETRIES}) [{from_addr}]")
                else:
                    body = _extract_body(msg)
                    print(f"email: [{from_addr}] {subject}")

                    # Try slash command: check subject first, then body
                    email_ctx = {"channel": "email"}
                    slash_reply = dispatch(
                        subject,
                        model_config.primary_provider,
                        model_config.primary_model,
                        context=email_ctx,
                    )
                    if slash_reply is None and body:
                        slash_reply = dispatch(
                            body,
                            model_config.primary_provider,
                            model_config.primary_model,
                            context=email_ctx,
                        )
                    if slash_reply is not None:
                        reply_text = slash_reply
                        print(f"email: [slash] {slash_reply[:60]}")
                    elif not body:
                        imap.store(msg_num, "+FLAGS", "\\Seen")
                        _failed.pop(msg_num, None)
                        continue
                    else:
                        if tid not in _conversations:
                            _conversations[tid] = Conversation(
                                id=f"email-{tid}",
                                provider=model_config.primary_provider,
                                model=model_config.primary_model,
                                remote_provider=model_config.remote_provider,
                                remote_model=model_config.remote_model,
                                routing_policy=model_config.routing_policy,
                            )
                            _session_files[tid] = _session_path()
                        conv = _conversations[tid]

                        try:
                            reply_text = process_message(conv, body, _session_files[tid])
                        except Exception as e:
                            error_type = type(e).__name__
                            print(f"email: process failed: {e}", file=sys.stderr)
                            reply_text = (
                                f"I couldn't process your message ({error_type}). "
                                "Try again later."
                            )

                try:
                    _send_reply(email_config, msg, reply_text)
                    imap.store(msg_num, "+FLAGS", "\\Seen")
                    _failed.pop(msg_num, None)
                    print(f"email: replied to {from_addr}")
                except Exception as e:
                    _failed[msg_num] = (attempts + 1, reply_text)
                    print(f"email: send failed ({attempts + 1}/{_MAX_RETRIES}): {e}", file=sys.stderr)

            # Clean stale tracking for messages no longer UNSEEN
            stale = [k for k in _failed if k not in seen_nums]
            for k in stale:
                del _failed[k]

            time.sleep(email_config["poll_interval"])

    except KeyboardInterrupt:
        print("\nemail: shutting down...")
    finally:
        runner.stop()
        set_task_runner(None)
        # Save all sessions
        for tid, conv in _conversations.items():
            try:
                save_session(conv, _session_files.get(tid))
            except Exception:
                pass
        if imap:
            try:
                imap.logout()
            except Exception:
                pass
        print("email: stopped")
