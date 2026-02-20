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

from tars.cli import _parse_todoist_add
from tars.conversation import Conversation, process_message, save_session
from tars.format import format_tool_result
from tars.sessions import _session_path
from tars.tools import run_tool


def _email_config() -> dict | None:
    """Load email config from env vars. Returns None if not configured."""
    address = os.environ.get("TARS_EMAIL_ADDRESS")
    password = os.environ.get("TARS_EMAIL_PASSWORD")
    allow = os.environ.get("TARS_EMAIL_ALLOW")
    if not address or not password or not allow:
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
    }


def _extract_body(msg: email.message.Message) -> str:
    """Extract plain text body from email.

    Prefers text/plain from multipart/alternative.
    Strips quoted reply lines (starting with '>').
    """
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
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


def _fetch_unseen(imap: imaplib.IMAP4_SSL, allowed: list[str]) -> list[email.message.Message]:
    """Fetch unread emails, filtered to allowed senders.

    Marks all fetched emails as SEEN regardless of sender.
    Only returns messages from allowed senders.
    """
    imap.select("INBOX")
    _, data = imap.search(None, "UNSEEN")
    if not data or not data[0]:
        return []

    messages = []
    for num in data[0].split():
        _, msg_data = imap.fetch(num, "(RFC822)")
        if not msg_data or not msg_data[0]:
            continue
        raw = msg_data[0]
        if isinstance(raw, tuple) and len(raw) >= 2:
            msg = email.message_from_bytes(raw[1])
        else:
            continue
        # Mark as seen (implicit via fetch with RFC822 in most configs,
        # but explicitly mark to be safe)
        imap.store(num, "+FLAGS", "\\Seen")
        if _is_allowed_sender(msg, allowed):
            messages.append(msg)

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


def _handle_slash_command(body: str) -> str | None:
    """Handle slash commands in email body. Returns reply text, or None if not a command."""
    stripped = body.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split()
    cmd = parts[0]

    try:
        if cmd == "/todoist":
            sub = parts[1] if len(parts) > 1 else ""
            if sub == "add" and len(parts) > 2:
                args = _parse_todoist_add(parts[2:])
                if not args.get("content"):
                    return "Usage: /todoist add <text> [--due D] [--project P] [--priority N]"
                name = "todoist_add_task"
            elif sub == "today":
                args = {}
                name = "todoist_today"
            elif sub == "upcoming":
                try:
                    days = int(parts[2]) if len(parts) > 2 else 7
                except ValueError:
                    return "Usage: /todoist upcoming [days]"
                args = {"days": days}
                name = "todoist_upcoming"
            elif sub == "complete" and len(parts) > 2:
                args = {"ref": " ".join(parts[2:])}
                name = "todoist_complete_task"
            else:
                return "Usage: /todoist add|today|upcoming|complete ..."
        elif cmd == "/weather":
            name, args = "weather_now", {}
        elif cmd == "/forecast":
            name, args = "weather_forecast", {}
        elif cmd == "/memory":
            name, args = "memory_recall", {}
        elif cmd == "/remember" and len(parts) >= 3:
            name = "memory_remember"
            args = {"section": parts[1], "content": " ".join(parts[2:])}
        elif cmd == "/note" and len(parts) >= 2:
            name = "note_daily"
            args = {"content": " ".join(parts[1:])}
        else:
            return None  # Not a recognized command — let the model handle it

        raw = run_tool(name, args, quiet=True)
        return format_tool_result(name, raw)
    except Exception as e:
        return f"Tool error: {e}"


# In-memory conversation state, keyed by thread ID
_conversations: dict[str, Conversation] = {}
_session_files: dict[str, Path | None] = {}


def run_email(provider: str, model: str) -> None:
    """Poll inbox and process messages. Runs until interrupted."""
    config = _email_config()
    if config is None:
        print(
            "email: missing config — set TARS_EMAIL_ADDRESS, "
            "TARS_EMAIL_PASSWORD, TARS_EMAIL_ALLOW",
            file=sys.stderr,
        )
        return

    print(
        f"email: polling {config['address']} every {config['poll_interval']}s "
        f"[{provider}:{model}]"
    )
    print(f"email: allowed senders: {', '.join(config['allow'])}")

    imap: imaplib.IMAP4_SSL | None = None

    try:
        while True:
            # Connect / reconnect
            if imap is None:
                try:
                    imap = _connect_imap(config["address"], config["password"])
                    print("email: connected to IMAP")
                except Exception as e:
                    print(f"email: IMAP connect failed: {e}", file=sys.stderr)
                    time.sleep(config["poll_interval"])
                    continue

            try:
                emails = _fetch_unseen(imap, config["allow"])
            except (imaplib.IMAP4.error, OSError) as e:
                print(f"email: fetch failed, reconnecting: {e}", file=sys.stderr)
                imap = None
                continue

            for msg in emails:
                tid = _thread_id(msg)
                body = _extract_body(msg)
                if not body:
                    continue

                _, from_addr = email.utils.parseaddr(msg.get("From", ""))
                subject = msg.get("Subject", "(no subject)")
                print(f"email: [{from_addr}] {subject}")

                # Try slash command first, fall back to model
                slash_reply = _handle_slash_command(body)
                if slash_reply is not None:
                    reply_text = slash_reply
                    print(f"email: [slash] {body.strip().split()[0]}")
                else:
                    # Get or create conversation
                    if tid not in _conversations:
                        _conversations[tid] = Conversation(
                            id=f"email-{tid}", provider=provider, model=model,
                        )
                        _session_files[tid] = _session_path()
                    conv = _conversations[tid]

                    try:
                        reply_text = process_message(conv, body, _session_files[tid])
                    except Exception as e:
                        print(f"email: process failed: {e}", file=sys.stderr)
                        reply_text = "Sorry, I encountered an error processing your message."

                try:
                    _send_reply(config, msg, reply_text)
                    print(f"email: replied to {from_addr}")
                except Exception as e:
                    print(f"email: send failed: {e}", file=sys.stderr)

            time.sleep(config["poll_interval"])

    except KeyboardInterrupt:
        print("\nemail: shutting down...")
    finally:
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
