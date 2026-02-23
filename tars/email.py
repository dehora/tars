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
from tars.config import ModelConfig, model_summary
from tars.conversation import Conversation, process_message, save_session
from tars.format import format_tool_result
from tars.sessions import _session_path
from tars.tools import run_tool

def _parse_todoist_natural(text: str, provider: str, model: str) -> dict:
    """Use the model to extract task fields from natural language."""
    import json as _json

    from tars.core import chat

    messages = [{"role": "user", "content": f"{_TODOIST_PARSE_PROMPT}{text}"}]
    try:
        raw = chat(messages, provider, model, use_tools=False)
        # Extract JSON from response (model might wrap in ```json blocks)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = _json.loads(raw)
        if isinstance(result, dict) and result.get("content"):
            return result
    except Exception as e:
        print(f"  [email] todoist parse failed, using raw text: {e}", file=sys.stderr)
    return {"content": text}


_TODOIST_PARSE_PROMPT = """\
Extract task details from this text. Return ONLY valid JSON with these fields:
- "content": the task description (required)
- "project": project name if mentioned, otherwise omit
- "due": due date if mentioned (e.g. "today", "tomorrow", "friday"), otherwise omit
- "priority": 1-4 if mentioned (4=urgent), otherwise omit

Examples:
"eggs to Groceries" → {"content": "eggs", "project": "Groceries"}
"buy milk --due tomorrow" → {"content": "buy milk", "due": "tomorrow"}
"call dentist p3" → {"content": "call dentist", "priority": 3}
"fix the bike" → {"content": "fix the bike"}

Text: """


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


def _handle_slash_command(
    body: str, provider: str = "", model: str = "",
) -> str | None:
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
                raw_text = " ".join(parts[2:])
                has_flags = any(p.startswith("--") for p in parts[2:])
                if has_flags or not provider:
                    args = _parse_todoist_add(parts[2:])
                else:
                    args = _parse_todoist_natural(raw_text, provider, model)
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
        elif cmd == "/read" and len(parts) >= 2:
            name = "web_read"
            args = {"url": parts[1]}
        elif cmd == "/capture" and len(parts) >= 2:
            from tars.capture import capture as _capture
            raw_flag = "--raw" in parts
            url = next((p for p in parts[1:] if p != "--raw"), "")
            result = _capture(url, provider, model, raw=raw_flag)
            return format_tool_result("capture", result)
        else:
            return None  # Not a recognized command — let the model handle it

        raw = run_tool(name, args, quiet=True)
        return format_tool_result(name, raw)
    except Exception as e:
        return f"Tool error: {e}"


# In-memory conversation state, keyed by thread ID
_conversations: dict[str, Conversation] = {}
_session_files: dict[str, Path | None] = {}


def run_email(model_config: ModelConfig) -> None:
    """Poll inbox and process messages. Runs until interrupted."""
    email_config = _email_config()
    if email_config is None:
        print(
            "email: missing config — set TARS_EMAIL_ADDRESS, "
            "TARS_EMAIL_PASSWORD, TARS_EMAIL_ALLOW",
            file=sys.stderr,
        )
        return

    summary = model_summary(model_config)
    print(
        f"email: polling {email_config['address']} every {email_config['poll_interval']}s "
        f"[{summary['primary']}]"
    )
    print(f"email: allowed senders: {', '.join(email_config['allow'])}")

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

            for msg_num, msg in emails:
                tid = _thread_id(msg)
                body = _extract_body(msg)
                _, from_addr = email.utils.parseaddr(msg.get("From", ""))
                subject = msg.get("Subject", "(no subject)")
                print(f"email: [{from_addr}] {subject}")

                # Try slash command: check subject first, then body
                slash_reply = _handle_slash_command(
                    subject,
                    model_config.primary_provider,
                    model_config.primary_model,
                )
                if slash_reply is None and body:
                    slash_reply = _handle_slash_command(
                        body,
                        model_config.primary_provider,
                        model_config.primary_model,
                    )
                if slash_reply is not None:
                    reply_text = slash_reply
                    print(f"email: [slash] {slash_reply[:60]}")
                elif not body:
                    # Mark seen even if we skip (no body, not a command).
                    imap.store(msg_num, "+FLAGS", "\\Seen")
                    continue
                else:
                    # Get or create conversation
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
                        print(f"email: process failed: {e}", file=sys.stderr)
                        reply_text = "Sorry, I encountered an error processing your message."

                try:
                    _send_reply(email_config, msg, reply_text)
                    # Mark as Seen only after successful reply.
                    imap.store(msg_num, "+FLAGS", "\\Seen")
                    print(f"email: replied to {from_addr}")
                except Exception as e:
                    print(f"email: send failed (will retry): {e}", file=sys.stderr)

            time.sleep(email_config["poll_interval"])

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
