"""Telegram channel for tars — bot polling + slash commands."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from tars.brief import build_brief_sections, format_brief_text
from tars.cli import _parse_todoist_add
from tars.config import ModelConfig, model_summary
from tars.conversation import Conversation, process_message, save_session
from tars.format import format_tool_result
from tars.sessions import _session_path
from tars.tools import run_tool

logger = logging.getLogger(__name__)

# Telegram message length limit
_MAX_MESSAGE_LENGTH = 4096

# Keyboard button aliases → slash commands
_KEYBOARD_ALIASES: dict[str, str] = {
    "Brief": "/brief",
    "Weather": "/weather",
    "Forecast": "/forecast",
    "Tasks": "/todoist today",
    "Memory": "/memory",
}

# In-memory conversation state, keyed by chat_id
_conversations: dict[int, Conversation] = {}
_session_files: dict[int, Path | None] = {}
_model_config: ModelConfig | None = None


def _telegram_config() -> dict | None:
    """Load Telegram config from env vars. Returns None if not configured."""
    token = os.environ.get("TARS_TELEGRAM_TOKEN")
    allow = os.environ.get("TARS_TELEGRAM_ALLOW")
    if not token or not allow:
        return None
    user_ids: list[int] = []
    for uid in allow.split(","):
        uid = uid.strip()
        if not uid:
            continue
        try:
            user_ids.append(int(uid))
        except ValueError:
            print(f"telegram: ignoring invalid user ID: {uid}", file=sys.stderr)
    if not user_ids:
        return None
    return {
        "token": token.strip(),
        "allow": user_ids,
    }


def _handle_slash_command(
    text: str, provider: str = "", model: str = "",
) -> str | None:
    """Handle slash commands. Returns reply text, or None if not a command."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split()
    cmd = parts[0]

    # Strip bot username suffix (e.g. /weather@tars_bot → /weather)
    if "@" in cmd:
        cmd = cmd.split("@")[0]

    try:
        if cmd == "/todoist":
            sub = parts[1] if len(parts) > 1 else ""
            if sub == "add" and len(parts) > 2:
                raw_text = " ".join(parts[2:])
                has_flags = any(p.startswith("--") for p in parts[2:])
                if has_flags or not provider:
                    args = _parse_todoist_add(parts[2:])
                else:
                    from tars.email import _parse_todoist_natural

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
        elif cmd == "/brief":
            sections = build_brief_sections()
            return format_brief_text(sections)
        else:
            return None  # Not a recognized command — let the model handle it

        raw = run_tool(name, args, quiet=True)
        return format_tool_result(name, raw)
    except Exception as e:
        return f"Tool error: {e}"


def _truncate(text: str, limit: int = _MAX_MESSAGE_LENGTH) -> str:
    """Truncate text to Telegram's message length limit."""
    if len(text) <= limit:
        return text
    return text[: limit - 15] + "\n...(truncated)"


def _get_keyboard():
    """Build the persistent reply keyboard."""
    from telegram import ReplyKeyboardMarkup

    return ReplyKeyboardMarkup(
        [["Brief", "Weather", "Forecast"], ["Tasks", "Memory"]],
        resize_keyboard=True,
    )


async def _cmd_start(update, context) -> None:
    """Handle /start — welcome message + keyboard."""
    await update.message.reply_text(
        "tars online. Use the keyboard or type a message.",
        reply_markup=_get_keyboard(),
    )


async def _cmd_help(update, context) -> None:
    """Handle /help — list commands."""
    help_text = (
        "/todoist add|today|upcoming|complete\n"
        "/weather — current conditions\n"
        "/forecast — hourly forecast\n"
        "/memory — show persistent memory\n"
        "/remember <section> <text>\n"
        "/note <text> — append to daily note\n"
        "/capture <url> [--raw]\n"
        "/brief — daily briefing digest\n"
        "/clear — reset conversation\n"
        "/help — this message"
    )
    await update.message.reply_text(help_text)


async def _cmd_clear(update, context) -> None:
    """Handle /clear — save session and reset conversation."""
    chat_id = update.effective_chat.id
    if chat_id in _conversations:
        conv = _conversations[chat_id]
        try:
            save_session(conv, _session_files.get(chat_id))
        except Exception:
            pass
        del _conversations[chat_id]
        _session_files.pop(chat_id, None)
    await update.message.reply_text("Conversation cleared.")


async def _handle_message(update, context) -> None:
    """Handle text messages — keyboard aliases, slash commands, chat."""
    global _model_config
    if _model_config is None:
        await update.message.reply_text("Bot not configured.")
        return

    text = update.message.text
    if not text:
        return

    # Resolve keyboard aliases
    if text in _KEYBOARD_ALIASES:
        text = _KEYBOARD_ALIASES[text]

    provider = _model_config.primary_provider
    model = _model_config.primary_model

    # Try slash command
    if text.startswith("/"):
        # Skip /start, /help, /clear — handled by dedicated handlers
        cmd = text.split()[0].split("@")[0]
        if cmd in ("/start", "/help", "/clear"):
            return

        await update.effective_chat.send_action("typing")
        result = await asyncio.to_thread(_handle_slash_command, text, provider, model)
        if result is not None:
            await update.message.reply_text(_truncate(result))
            return
        # Fall through to chat if not a recognized slash command

    # Chat message
    chat_id = update.effective_chat.id
    if chat_id not in _conversations:
        _conversations[chat_id] = Conversation(
            id=f"telegram-{chat_id}",
            provider=provider,
            model=model,
            remote_provider=_model_config.remote_provider,
            remote_model=_model_config.remote_model,
            routing_policy=_model_config.routing_policy,
        )
        _session_files[chat_id] = _session_path()

    conv = _conversations[chat_id]
    await update.effective_chat.send_action("typing")

    try:
        reply = await asyncio.to_thread(
            process_message, conv, text, _session_files[chat_id]
        )
    except Exception as e:
        logger.error("process_message failed: %s", e)
        reply = "Sorry, I encountered an error processing your message."

    await update.message.reply_text(_truncate(reply))


async def send_brief_telegram() -> None:
    """Send the daily brief to all allowed Telegram users."""
    from telegram import Bot

    config = _telegram_config()
    if config is None:
        raise RuntimeError(
            "missing config — set TARS_TELEGRAM_TOKEN and TARS_TELEGRAM_ALLOW"
        )
    sections = build_brief_sections()
    body = format_brief_text(sections)
    bot = Bot(token=config["token"])
    async with bot:
        for user_id in config["allow"]:
            await bot.send_message(chat_id=user_id, text=_truncate(body))


def send_brief_telegram_sync() -> None:
    """Sync wrapper for send_brief_telegram()."""
    asyncio.run(send_brief_telegram())


def run_telegram(model_config: ModelConfig) -> None:
    """Start the Telegram bot. Blocks until interrupted."""
    from telegram import Update
    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

    config = _telegram_config()
    if config is None:
        print(
            "telegram: missing config — set TARS_TELEGRAM_TOKEN and TARS_TELEGRAM_ALLOW",
            file=sys.stderr,
        )
        return

    global _model_config
    _model_config = model_config

    summary = model_summary(model_config)
    print(f"telegram: starting bot [{summary['primary']}]")
    print(f"telegram: allowed users: {config['allow']}")

    user_filter = filters.User(user_id=config["allow"])

    app = ApplicationBuilder().token(config["token"]).build()

    app.add_handler(CommandHandler("start", _cmd_start, filters=user_filter))
    app.add_handler(CommandHandler("help", _cmd_help, filters=user_filter))
    app.add_handler(CommandHandler("clear", _cmd_clear, filters=user_filter))
    # Catch-all for other slash commands and text messages
    app.add_handler(
        MessageHandler(filters.TEXT & user_filter, _handle_message)
    )

    async def _shutdown(app) -> None:
        """Save sessions on shutdown."""
        for chat_id, conv in _conversations.items():
            try:
                save_session(conv, _session_files.get(chat_id))
            except Exception:
                pass

    app.post_shutdown = _shutdown

    print("telegram: polling (ctrl-c to stop)")
    app.run_polling(drop_pending_updates=True)
    print("telegram: stopped")
