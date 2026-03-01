"""Telegram channel for tars — bot polling + slash commands."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from tars.brief import build_brief_sections, format_brief_text
from tars.commands import dispatch
from tars.config import ModelConfig, model_summary
from tars.conversation import Conversation, process_message, save_session
from tars.sessions import _session_path

logger = logging.getLogger(__name__)

# Telegram message length limit
_MAX_MESSAGE_LENGTH = 4096

# Keyboard button aliases → slash commands
_KEYBOARD_ALIASES: dict[str, str] = {
    "Brief": "/brief",
    "Weather": "/weather",
    "Forecast": "/forecast",
    "Tasks": "/todoist today",
    "Todoist": "/todoist",
    "Note": "/note",
    "Remember": "/remember",
    "Capture": "/capture",
    "Search": "/search",
    "Sessions": "/sessions",
    "Find": "/find",
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


def _truncate(text: str, limit: int = _MAX_MESSAGE_LENGTH) -> str:
    """Truncate text to Telegram's message length limit."""
    if len(text) <= limit:
        return text
    return text[: limit - 15] + "\n...(truncated)"


def _get_keyboard():
    """Build the persistent reply keyboard."""
    from telegram import ReplyKeyboardMarkup

    return ReplyKeyboardMarkup(
        [["Brief", "Weather", "Forecast"], ["Tasks", "Todoist", "Note"], ["Remember", "Capture", "Search"], ["Sessions", "Find"]],
        resize_keyboard=True,
    )


async def _cmd_start(update, context) -> None:
    """Handle /start — welcome message + keyboard."""
    await update.message.reply_text(
        "tars online. Use the keyboard or type a message.",
        reply_markup=_get_keyboard(),
    )


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
        # Strip @botname suffix (e.g. /weather@tars_bot → /weather)
        first = text.split()[0]
        if "@" in first:
            cmd_stripped = first.split("@")[0]
            rest = text[len(first):]
            text = cmd_stripped + rest

        # Skip /start — handled by dedicated handler
        cmd = text.split()[0]
        if cmd == "/start":
            return

        # Get or create conversation for export support
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

        await update.effective_chat.send_action("typing")
        conv = _conversations[chat_id]
        ctx = {"channel": "telegram"}
        result = await asyncio.to_thread(
            dispatch, text, provider, model, conv=conv, context=ctx,
        )
        if result is not None:
            if result == "__clear__":
                try:
                    save_session(conv, _session_files.get(chat_id))
                except Exception:
                    pass
                del _conversations[chat_id]
                _session_files.pop(chat_id, None)
                await update.message.reply_text("Conversation cleared.")
            else:
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

    from tars.commands import set_task_runner
    from tars.mcp import MCPClient, _load_mcp_config
    from tars.taskrunner import TaskRunner
    from tars.tools import set_mcp_client

    global _model_config
    _model_config = model_config

    summary = model_summary(model_config)
    print(f"telegram: starting bot [{summary['primary']}]")
    print(f"telegram: allowed users: {config['allow']}")

    mcp_client = None
    mcp_config = _load_mcp_config()
    if mcp_config:
        mcp_client = MCPClient(mcp_config)
        mcp_client.start()
        set_mcp_client(mcp_client)
        from tars.router import update_tool_names
        update_tool_names({t["name"] for t in mcp_client.discover_tools()})

    runner = TaskRunner(model_config.primary_provider, model_config.primary_model)
    runner.start()
    set_task_runner(runner)

    user_filter = filters.User(user_id=config["allow"])
    private_filter = filters.ChatType.PRIVATE & user_filter

    app = ApplicationBuilder().token(config["token"]).build()

    app.add_handler(CommandHandler("start", _cmd_start, filters=private_filter))
    # All other commands (including /help, /clear) are handled by dispatch
    app.add_handler(
        MessageHandler(filters.TEXT & private_filter, _handle_message)
    )

    async def _shutdown(app) -> None:
        """Save sessions, stop MCP client, and stop task runner on shutdown."""
        runner.stop()
        set_task_runner(None)
        if mcp_client:
            mcp_client.stop()
            set_mcp_client(None)
        for chat_id, conv in _conversations.items():
            try:
                save_session(conv, _session_files.get(chat_id))
            except Exception:
                pass

    app.post_shutdown = _shutdown

    print("telegram: polling (ctrl-c to stop)")
    app.run_polling(drop_pending_updates=True)
    print("telegram: stopped")
