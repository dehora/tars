import sys
from dataclasses import dataclass, field
from pathlib import Path

from tars.core import _search_relevant_context, chat
from tars.sessions import (
    SESSION_COMPACTION_INTERVAL,
    _save_session,
    _summarize_session,
)


@dataclass
class Conversation:
    id: str
    provider: str
    model: str
    messages: list[dict] = field(default_factory=list)
    search_context: str = ""
    msg_count: int = 0
    last_compaction: int = 0
    last_compaction_index: int = 0
    cumulative_summary: str = ""


def _merge_summary(existing: str, new: str) -> str:
    if not existing:
        return new
    if not new:
        return existing
    return f"{existing.rstrip()}\n{new.lstrip()}"


def process_message(
    conv: Conversation, user_input: str, session_file: Path | None = None,
) -> str:
    """Send a user message through the conversation and return the reply."""
    # Search on first message only
    if not conv.messages and not conv.search_context:
        try:
            conv.search_context = _search_relevant_context(user_input)
        except Exception as e:
            print(f"  [warning] startup search failed: {e}", file=sys.stderr)

    conv.messages.append({"role": "user", "content": user_input})
    reply = chat(
        conv.messages, conv.provider, conv.model, search_context=conv.search_context,
    )
    conv.messages.append({"role": "assistant", "content": reply})
    conv.msg_count += 1

    _maybe_compact(conv, session_file)
    return reply


def _maybe_compact(conv: Conversation, session_file: Path | None) -> None:
    """Run session compaction if the interval has been reached."""
    if not session_file:
        return
    if conv.msg_count - conv.last_compaction < SESSION_COMPACTION_INTERVAL:
        return
    try:
        new_messages = conv.messages[conv.last_compaction_index:]
        summary = _summarize_session(
            new_messages, conv.provider, conv.model,
            previous_summary=conv.cumulative_summary,
        )
        conv.cumulative_summary = _merge_summary(conv.cumulative_summary, summary)
        _save_session(session_file, conv.cumulative_summary, is_compaction=True)
        conv.last_compaction = conv.msg_count
        conv.last_compaction_index = len(conv.messages)
    except Exception as e:
        print(f"  [warning] session compaction failed: {e}", file=sys.stderr)


def save_session(conv: Conversation, session_file: Path | None) -> None:
    """Save final session summary on exit."""
    if not session_file or not conv.messages:
        return
    if conv.msg_count <= conv.last_compaction:
        return
    try:
        new_messages = conv.messages[conv.last_compaction_index:]
        summary = _summarize_session(
            new_messages, conv.provider, conv.model,
            previous_summary=conv.cumulative_summary,
        )
        conv.cumulative_summary = _merge_summary(conv.cumulative_summary, summary)
        _save_session(session_file, conv.cumulative_summary)
    except Exception as e:
        print(f"  [warning] session save failed: {e}", file=sys.stderr)
