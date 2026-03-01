import sys
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from tars.config import ModelConfig
from tars.core import _search_relevant_context, chat, chat_stream
from tars.extractor import extract_facts
from tars.memory import append_daily
from tars.router import route_message
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
    remote_provider: str | None = None
    remote_model: str | None = None
    routing_policy: str = "tool"
    channel: str = ""
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


def _model_config_for(conv: Conversation) -> ModelConfig:
    return ModelConfig(
        primary_provider=conv.provider,
        primary_model=conv.model,
        remote_provider=conv.remote_provider,
        remote_model=conv.remote_model,
        routing_policy=conv.routing_policy,
    )


_FALLBACK_STATUS_CODES = {408, 409, 429, 529}


def _should_fallback(exc: Exception) -> bool:
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status is None:
            return False
        if status in _FALLBACK_STATUS_CODES:
            return True
        if 500 <= status <= 599:
            return True
        return False
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APITimeoutError):
        return True
    return False


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
    route = route_message(user_input, _model_config_for(conv))
    provider, model = route.provider, route.model
    escalated = (provider, model) != (conv.provider, conv.model)
    try:
        reply = chat(
            conv.messages, provider, model,
            search_context=conv.search_context, tool_hints=route.tool_hints,
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
        if not escalated or not _should_fallback(exc):
            raise
        status = getattr(exc, "status_code", "connection")
        print(f"  [router] escalation failed ({status}), falling back to {conv.provider}:{conv.model}", file=sys.stderr)
        reply = chat(
            conv.messages, conv.provider, conv.model,
            search_context=conv.search_context, tool_hints=route.tool_hints,
        )
    conv.messages.append({"role": "assistant", "content": reply})
    conv.msg_count += 1

    _maybe_compact(conv, session_file)
    return reply


def process_message_stream(
    conv: Conversation, user_input: str, session_file: Path | None = None,
) -> Generator[str, None, None]:
    """Streaming version of process_message. Yields text deltas.

    Collects all deltas into the full reply so the conversation history
    gets the complete message, same as the non-streaming path. The caller
    (API endpoint) forwards each delta to the client as an SSE event.
    """
    if not conv.messages and not conv.search_context:
        try:
            conv.search_context = _search_relevant_context(user_input)
        except Exception as e:
            print(f"  [warning] startup search failed: {e}", file=sys.stderr)

    conv.messages.append({"role": "user", "content": user_input})

    # Yield deltas to the caller while accumulating the full reply.
    route = route_message(user_input, _model_config_for(conv))
    provider, model = route.provider, route.model
    escalated = (provider, model) != (conv.provider, conv.model)
    full_reply: list[str] = []
    if escalated:
        # Buffer escalated calls so we can fall back cleanly without emitting
        # partial output followed by a second full response.
        try:
            reply = chat(
                conv.messages, provider, model,
                search_context=conv.search_context, tool_hints=route.tool_hints,
            )
            full_reply = [reply]
            yield reply
        except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            if not _should_fallback(exc):
                raise
            status = getattr(exc, "status_code", "connection")
            print(f"  [router] escalation failed ({status}), falling back to {conv.provider}:{conv.model}", file=sys.stderr)
            for delta in chat_stream(
                conv.messages, conv.provider, conv.model,
                search_context=conv.search_context, tool_hints=route.tool_hints,
            ):
                full_reply.append(delta)
                yield delta
    else:
        for delta in chat_stream(
            conv.messages, provider, model,
            search_context=conv.search_context, tool_hints=route.tool_hints,
        ):
            full_reply.append(delta)
            yield delta

    # Store the complete reply in conversation history.
    reply = "".join(full_reply)
    conv.messages.append({"role": "assistant", "content": reply})
    conv.msg_count += 1

    _maybe_compact(conv, session_file)


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
        try:
            tag = f"[{conv.channel}] " if conv.channel else ""
            append_daily(f"{tag}session compacted — {summary[:80]}")
        except Exception:
            pass
        try:
            tag = f"[{conv.channel}] " if conv.channel else ""
            for fact in extract_facts(new_messages, conv.provider, conv.model):
                append_daily(f"{tag}[extracted] {fact}")
        except Exception:
            pass
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
        try:
            tag = f"[{conv.channel}] " if conv.channel else ""
            append_daily(f"{tag}session saved — {summary[:80]}")
        except Exception:
            pass
        try:
            tag = f"[{conv.channel}] " if conv.channel else ""
            for fact in extract_facts(new_messages, conv.provider, conv.model):
                append_daily(f"{tag}[extracted] {fact}")
        except Exception:
            pass
    except Exception as e:
        print(f"  [warning] session save failed: {e}", file=sys.stderr)
