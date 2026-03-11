import hmac
import json
import logging
import sys
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import re

from pydantic import BaseModel, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from tars.config import load_model_config, model_summary
from tars.conversation import Conversation, process_message, process_message_stream, save_session
import tars.debug as debug
from tars.format import format_tool_result
from tars.indexer import build_index
from tars.memory import save_correction, save_reward
from tars.search import search as memory_search, search_notes
from tars.sessions import _session_path, list_sessions
from tars.tools import run_tool

load_dotenv()
debug.configure(from_env=True)

_API_TOKEN = os.environ.get("TARS_API_TOKEN", "")

# Paths that don't require auth (static files served via mount).
_PUBLIC_PATHS = {"/", "/index.html", "/dompurify.min.js", "/marked.min.js"}


class _AuthMiddleware(BaseHTTPMiddleware):
    """Optional bearer token auth. Skipped when TARS_API_TOKEN is empty."""

    async def dispatch(self, request: Request, call_next):
        if not _API_TOKEN:
            return await call_next(request)
        # Allow static assets through.
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not hmac.compare_digest(auth, f"Bearer {_API_TOKEN}"):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from tars.services import start_services, stop_services

    logging.getLogger("uvicorn").info(
        f"tars [{_provider}:{_model}] remote={model_summary(_model_config)['remote']}"
    )
    if not _API_TOKEN:
        logging.getLogger("uvicorn").warning(
            "TARS_API_TOKEN is not set — all API endpoints are unauthenticated"
        )
    try:
        build_index()
    except Exception as e:
        print(f"  [warning] index update failed ({type(e).__name__}): {e}", file=sys.stderr)

    mcp_client, runner = start_services(_provider, _model)

    yield

    stop_services(mcp_client, runner)
    # Save all active conversations on shutdown.
    for conv_id, conv in _conversations.items():
        session_file = _session_files.get(conv_id)
        try:
            save_session(conv, session_file)
        except Exception as e:
            print(f"  [warning] session save failed for {conv_id}: {e}", file=sys.stderr)


app = FastAPI(title="tars", lifespan=lifespan)
app.add_middleware(_AuthMiddleware)

_MAX_CONVERSATIONS = 50

_conversations: OrderedDict[str, Conversation] = OrderedDict()
_session_files: dict[str, Path | None] = {}


def _touch_conversation(conv_id: str) -> None:
    """Move conversation to end (most recently used). Evict oldest if over limit."""
    _conversations.move_to_end(conv_id)
    while len(_conversations) > _MAX_CONVERSATIONS:
        oldest_id, oldest_conv = _conversations.popitem(last=False)
        session_file = _session_files.pop(oldest_id, None)
        try:
            save_session(oldest_conv, session_file)
        except Exception as e:
            print(f"  [warning] eviction save failed for {oldest_id}: {e}", file=sys.stderr)

_model_config = load_model_config()
_provider = _model_config.primary_provider
_model = _model_config.primary_model


_CONV_ID_RE = re.compile(r"^[a-zA-Z0-9._-]{1,64}$")


class ChatRequest(BaseModel):
    conversation_id: str
    message: str

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v: str) -> str:
        if not _CONV_ID_RE.fullmatch(v):
            raise ValueError("conversation_id must be 1-64 alphanumeric/._- characters")
        return v


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


class FeedbackRequest(BaseModel):
    conversation_id: str
    note: str = ""
    kind: str = "correction"  # "correction" or "reward"

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v: str) -> str:
        if not _CONV_ID_RE.fullmatch(v):
            raise ValueError("conversation_id must be 1-64 alphanumeric/._- characters")
        return v


class ToolRequest(BaseModel):
    name: str
    args: dict = {}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    conv_id = req.conversation_id
    if conv_id not in _conversations:
        _conversations[conv_id] = Conversation(
            id=conv_id,
            provider=_provider,
            model=_model,
            remote_provider=_model_config.remote_provider,
            remote_model=_model_config.remote_model,
            routing_policy=_model_config.routing_policy,
            channel="web",
        )
        _session_files[conv_id] = _session_path(channel="web")
    _touch_conversation(conv_id)
    conv = _conversations[conv_id]
    session_file = _session_files.get(conv_id)
    reply = process_message(conv, req.message, session_file)
    return ChatResponse(conversation_id=conv_id, reply=reply)


@app.get("/conversations")
def list_conversations() -> dict:
    return {
        "conversations": [
            {"id": conv.id, "message_count": conv.msg_count}
            for conv in _conversations.values()
        ]
    }


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    conv = _conversations.pop(conversation_id, None)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    session_file = _session_files.pop(conversation_id, None)
    save_session(conv, session_file)
    return {"ok": True}


@app.post("/conversations/{conversation_id}/save")
def save_conversation(conversation_id: str) -> dict:
    conv = _conversations.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    session_file = _session_files.get(conversation_id)
    save_session(conv, session_file)
    return {"ok": True}


@app.post("/chat/stream")
def chat_stream_endpoint(req: ChatRequest):
    """Streaming chat via Server-Sent Events.

    Returns a text/event-stream response. Each event is a JSON object:
      data: {"delta": "token"}   — a piece of the response text
      data: {"done": true}       — signals the stream is complete

    The browser reads these with fetch() + ReadableStream, appending each
    delta to the message div so tokens appear as they arrive.
    """
    conv_id = req.conversation_id
    if conv_id not in _conversations:
        _conversations[conv_id] = Conversation(
            id=conv_id,
            provider=_provider,
            model=_model,
            remote_provider=_model_config.remote_provider,
            remote_model=_model_config.remote_model,
            routing_policy=_model_config.routing_policy,
            channel="web",
        )
        _session_files[conv_id] = _session_path(channel="web")
    _touch_conversation(conv_id)
    conv = _conversations[conv_id]
    session_file = _session_files.get(conv_id)

    def event_stream():
        for delta in process_message_stream(conv, req.message, session_file):
            yield f"data: {json.dumps({'delta': delta})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        actual = f"{conv.last_provider}:{conv.last_model}" if conv.last_provider else f"{_provider}:{_model}"
        yield f"data: {json.dumps({'meta': {'model': actual}})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest) -> dict:
    conv = _conversations.get(req.conversation_id)
    if not conv or len(conv.messages) < 2:
        raise HTTPException(status_code=400, detail="No messages to flag")
    user_msg = conv.messages[-2]["content"]
    assistant_msg = conv.messages[-1]["content"]
    fn = save_reward if req.kind == "reward" else save_correction
    result = fn(user_msg, assistant_msg, req.note)
    return {"ok": True, "message": result}


_ALLOWED_TOOLS = {
    "todoist_add_task", "todoist_today", "todoist_upcoming", "todoist_complete_task",
    "weather_now", "weather_forecast",
    "memory_recall", "memory_remember", "memory_update", "memory_forget", "memory_search",
    "note_daily",
    "notes_search",
    "web_read",
    "strava_activities", "strava_user", "strava_summary", "strava_analysis",
}


@app.post("/tool")
def tool_endpoint(req: ToolRequest) -> dict:
    if req.name not in _ALLOWED_TOOLS:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {req.name}")
    raw = run_tool(req.name, req.args, quiet=True)
    formatted = format_tool_result(req.name, raw)
    return {"result": formatted}


_VALID_SEARCH_MODES = {"hybrid", "fts", "vec"}


@app.get("/search")
def search_endpoint(q: str = "", mode: str = "hybrid", limit: int = 10) -> dict:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Missing query parameter 'q'")
    if mode not in _VALID_SEARCH_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    limit = min(max(limit, 1), 100)
    results = memory_search(q, mode=mode, limit=limit)
    return {
        "results": [
            {
                "content": r.content,
                "score": r.score,
                "file_path": r.file_path,
                "file_title": r.file_title,
                "memory_type": r.memory_type,
                "start_line": r.start_line,
                "end_line": r.end_line,
            }
            for r in results
        ]
    }


@app.get("/find")
def find_endpoint(q: str = "", limit: int = 10) -> dict:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Missing query parameter 'q'")
    limit = min(max(limit, 1), 100)
    results = search_notes(q, limit=limit)
    return {
        "results": [
            {
                "content": r.content,
                "score": r.score,
                "file_path": r.file_path,
                "file_title": r.file_title,
                "memory_type": r.memory_type,
                "start_line": r.start_line,
                "end_line": r.end_line,
            }
            for r in results
        ]
    }


_BRIEF_LABEL_TO_API_KEY = {
    "tasks": "todoist_today",
    "weather": "weather_now",
    "forecast": "weather_forecast",
    "strava": "strava",
}


@app.get("/brief")
def brief_endpoint() -> dict:
    from tars.brief import build_brief_sections

    sections = {
        _BRIEF_LABEL_TO_API_KEY.get(label, label): content
        for label, content in build_brief_sections()
    }
    return {"sections": sections}


@app.get("/stats")
def stats_endpoint() -> dict:
    from tars.db import db_stats
    from tars.sessions import session_count
    stats = db_stats()
    stats["sessions"] = session_count()
    return stats


@app.get("/model")
def model_endpoint() -> dict:
    return model_summary(_model_config)


@app.get("/conversations/{conversation_id}/messages")
def conversation_messages_endpoint(conversation_id: str) -> dict:
    conv = _conversations.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "messages": [
            {"role": m.get("role", "unknown"), "content": m.get("content", "")}
            for m in conv.messages
        ]
    }


@app.get("/sessions")
def sessions_endpoint(limit: int = 10) -> dict:
    limit = min(max(limit, 1), 100)
    sessions = list_sessions(limit=limit)
    return {
        "sessions": [
            {
                "date": s.date, "title": s.title,
                "filename": s.filename, "channel": s.channel,
            }
            for s in sessions
        ]
    }


@app.get("/sessions/search")
def session_search_endpoint(q: str = "", limit: int = 10) -> dict:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Missing query parameter 'q'")
    limit = min(max(limit, 1), 100)
    results = memory_search(q, mode="hybrid", limit=limit)
    episodic = [r for r in results if r.memory_type == "episodic"]
    return {
        "results": [
            {
                "content": r.content,
                "score": r.score,
                "file_path": r.file_path,
                "file_title": r.file_title,
                "start_line": r.start_line,
                "end_line": r.end_line,
            }
            for r in episodic
        ]
    }


@app.get("/sessions/{filename}")
def session_content_endpoint(filename: str) -> dict:
    from tars.sessions import load_session
    content = load_session(filename)
    if content is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"filename": filename, "content": content}


@app.post("/index")
def index_endpoint() -> dict:
    stats = build_index()
    return stats


@app.get("/mcp")
def mcp_endpoint() -> dict:
    from tars.tools import get_mcp_client

    client = get_mcp_client()
    if client is None:
        return {"servers": []}
    return {"servers": client.list_servers()}


@app.get("/schedule")
def schedule_endpoint() -> dict:
    from tars.commands import get_task_runner
    from tars.scheduler import schedule_list

    os_schedules = schedule_list()
    in_process = []
    runner = get_task_runner()
    if runner is not None:
        in_process = [
            {
                "name": t.name,
                "schedule": t.schedule,
                "action": t.action,
                "deliver": t.deliver,
                "last_run": t.last_run.strftime("%Y-%m-%d %H:%M") if t.last_run else None,
            }
            for t in runner.list_tasks()
        ]
    return {"os": os_schedules, "in_process": in_process}


@app.get("/conversations/{conversation_id}/export")
def export_conversation_endpoint(conversation_id: str) -> dict:
    conv = _conversations.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    from tars.commands import _export_conversation

    return {"markdown": _export_conversation(conv)}


_static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
