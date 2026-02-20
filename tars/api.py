import json
import logging
import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tars.conversation import Conversation, process_message, process_message_stream, save_session
from tars.core import DEFAULT_MODEL, parse_model
from tars.indexer import build_index
from tars.memory import save_correction, save_reward
from tars.sessions import _session_path
from tars.tools import run_tool

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger("uvicorn").info(f"tars [{_provider}:{_model}]")
    try:
        build_index()
    except Exception as e:
        print(f"  [warning] index update failed ({type(e).__name__}): {e}", file=sys.stderr)
    yield
    # Save all active conversations on shutdown.
    for conv_id, conv in _conversations.items():
        session_file = _session_files.get(conv_id)
        try:
            save_session(conv, session_file)
        except Exception as e:
            print(f"  [warning] session save failed for {conv_id}: {e}", file=sys.stderr)


app = FastAPI(title="tars", lifespan=lifespan)

_conversations: dict[str, Conversation] = {}
_session_files: dict[str, Path | None] = {}

_model_str = os.environ.get("TARS_MODEL", DEFAULT_MODEL)
_provider, _model = parse_model(_model_str)


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


class FeedbackRequest(BaseModel):
    conversation_id: str
    note: str = ""
    kind: str = "correction"  # "correction" or "reward"


class ToolRequest(BaseModel):
    name: str
    args: dict = {}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    conv_id = req.conversation_id
    if conv_id not in _conversations:
        _conversations[conv_id] = Conversation(
            id=conv_id, provider=_provider, model=_model,
        )
        _session_files[conv_id] = _session_path()
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
            id=conv_id, provider=_provider, model=_model,
        )
        _session_files[conv_id] = _session_path()
    conv = _conversations[conv_id]
    session_file = _session_files.get(conv_id)

    def event_stream():
        for delta in process_message_stream(conv, req.message, session_file):
            yield f"data: {json.dumps({'delta': delta})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

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


@app.post("/tool")
def tool_endpoint(req: ToolRequest) -> dict:
    result = run_tool(req.name, req.args)
    return {"result": result}


@app.post("/index")
def index_endpoint() -> dict:
    stats = build_index()
    return stats


_static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
