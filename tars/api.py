import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tars.conversation import Conversation, process_message, save_session
from tars.core import DEFAULT_MODEL, parse_model
from tars.indexer import build_index
from tars.sessions import _session_path

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        build_index()
    except Exception as e:
        print(f"  [warning] index update failed ({type(e).__name__}): {e}", file=sys.stderr)
    yield


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


@app.post("/index")
def index_endpoint() -> dict:
    stats = build_index()
    return stats


_static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
