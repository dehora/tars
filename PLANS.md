# Implementation Plans

## Build sequence: Memory and automation layer

The following four plans form a coherent system. Each piece makes the others more useful. The dependency chain determines the build order.

```
Layer 1 (parallel, no deps)     Layer 2 (depends on L1)
┌─────────────────────┐         ┌──────────────────────────┐
│ ✓ Centralized dispatch│───────▶│ Scheduler                │
└─────────────────────┘         │ (synthetic commands via   │
                                │  dispatch, manages daily  │
┌─────────────────────┐         │  file rotation)           │
│ ✓ Daily memory files │───┬───▶└──────────────────────────┘
└─────────────────────┘   │
                          │     ┌──────────────────────────┐
                          └────▶│ ✓ Memory extraction       │
                                │ (writes to daily file,    │
                                │  /review promotes to      │
                                │  permanent memory)        │
                                └──────────────────────────┘
```

### Layer 1: Foundation — DONE

**1a. Centralized dispatch** — Done (33adc69). Collapsed duplicated command handling across cli/email/telegram into a single registry in `commands.py`.

**1b. Daily memory files** — Done (c850983). `YYYY-MM-DD.md` daily files in memory dir with timestamped event logging. Loads as `<daily-context>` in system prompt.

### Layer 2: Builds on foundation

**2a. Memory extraction** — Done (f9a7d8e). `tars/extractor.py` extracts facts after compaction/save via `chat(use_tools=False)`, writes to daily file tagged `[extracted]`. Controlled by `TARS_AUTO_EXTRACT` env var. Caps at 5 facts per extraction, skips trivial conversations (<3 user messages).

**2b. Scheduler** — Depends on daily memory (1b), benefits from centralized dispatch (1a). Background thread fires scheduled tasks as synthetic messages through `process_message()`. Replaces the hardcoded email brief with configuration. Enables morning briefs, end-of-day review, recurring task checks. Estimated scope: ~1-2 sessions.

### Future (Layer 3, independent)

**MCP integration** — Independent of the above but benefits from centralized dispatch (MCP tools route through the same dispatcher). Adds config-driven external tool servers. Larger scope, new dependency.

---

## Plan: Hybrid search over memory index

### Context
The indexing pipeline (chunker → embeddings → sqlite-vec) is built and working. Now we need the query side: hybrid search combining FTS5 keyword search with sqlite-vec vector KNN, fused via Reciprocal Rank Fusion (RRF). This replaces the fixed recent-sessions loading with relevance-based retrieval at conversation start, and gives the agent a `memory_search` tool for mid-conversation lookups.

### Approach: FTS5 + sqlite-vec, zero new dependencies

FTS5 is built into SQLite. A standalone FTS5 table (`chunks_fts`) is populated alongside `vec_chunks` during indexing. Content-sync mode (`content=vec_chunks`) won't work because SQLite doesn't fire triggers on virtual table writes. Duplicating the text in FTS5 is fine — these are small personal memory files.

---

### Step 1: FTS5 schema + sync in `tars/db.py`

#### New FTS5 table
Add to `init_db` flow (after vec_chunks creation):
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, tokenize='porter unicode61'
);
```
`porter` gives stemming ("running" → "run"). Rowids align with `vec_chunks` rowids.

#### Backfill helper for existing DBs
```python
def _ensure_fts(conn):
```
If `chunks_fts` doesn't exist, create it and backfill from `vec_chunks`. Called at end of `init_db`.

#### Modify `insert_chunks`
After each `vec_chunks` INSERT, capture `cur.lastrowid` and INSERT into `chunks_fts` with matching rowid:
```python
cur = conn.execute("INSERT INTO vec_chunks ...", (...))
conn.execute("INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
             (cur.lastrowid, chunk.content))
```

#### Modify `delete_chunks_for_file` and `delete_file`
Before deleting from `vec_chunks`, SELECT rowids, then DELETE matching rows from `chunks_fts`:
```python
rows = conn.execute("SELECT rowid FROM vec_chunks WHERE file_id = ?", (file_id,)).fetchall()
if rows:
    placeholders = ",".join("?" * len(rows))
    conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", [r["rowid"] for r in rows])
conn.execute("DELETE FROM vec_chunks WHERE file_id = ?", (file_id,))
```

---

### Step 2: New module `tars/search.py`

#### SearchResult dataclass
```python
@dataclass(frozen=True, slots=True)
class SearchResult:
    content: str
    score: float            # RRF score, normalized 0-1
    file_path: str
    file_title: str | None
    memory_type: str | None # "semantic" | "procedural" | "episodic"
    start_line: int
    end_line: int
    chunk_rowid: int        # for deduplication
```

#### Low-level search functions

**`search_vec(conn, query_embedding, *, limit=20) -> list[int]`**
sqlite-vec KNN query, returns chunk rowids in distance order:
```sql
SELECT rowid, distance FROM vec_chunks
WHERE embedding MATCH ? AND k = ?
```
Query vector serialized via `_serialize_f32`.

**`search_fts(conn, query, *, limit=20) -> list[int]`**
FTS5 BM25 query, returns chunk rowids in rank order:
```sql
SELECT rowid, rank FROM chunks_fts
WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
```
Query sanitized by quoting each token to avoid FTS5 syntax errors (`"each" "token"` — implicit AND).

**`_sanitize_fts_query(query: str) -> str`**
Split on whitespace, quote each token, rejoin. Prevents colons/parens/operators from causing FTS5 parse errors.

#### RRF fusion

**`_reciprocal_rank_fusion(*ranked_lists, k=60) -> list[tuple[int, float]]`**
- For each result at position `rank` (1-indexed) in each list: `score += 1/(k + rank)`
- Normalize to 0-1 by dividing by `n_lists / (k + 1)` (theoretical max)
- Return `[(rowid, score)]` sorted descending by score

#### Main search function

**`search(query, *, model="qwen3-embedding:0.6b", limit=10, min_score=0.0, mode="hybrid") -> list[SearchResult]`**

Flow:
1. Open connection via `_db_path()` + `_connect()` — return `[]` if no DB
2. If mode includes vec: `embed(query, model=model)[0]` → `search_vec(conn, vec, limit=limit*2)`
3. If mode includes fts: `search_fts(conn, query, limit=limit*2)`
4. Fuse with RRF (even for single-mode — gives normalized scores)
5. Filter by `min_score`, trim to `limit`
6. Hydrate results by joining `vec_chunks` ↔ `files` on `file_id`
7. Close connection, return `list[SearchResult]`

Over-fetch `limit*2` from each source so RRF fusion has enough candidates.

#### Tool handler

**`_run_search_tool(name, args) -> str`**
JSON wrapper around `search()`. Returns `{"results": [...]}` or `{"error": "..."}`.

---

### Step 3: Tool integration in `tars/tools.py`

Add `memory_search` tool to `ANTHROPIC_TOOLS` (OLLAMA_TOOLS derives automatically):
- `query` (required string): what to search for
- `limit` (optional int): max results, default 5

Route in `run_tool`: `if name == "memory_search": return _run_search_tool(name, args)` (imported from `tars.search`).

---

### Step 4: Startup search in `tars/core.py` and `tars/cli.py`

#### `core.py` changes

**`_search_relevant_context(opening_message, limit=5) -> str`**
Calls `search(opening_message, limit=limit, min_score=0.25)`, formats results as labeled blocks.

**`_build_system_prompt(*, search_context="")`**
Add `search_context` kwarg. Replace the `_load_recent_sessions()` call with the passed-in search context. `<recent-sessions>` tag becomes `<relevant-context>`. Memory.md stays always-loaded (it's small and semantic).

**`chat()`, `chat_anthropic()`, `chat_ollama()`**
Thread `search_context: str = ""` kwarg through to `_build_system_prompt(search_context=search_context)`.

#### `cli.py` changes

**Single-shot mode:** before calling `chat()`, run `_search_relevant_context(message)`, pass result through.

**REPL mode:** search on the first user message only. Store the result and pass it to `chat()` on the first call. Subsequent turns use the `memory_search` tool if needed.

Both wrapped in try/except with warning to stderr on failure.

---

### Step 5: Tests in `tests/test_search.py`

Mock setup: same pattern as test_indexer.py (module-level ollama mock, `mock.patch.object(embeddings, "ollama")`, real sqlite-vec + temp dirs).

**Key cases:**
- `_sanitize_fts_query`: handles special chars, empty input
- `_reciprocal_rank_fusion`: merges overlapping lists, scores normalized 0-1, single list works
- `search_vec`: returns rowids in distance order for known embeddings
- `search_fts`: returns rowids matching keyword query
- `search` hybrid end-to-end: mock `embed()`, insert chunks, verify results contain expected content
- `search` mode="vec" and mode="fts": single-source modes work
- `search` with `min_score` filter: low-scoring results excluded
- `search` empty/missing DB: returns `[]`
- FTS sync on delete: after `delete_chunks_for_file`, FTS table is also cleaned
- FTS backfill: DB with vec_chunks but no chunks_fts gets FTS populated on `_ensure_fts`
- `_run_search_tool`: returns well-formed JSON
- Startup search: `_search_relevant_context` returns formatted string or empty

---

### Implementation order

1. **FTS5 in db.py** — schema, `_ensure_fts`, modify `insert_chunks`/`delete_chunks_for_file`/`delete_file`. Run existing tests to confirm no regressions.
2. **search.py** — `SearchResult`, `_sanitize_fts_query`, `search_vec`, `search_fts`, `_reciprocal_rank_fusion`, `search()`, `_run_search_tool`.
3. **test_search.py** — all search tests.
4. **tools.py** — `memory_search` tool definition + routing.
5. **core.py** — `_search_relevant_context`, modify `_build_system_prompt`, thread through `chat()`.
6. **cli.py** — wire startup search for first message.

### Verification
```
uv run python -m unittest discover -s tests -v
uv run tars index
uv run tars "what do you know about me?"
```

### Design notes
- **No new dependencies.** FTS5 built into SQLite, everything else exists.
- **Distance metric:** L2 (sqlite-vec default). Works fine for KNN ranking.
- **RRF k=60:** standard value from the original paper. No tuning needed.
- **Graceful degradation:** empty DB, missing dir, no index → empty results, no exceptions.
- **`_load_recent_sessions` removed from startup path** — replaced by search-based retrieval. The function stays in memory.py for now (session compaction still uses it).

---

## Plan: HTTP API for tars

### Context

tars needs an HTTP API layer so frontends (web, WhatsApp, future channels) can talk to it. Currently only the CLI REPL exists. The API is the shared surface — each frontend becomes a thin client that POSTs messages and gets replies.

WhatsApp via Baileys was the original goal but Meta's ban risk on unofficial clients is high (connection-level, not just volume). Building the API first means the plumbing is ready when we revisit WhatsApp or add a web UI.

### Design decisions

**Framework: FastAPI**
- Lightweight, async, auto-generates OpenAPI docs. Already the default for Python APIs. One dependency add (`fastapi[standard]` bundles uvicorn).

**State: in-memory dict**
- This is a personal single-user bot. Conversations are ephemeral — if the server restarts, start fresh. No database for conversation state.
- A dict keyed by `conversation_id` holding messages, search_context, compaction state.
- If persistence becomes needed later, swap the dict for SQLite. But not yet.

**Conversation lifecycle**
- Auto-create on first message to a new `conversation_id`. No explicit create endpoint.
- Client picks the ID (e.g. `"cli"`, `"whatsapp"`, `"web"`, or a UUID). This lets each channel maintain its own conversation naturally.
- Optional DELETE to clear a conversation.

**Session compaction**
- Same logic as REPL: compact every N messages, save summary to session file.
- Extract the compaction logic from `cli.py` into a shared helper rather than duplicating it. This is the one place where extraction is justified — the REPL and API both need identical compaction behaviour.

### Files to modify

**`tars/conversation.py` (new)** — conversation state + compaction
- `Conversation` dataclass: id, provider, model, messages, search_context, compaction state
- `process_message(conv, user_input, session_file)` — appends user message, calls `chat()`, appends reply, triggers compaction if needed, returns reply text
- `_maybe_compact(conv, session_file)` — compaction check + execute
- `save_session(conv, session_file)` — final session save

**`tars/api.py` (new)** — FastAPI app
- `POST /chat` — send message, get reply (auto-creates conversations)
- `GET /conversations` — list active conversations
- `DELETE /conversations/{id}` — save session and remove
- `POST /index` — trigger memory reindex

**`tars/cli.py`** — refactor REPL to use Conversation, add `serve` subcommand

**`pyproject.toml`** — add `fastapi[standard]>=0.115.0`

### Implementation order

1. `tars/conversation.py` — Conversation dataclass + `process_message()` + compaction helpers
2. `tars/cli.py` — refactor REPL to use Conversation (verify existing tests still pass)
3. `tars/api.py` — FastAPI app with /chat, /conversations, /conversations/{id}, /index
4. `tars/cli.py` — add `serve` subcommand
5. `pyproject.toml` — add fastapi dependency
6. `tests/test_api.py` — test API endpoints using FastAPI TestClient
7. `tests/test_conversation.py` — test process_message, compaction logic
8. Run all tests

### Verification
```
uv run python -m unittest discover -s tests -v
uv run tars serve &
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"conversation_id": "test", "message": "hello"}'
curl http://localhost:8000/conversations
curl -X DELETE http://localhost:8000/conversations/test
uv run tars   # REPL still works as before
```

---

## Plan: `/review` corrections feedback loop

### Context

Tars captures wrong (`/w`) and good (`/r`) responses to `corrections.md` and `rewards.md` in the memory dir. Currently this data is write-only — nobody reads it. A `/review` command distills patterns from the corrections into actionable procedural memory rules, closing the feedback loop.

The goal: turn "tars keeps doing X wrong" into a procedural memory entry like "when the user says X, use the Y tool" — without manual effort.

### Design

**`/review`** reads `corrections.md` and `rewards.md`, sends them to the model with a prompt asking it to:
1. Identify patterns in the corrections (common misroutes, bad responses)
2. Note what worked well from the rewards
3. Propose concise procedural rules (one-liners in the style of `Procedural.md`)
4. Present the rules for user approval before writing

**Flow:**
```
you> /review
reviewing 3 corrections, 2 rewards...

suggested rules:
  1. when user says "add X to Y", route to todoist, not memory
  2. weather requests should use weather_now, not chat
  3. (reward) natural language todoist routing works well for "remind me" phrases

apply? (y/n/edit)
you> y
  2 rules added to Procedural.md
  corrections.md archived
```

**Key decisions:**
- Uses the model (via `chat()`) to analyze — this is a model-assisted review, not pure automation
- Shows proposed rules before writing — user stays in control
- Archives corrections after review (rename to `corrections-{timestamp}.md`) so they don't accumulate and get re-reviewed
- Rewards get archived too since the patterns have been extracted
- Works in REPL only (not web) — this is an admin/maintenance command, interactive approval is important

### Files to modify

#### 1. `tars/memory.py` — add `load_feedback()` and `archive_feedback()`

```python
def load_feedback() -> tuple[str, str]:
    """Load corrections.md and rewards.md content."""
    md = _memory_dir()
    if not md:
        return "", ""
    corrections = ""
    rewards = ""
    cp = md / "corrections.md"
    rp = md / "rewards.md"
    if cp.exists():
        corrections = cp.read_text(encoding="utf-8", errors="replace")
    if rp.exists():
        rewards = rp.read_text(encoding="utf-8", errors="replace")
    return corrections, rewards

def archive_feedback() -> None:
    """Rename corrections.md and rewards.md with timestamp suffix."""
    md = _memory_dir()
    if not md:
        return
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    for name in ("corrections.md", "rewards.md"):
        p = md / name
        if p.exists():
            p.rename(md / f"{p.stem}-{ts}.md")
```

#### 2. `tars/cli.py` — add `/review` command

In the REPL loop, handle `/review`:
- Call `load_feedback()` to get corrections + rewards text
- If both empty, print "nothing to review" and return
- Build a review prompt asking the model to propose procedural rules
- Call `chat()` with the review prompt (using the conversation's provider/model)
- Print the model's suggestions
- Prompt `apply? (y/n)` via `input()`
- On `y`: parse out the rules, append each to `Procedural.md` via `_append_to_file()`, then `archive_feedback()`
- On `n`: do nothing

The review prompt:

```
Review these corrections (wrong responses) and rewards (good responses) from a tars AI assistant session.

<corrections>
{corrections}
</corrections>

<rewards>
{rewards}
</rewards>

Based on the patterns you see:
1. Identify what went wrong and propose concise procedural rules to prevent it
2. Note what worked well that should be reinforced
3. Output ONLY the rules as a bulleted list, one per line, starting with "- "
4. Each rule should be a short, actionable instruction (like "route 'add X to Y' requests to todoist, not memory")
5. Skip rules that are too generic to be useful
```

#### 3. `/help` — add `/review` to the output

Add under the feedback section:
```
    /review          review corrections and apply learnings
```

#### 4. Tests — `tests/test_memory.py`

- `test_load_feedback_reads_files`: create corrections.md and rewards.md, verify content returned
- `test_load_feedback_empty`: no files, returns empty strings
- `test_load_feedback_no_memory_dir`: returns empty strings
- `test_archive_feedback_renames_files`: verify files renamed with timestamp
- `test_archive_feedback_no_files`: no error when files don't exist

### Verification
```
uv run python -m unittest discover -s tests -v
# Create some test corrections: chat with tars, /w on a bad response
# Run /review, verify model proposes rules
# Accept with y, check Procedural.md has new entries
# Verify corrections.md is archived with timestamp
```

## Plan: Bug fixes — tool leakage, web UI gaps

### Context

After shipping session browsing, routing confidence, web search, and daily notes in one session, testing revealed several issues. Some were code bugs, others were prompt/state problems. The interesting ones were all variations of "tools firing in contexts where they shouldn't."

### Bug analysis

**1. note_daily called during session save (CLI + web)**

Root cause: `_summarize_session()` in `sessions.py` calls `chat()`, which passes the full tool list to the provider. The summarization prompt asks the model to summarize a conversation — but the model sees the tools and sometimes decides to act on conversation content instead of summarizing it. For example, if the conversation mentioned "jot this down", the model would call `note_daily` during summarization.

The fix was structural, not prompt-based. Added `use_tools: bool = True` to `chat()`, `chat_anthropic()`, and `chat_ollama()`. When `False`, no tools are passed to the provider API — the model physically cannot call them. `_summarize_session()` now passes `use_tools=False`.

Why not fix with a prompt? Because "please don't call tools" in a system prompt is a suggestion. Removing tools from the API call is a guarantee. Defence in depth: if a code path should never use tools, don't offer them.

**2. /review calling todoist on web**

Root cause: the web UI didn't handle `/review`. Unrecognised commands fell through to the chat endpoint, where the model received "/review" as a user message. The model interpreted this as a request and called todoist. Same issue for `/tidy`.

Fix: intercept `/review` and `/tidy` in the web UI JS before they reach the chat endpoint. These are interactive CLI commands (they need `input()` for approval), so the web UI shows "CLI only" instead.

**3. /help missing from web**

Straightforward omission — `/help` was implemented in the CLI REPL but never added to the web UI. Added a handler that renders the full command list as markdown, matching CLI parity.

**4. Web UI doesn't know its own capabilities**

The model's system prompt describes tools (todoist, weather, memory, notes) but not slash commands. Users typing "what can you do?" get tool descriptions but not `/search`, `/sessions`, etc. The `/help` command fixes discoverability. The system prompt intentionally doesn't list slash commands because they're client-side — the model doesn't need to know about them.

### Key insight

The tool leakage bug (#1) is a class of problem worth watching for: any code path that calls `chat()` for an internal purpose (summarization, review, tidy) risks the model calling tools as a side effect. The `use_tools=False` parameter makes this opt-in rather than default. The existing `_handle_review()` and `_handle_tidy()` in `cli.py` also call `chat()` but those are intentional model interactions where tool calls would be harmless (the model is analysing text, not acting on user requests). If they ever cause problems, same fix applies.

### Changes

| File | Change |
|------|--------|
| `tars/core.py` | Added `use_tools` param to `chat()`, `chat_anthropic()`, `chat_ollama()` |
| `tars/sessions.py` | `_summarize_session()` passes `use_tools=False` |
| `tars/static/index.html` | Added `/help` handler, `/review` and `/tidy` interception |
| `tests/test_core.py` | Updated `test_chat_passes_search_context` for new kwarg |
| `tests/test_sessions.py` | Updated `fake_chat` signatures with `**kwargs` |

---

## Plan: Daily memory files

### Context

Tars has three memory tiers: semantic (Memory.md — durable facts), procedural (Procedural.md — behavioural rules), and episodic (session logs — full conversation transcripts). There's a gap between session logs (too granular, per-conversation) and semantic/procedural memory (too distilled, manually maintained). A daily memory layer bridges this — a cross-session running log of what happened *today*, regardless of channel or conversation.

OpenClaw uses a similar pattern (`memory/YYYY-MM-DD.md`). Obsidian daily notes are already a well-established convention in the vault ecosystem.

### Design

**Daily file**: `YYYY-MM-DD.md` in the memory dir, one per day. Contains timestamped entries of notable events, decisions, and context from all interactions that day.

**Lifecycle**:
1. During conversations, notable events are appended to today's daily file (tool calls, decisions, user corrections, key topics)
2. At end of day (or via scheduler/manual trigger), the daily file is reviewed — anything worth keeping is promoted to semantic or procedural memory via `/review`
3. Raw daily files age out naturally — recent days provide immediate context, older ones sit in the vault as searchable history
4. The indexer picks them up for hybrid search like any other markdown file

**What gets logged** (append-only, lightweight):
- Tool calls and their outcomes (e.g. "added 'buy milk' to todoist")
- User corrections (`/w`) and rewards (`/r`) with context
- Topic summaries when a conversation compacts
- Channel annotations (which channel the interaction came from)

**What doesn't get logged**:
- Full conversation transcripts (that's what session logs are for)
- Raw model responses (too noisy)

### Relationship to existing memory

| Layer | Scope | Granularity | Persistence |
|-------|-------|-------------|-------------|
| Session logs | Single conversation | Full transcript | Archived |
| **Daily memory** | **All interactions in a day** | **Notable events** | **Rolling window** |
| Semantic (Memory.md) | Cross-session facts | Distilled | Permanent |
| Procedural (Procedural.md) | Behavioural rules | Distilled | Permanent |

Daily memory complements sessions — sessions capture *what was said*, daily memory captures *what happened*. The `/review` command already distills corrections into procedural rules; it could also distill daily memory entries into semantic memory.

### Files to modify

#### 1. `tars/memory.py` — daily file helpers

```python
def daily_memory_path() -> Path | None:
    """Return path to today's daily memory file (YYYY-MM-DD.md)."""
    md = _memory_dir()
    if not md:
        return None
    return md / f"{datetime.now().strftime('%Y-%m-%d')}.md"

def append_daily(entry: str) -> None:
    """Append a timestamped entry to today's daily memory file."""
    path = daily_memory_path()
    if not path:
        return
    timestamp = datetime.now().strftime("%H:%M")
    line = f"- {timestamp} {entry}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)

def load_daily(date: str | None = None) -> str:
    """Load a daily memory file. Defaults to today."""
    md = _memory_dir()
    if not md:
        return ""
    date = date or datetime.now().strftime("%Y-%m-%d")
    path = md / f"{date}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")
```

#### 2. `tars/conversation.py` — log notable events

After tool calls in `process_message()`, append a summary to daily memory:
```python
if tool_result:
    append_daily(f"[{channel}] tool:{tool_name} — {brief_summary}")
```

After compaction:
```python
append_daily(f"[{channel}] session compacted — {topic_summary}")
```

#### 3. `tars/cli.py` — log corrections/rewards with context

When `/w` or `/r` is used, also append to daily memory:
```python
append_daily(f"correction: {last_response[:80]}...")
append_daily(f"reward: {last_response[:80]}...")
```

#### 4. `tars/core.py` — include recent daily context in system prompt

Load today's daily memory (and optionally yesterday's) as part of system prompt assembly, giving the model awareness of what's already happened today:
```python
daily = load_daily()  # today
if daily:
    search_context += f"\n<daily-context>\n{daily}\n</daily-context>"
```

#### 5. Tests

- `test_daily_memory_path`: returns correct path format
- `test_append_daily`: creates file, appends timestamped entries
- `test_load_daily_today`: reads back appended entries
- `test_load_daily_missing`: returns empty string
- `test_load_daily_specific_date`: loads a named date

### Connects to

- **Scheduler (future)**: a scheduler could create/close daily files, run end-of-day review
- **`/review`**: could be extended to review daily memory entries alongside corrections/rewards
- **Indexer**: daily files are markdown in the memory dir — automatically indexed for hybrid search

### Verification

```
uv run python -m unittest discover -s tests -v
uv run tars
# Use some tools, make corrections, chat across topics
# Check YYYY-MM-DD.md in memory dir for entries
# Next session: verify daily context appears in system prompt
```

---

## Plan: MCP integration

### Context

The Model Context Protocol (MCP) is emerging as the standard way to expose tools to AI agents. Anthropic released the spec in November 2024, it moved to the Linux Foundation, and as of early 2026 most open-source assistants (OpenClaw, nanobot, PyGPT, Agent Zero) support it. MCP lets tars consume any MCP-compatible tool server without writing bespoke integration code per tool — Home Assistant, GitHub, filesystem, databases, and community-built servers all become available through a uniform protocol.

Currently tars has hand-rolled tools in `tools.py` (todoist, weather, memory, web). Each new tool requires: defining the schema, writing a handler, wiring dispatch, auditing internal `chat()` paths for tool leakage, and updating help across CLI/web/telegram. MCP doesn't replace the existing tools but provides a standard way to add new ones without this per-tool overhead.

### Design

**MCP as an additional tool source, not a replacement.** Built-in tools (todoist, weather, memory search) stay native — they're tightly integrated and don't benefit from the MCP abstraction. MCP provides the extension point for everything else.

**Client-side only.** Tars consumes MCP servers, it doesn't expose one. A single-user assistant doesn't need to be a server — it needs to talk to servers.

**Configuration-driven.** MCP servers are declared in config (env var or config file), discovered at startup, and their tools are merged into the tool list sent to the model. No code changes needed to add a new MCP server.

### Architecture

```
[core.py chat()] → tools = native_tools + mcp_tools
                          ↓
                   [tools.py run_tool()]
                          ↓
              native?  → todoist/weather/memory/web handler
              mcp?     → mcp_client.call_tool(server, name, args)
```

**Tool discovery**: At startup, connect to each configured MCP server, call `tools/list`, and merge the returned tool schemas into `ANTHROPIC_TOOLS` / `OLLAMA_TOOLS`. Prefix tool names with server name to avoid collisions (e.g. `github.create_issue`).

**Tool dispatch**: `run_tool()` checks if the tool name matches a native handler; if not, routes to the MCP client which forwards to the appropriate server.

**Lifecycle**: MCP servers can be local processes (stdio transport) or remote (SSE/HTTP). For personal use, stdio is the common case — tars spawns the server process and communicates over stdin/stdout.

### Files to modify

#### 1. `tars/mcp.py` (new) — MCP client

```python
class MCPClient:
    """Manages connections to MCP tool servers."""

    def __init__(self, server_configs: list[dict]):
        self.servers = {}  # name -> connection

    def discover_tools(self) -> list[dict]:
        """Connect to all servers, return merged tool schemas."""

    def call_tool(self, server: str, tool: str, args: dict) -> str:
        """Forward a tool call to the appropriate MCP server."""

    def close(self):
        """Shutdown all server connections."""
```

Use the `mcp` Python SDK (`pip install mcp`) for protocol handling. Supports stdio and SSE transports.

#### 2. Configuration

Environment variable with JSON or path to config file:

```
TARS_MCP_SERVERS='[{"name": "github", "command": "npx @modelcontextprotocol/server-github", "env": {"GITHUB_TOKEN": "..."}}, {"name": "filesystem", "command": "npx @modelcontextprotocol/server-filesystem /home/user/docs"}]'
```

Or a `mcp_servers.json` file in the memory dir, matching the pattern used by Claude Code and other MCP clients.

#### 3. `tars/tools.py` — merge MCP tools into tool list

```python
def get_all_tools(mcp_client: MCPClient | None = None) -> list[dict]:
    """Return native tools + discovered MCP tools."""
    tools = list(ANTHROPIC_TOOLS)
    if mcp_client:
        tools.extend(mcp_client.discover_tools())
    return tools
```

Modify `run_tool()` to check MCP client for unknown tool names before returning an error.

#### 4. `tars/core.py` — pass MCP tools to model

Initialize `MCPClient` at startup (in `_load_config` or similar), pass merged tool list to `chat_anthropic()` / `chat_ollama()`.

#### 5. `tars/cli.py` — lifecycle management

Initialize MCP client before REPL loop, close on exit. Add `/mcp` command to list connected servers and available tools.

#### 6. Tests

- `test_mcp_discover_tools`: mock MCP server, verify tool schemas are returned
- `test_mcp_call_tool`: mock server, verify tool call is forwarded and result returned
- `test_mcp_tool_merge`: verify native + MCP tools are combined without collision
- `test_mcp_dispatch`: verify `run_tool()` routes unknown tools to MCP client
- `test_mcp_no_servers`: verify graceful behavior when no MCP servers configured
- `test_mcp_server_failure`: verify error handling when a server is unreachable

### Dependencies

- `mcp` — official MCP Python SDK

### Risks and considerations

- **Prompt bloat**: each MCP server adds tool definitions to the prompt. The selective skill injection pattern (from OpenClaw) becomes important if many servers are configured — score tool relevance per turn rather than injecting all.
- **Security**: MCP servers run as local processes with the user's permissions. Untrusted servers are a risk. Config-only (no auto-discovery) mitigates this.
- **Startup latency**: connecting to multiple MCP servers adds startup time. Lazy connection (connect on first use) is an option.
- **use_tools=False**: the existing guard for internal `chat()` paths (summarization, review) automatically excludes MCP tools too, since they're merged into the same tool list.

### Verification

```
uv run python -m unittest discover -s tests -v
# Configure a simple MCP server (e.g. filesystem)
TARS_MCP_SERVERS='[{"name": "fs", "command": "npx @modelcontextprotocol/server-filesystem /tmp"}]' uv run tars
you> /mcp
# Should list "fs" server with its tools
you> list files in /tmp
# Should route to filesystem MCP server
```

---

## Plan: Scheduled task runner

### Context

Tars already has the pieces for scheduled work — `brief` generates a daily summary, email polling runs on an interval, and tools exist to do useful things (todoist, weather, memory, search). What's missing is a general-purpose way to say "run this at 8am every day" or "check todoist every hour." The email brief is a hardcoded instance of this pattern. A lightweight scheduler generalises it.

This also connects to daily memory files — the scheduler is the natural mechanism for end-of-day review, daily file rotation, and periodic memory maintenance.

### Design

**Cron-style, not event-driven.** Personal assistants need predictable recurring tasks, not complex event chains. A simple schedule definition (time + recurrence + action) covers the use cases.

**Actions are synthetic messages.** Rather than a new execution model, scheduled tasks inject a message into `process_message()` as if a user sent it. This reuses all existing tool routing, memory, and response handling. A scheduled brief is just `process_message(conv, "/brief")` triggered by the clock.

**Single-process, in-memory scheduler.** No external dependencies (no celery, no cron). A background thread checks the schedule every minute and fires due tasks. State is simple enough that restarts are fine — missed tasks just run on next check.

**Channel-aware delivery.** Scheduled task output needs to go somewhere. Options per task: log to daily memory (default), send via email, post to telegram, or print to CLI if active. Email brief already does this — the scheduler generalises the delivery.

### Schedule definition

Config file `schedules.json` in the memory dir (or env var):

```json
[
    {
        "name": "morning_brief",
        "schedule": "08:00",
        "recurrence": "daily",
        "action": "/brief",
        "deliver": "email"
    },
    {
        "name": "todoist_check",
        "schedule": "*/60",
        "recurrence": "interval_minutes",
        "action": "/todoist today",
        "deliver": "daily_memory"
    },
    {
        "name": "end_of_day_review",
        "schedule": "22:00",
        "recurrence": "daily",
        "action": "/review",
        "deliver": "daily_memory"
    }
]
```

### Architecture

```
[scheduler.py]
    ↓ background thread, checks every 60s
    ↓ task due?
    ↓
[conversation.py process_message()] ← synthetic message
    ↓
[core.py chat()] → tools → response
    ↓
[delivery] → daily_memory / email / telegram / cli
```

### Files to modify

#### 1. `tars/scheduler.py` (new) — scheduler engine

```python
@dataclass
class ScheduledTask:
    name: str
    schedule: str        # "HH:MM" or "*/N"
    recurrence: str      # "daily" | "interval_minutes"
    action: str          # slash command or message
    deliver: str         # "daily_memory" | "email" | "telegram" | "cli"
    last_run: datetime | None = None

class Scheduler:
    def __init__(self, tasks: list[ScheduledTask], config: dict):
        self.tasks = tasks
        self.config = config
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the scheduler background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            now = datetime.now()
            for task in self.tasks:
                if self._is_due(task, now):
                    self._execute(task)
                    task.last_run = now
            time.sleep(60)

    def _is_due(self, task: ScheduledTask, now: datetime) -> bool:
        """Check if a task should run now."""

    def _execute(self, task: ScheduledTask):
        """Run the task action and deliver the result."""

    def _deliver(self, task: ScheduledTask, result: str):
        """Route output to the configured delivery channel."""
```

#### 2. `tars/scheduler.py` — config loading

```python
def load_schedules() -> list[ScheduledTask]:
    """Load schedule definitions from memory dir or env var."""
    md = _memory_dir()
    if md:
        path = md / "schedules.json"
        if path.exists():
            return _parse_schedules(json.loads(path.read_text()))
    env = os.environ.get("TARS_SCHEDULES")
    if env:
        return _parse_schedules(json.loads(env))
    return []
```

#### 3. `tars/cli.py` — start scheduler in REPL and serve modes

```python
scheduler = Scheduler(load_schedules(), config)
scheduler.start()
# ... repl loop ...
scheduler.stop()
```

Add `/schedule` command to list scheduled tasks and their last run times.

#### 4. `tars/api.py` — start scheduler with server

Same lifecycle — start on server boot, stop on shutdown.

#### 5. `tars/email.py` — refactor email brief as a scheduled task

The existing email polling + brief logic becomes a configuration of the scheduler rather than a hardcoded loop. The `email` subcommand still works but delegates to the scheduler internally.

#### 6. Tests

- `test_is_due_daily`: task at "08:00" fires at 08:00, not at 07:59 or 08:02
- `test_is_due_interval`: task at "*/60" fires 60 minutes after last run
- `test_is_due_no_last_run`: first run fires immediately for intervals, at scheduled time for daily
- `test_execute_calls_process_message`: verify action is routed through process_message
- `test_deliver_daily_memory`: verify output is appended to daily memory file
- `test_load_schedules_from_file`: verify JSON parsing
- `test_load_schedules_empty`: no config returns empty list
- `test_scheduler_start_stop`: verify thread lifecycle

### Connects to

- **Daily memory files**: scheduler writes task outputs to daily memory by default, handles end-of-day review
- **Email brief**: becomes a scheduled task configuration rather than hardcoded logic
- **`/review`**: can be scheduled for end-of-day automatic distillation (with auto-apply or log-only mode for unattended runs)

### Risks and considerations

- **Unattended `/review`**: the current `/review` requires interactive approval. Scheduled review needs an auto-apply mode or a "propose and log" mode that writes suggestions to daily memory for later human review.
- **Overlapping runs**: if a task takes longer than the check interval, the next check could fire it again. Guard with `last_run` check + a running flag per task.
- **Process model**: scheduler only runs while tars is running (REPL, serve, or email mode). No persistent daemon. This is fine for personal use — the server or telegram bot is typically always-on.

### Verification

```
uv run python -m unittest discover -s tests -v
# Create schedules.json with a test task (interval 1 minute)
uv run tars
# Wait 1 minute, verify task fires and output appears
/schedule
# Should list tasks with last run times
```

---

## Plan: Centralized slash command dispatch

### Context

Slash command dispatch is duplicated across `cli.py`, `email.py`, and `telegram.py`. This was flagged in CLAUDE.md from day one. A shared `commands.py` already exists and handles a significant subset (todoist, weather, forecast, memory, remember, note, read, capture, brief, search, find, sessions, export). But several commands are still only handled in `cli.py` with no equivalent in the other channels:

- `/review` — CLI only (interactive approval), but not intercepted in telegram/email, so it falls through to chat
- `/tidy` — CLI only (interactive), same fall-through problem
- `/session <query>` — CLI only
- `/sgrep`, `/svec` — CLI only (search mode variants)
- `/stats` — CLI only
- `/schedule` — CLI only
- `/model` — CLI only
- `/clear` — CLI only (REPL state)
- `/w`, `/r` — CLI only (feedback on last response)
- `/help` — CLI renders help text; web has its own; telegram has none

The concrete problem: adding a new command means touching three files, and forgetting one leads to silent fall-through where the model receives `/review` as a user message and does something unpredictable (the tool leakage bug class).

### Design

**Extend `commands.py` to be the single registry of all commands.** Each command declares:
- Name and aliases
- Handler function
- Channel support (`all`, `interactive_only`, `cli_only`)
- Usage string

**Channel files become thin dispatchers:**
1. Call `commands.dispatch(text, ...)`
2. If it returns a result → display it
3. If it returns a "not supported on this channel" sentinel → show a clean message
4. If it returns `None` → not a command, pass through to chat

**Commands that need channel-specific state** (like `/w` which needs the last model response, or `/clear` which resets REPL state) use a callback/context pattern — the channel passes a context dict that the command can read from.

### Files to modify

#### 1. `tars/commands.py` — command registry

```python
@dataclass
class Command:
    name: str
    handler: Callable
    channels: str = "all"        # "all" | "interactive" | "cli_only"
    usage: str = ""
    aliases: tuple[str, ...] = ()

_COMMANDS: dict[str, Command] = {}

def register(name, *, channels="all", usage="", aliases=()):
    """Decorator to register a slash command."""

def dispatch(text, *, provider="", model="", conv=None, context=None) -> str | None:
    """Dispatch a slash command. Returns result, or None if not a command.

    If the command exists but isn't supported on the current channel,
    returns a message like "'/review' is only available in the CLI."
    """
```

The `context` dict carries channel-specific state:
```python
context = {
    "channel": "cli" | "telegram" | "email" | "web",
    "last_response": str,         # for /w, /r
    "conversation": Conversation,  # for /export, /clear
    "config": dict,               # for /model
}
```

#### 2. Move CLI-only commands into `commands.py` with channel guards

Commands like `/review`, `/tidy`, `/stats`, `/schedule`, `/model` get registered with `channels="interactive"` or `channels="cli_only"`. Their handler functions move from `cli.py` into `commands.py` (or stay in their own modules and get registered).

#### 3. `tars/cli.py` — simplify REPL loop

The REPL loop's long chain of `if user_input.strip() == "/review":` blocks collapses to:
```python
result = dispatch(text, provider=p, model=m, conv=conv, context=ctx)
if result is not None:
    print(result)
    continue
# Not a command — pass to chat
```

Special cases (`/help` rendering with ANSI, `/clear` resetting REPL state) either use the context dict or remain as thin wrappers around the registered handler.

#### 4. `tars/telegram.py` — replace ad-hoc dispatch

Currently has its own partial command handling. Replace with:
```python
result = dispatch(text, provider=p, model=m, conv=conv,
                  context={"channel": "telegram"})
if result is not None:
    await update.message.reply_text(result)
    return
```

#### 5. `tars/email.py` — same pattern

#### 6. Tests

- `test_dispatch_recognized_command`: returns result string
- `test_dispatch_unknown_command`: returns None
- `test_dispatch_channel_guard`: command exists but wrong channel returns "CLI only" message
- `test_all_commands_registered`: every entry in `_SLASH_COMMANDS` has a registered handler
- `test_help_lists_all_commands`: `/help` output includes every registered command

### Benefits

- **One place to add a command** — define it once, it works everywhere it should
- **Explicit channel guards** — no more silent fall-through to chat
- **`/help` auto-generated** from the registry — always complete, never stale
- **Test coverage becomes meaningful** — one dispatcher, one test suite
- **Tab completion derived from registry** — `_SLASH_COMMANDS` list can be generated from `_COMMANDS.keys()`

### Verification

```
uv run python -m unittest discover -s tests -v
uv run tars
# /help should list all commands
# /review should work in CLI
# Telegram: /review should return "CLI only" not fall through to chat
```

---

## Plan: Post-conversation memory extraction

### Context

Tars has manual memory capture (`/remember semantic|procedural <text>`) and a feedback loop (`/w`, `/r`, `/review`). But users have to explicitly decide to save something. Ted's demand-side research found the #1 conversion moment is "the assistant remembers something unprompted from a previous session." The #1 abandonment reason is having to re-explain context.

Post-conversation extraction bridges this gap: after a conversation ends (or compacts), the model extracts discrete facts and saves them automatically. This connects to the daily memory plan — extracted facts write to the daily file, and `/review` can promote important ones to semantic or procedural memory.

### Design

**Extract on compaction and session save.** These are the two natural points where a conversation has accumulated enough content to be worth extracting from. Extraction is a single model call with `use_tools=False` — no risk of tool leakage.

**Write to daily memory, not directly to Memory.md.** This is important: automatic extraction is lossy and sometimes wrong. Writing to the daily file means facts are available immediately for context but require human review (or `/review`) before promotion to permanent memory. This preserves the user's control over what's in their durable memory.

**Extraction prompt asks for discrete facts, not summaries.** Summaries are what session compaction already produces. Extraction targets: user preferences, stated facts about themselves, project context, tool usage patterns, corrections.

### Architecture

```
[conversation ends / compacts]
    ↓
[_extract_facts(messages)] → model call, use_tools=False
    ↓
[list of fact strings]
    ↓
[append_daily(fact) for each fact]
    ↓
[available in daily context next session]
    ↓
[/review can promote to Memory.md / Procedural.md]
```

### Files to modify

#### 1. `tars/extractor.py` (new) — fact extraction

```python
_EXTRACTION_PROMPT = """\
Extract discrete facts from this conversation that would be useful to remember \
for future interactions. Focus on:
- User preferences and stated facts about themselves
- Project names, tools, and technologies mentioned
- Corrections or clarifications the user made
- Decisions that were reached
- Important context that would save the user from re-explaining

Return ONLY a JSON array of short fact strings. Each fact should be a single \
sentence. If nothing worth extracting, return an empty array [].

Do not include:
- Conversation pleasantries
- Facts already obvious from the system prompt
- Speculative or uncertain information
"""

def extract_facts(messages: list[dict], provider: str, model: str) -> list[str]:
    """Extract discrete memorable facts from a conversation."""
    # Filter to user and assistant messages, skip system
    # Truncate if too long (last N messages)
    # Single model call with use_tools=False
    # Parse JSON array from response
    # Return list of fact strings, empty list on failure
```

#### 2. `tars/conversation.py` — call extraction on compaction and save

After `_maybe_compact()` produces a summary:
```python
facts = extract_facts(compacted_messages, provider, model)
for fact in facts:
    append_daily(f"[extracted] {fact}")
```

After `save_session()`:
```python
facts = extract_facts(conv.messages, provider, model)
for fact in facts:
    append_daily(f"[extracted] {fact}")
```

Guard with try/except — extraction failure should never break the conversation flow.

#### 3. `tars/memory.py` — `append_daily()` (from daily memory plan)

Already designed — this plan depends on daily memory being implemented first.

#### 4. Tests

- `test_extract_facts_returns_list`: mock model response with JSON array, verify parsed
- `test_extract_facts_empty_conversation`: returns empty list
- `test_extract_facts_invalid_json`: returns empty list (graceful failure)
- `test_extract_facts_model_error`: returns empty list
- `test_extraction_on_compaction`: verify facts appended to daily memory after compaction
- `test_extraction_on_save`: verify facts appended to daily memory on session save
- `test_extracted_facts_tagged`: each daily entry starts with `[extracted]` prefix

### Depends on

- **Daily memory files** — extraction writes to the daily file

### Connects to

- **`/review`** — can review extracted facts alongside corrections/rewards and promote to permanent memory
- **Scheduler** — could run a nightly extraction pass over sessions that didn't get extracted (e.g. if tars crashed)
- **Search** — extracted facts in daily files get indexed and are searchable

### Risks and considerations

- **Cost**: one extra model call per compaction/save. For personal use this is ~2-5 calls per day — negligible.
- **Quality**: extraction will sometimes be wrong or noisy. The daily memory buffer absorbs this — bad extractions age out, good ones get promoted.
- **Privacy**: extraction happens through the same model path as conversation — no new data exposure. But users should know it's happening. Add a config flag `TARS_AUTO_EXTRACT=true|false`.
- **Duplicate facts**: the same fact may be extracted from multiple conversations. Deduplication can be a future refinement (semantic similarity check before appending), but isn't critical for a personal vault.

### Verification

```
uv run python -m unittest discover -s tests -v
uv run tars
# Have a conversation mentioning your project name, a preference, a correction
# End session (Ctrl-D)
# Check YYYY-MM-DD.md in memory dir for [extracted] entries
# Start new session — verify daily context includes the extracted facts
```
