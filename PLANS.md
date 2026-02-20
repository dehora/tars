# Implementation Plans

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
