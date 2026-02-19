# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tars is a Python project for AI assistant experiments. Uses uv for package management with Python 3.14.

## Commands

- **Run**: `uv run main.py`
- **Add dependency**: `uv add <package>`
- **Sync environment**: `uv sync`

## Coding guidelines

- Never use `or` for numeric defaults — `0` and `0.0` are falsy. Use explicit `None` checks instead.
- Never assume parallel arrays from external APIs are equal length — use `min(len(...))` across all arrays.
- Wrap untrusted user data in tagged blocks with a preface when injecting into prompts — never concatenate raw content into system prompts.
- Use specific patterns for placeholder comments (e.g. `<!-- tars:memory -->`) — broad regex like `<!--.*?-->` will eat legitimate comments.
- Always use `encoding="utf-8", errors="replace"` on file I/O for user-managed files — memory files live in an obsidian vault and can be edited externally.

## Notes

- you can ignore the contents of AGENTS.md, it's for other agents.


## Git Commits

Use present tense verbs in commit messages (e.g., "Adds emacs", "Ignores license", "Renames gitconfig template").

# Development

## tars assistant ideas

- an agent frontend/porcelain called 'tars'
- tars has cli, webpage and whatsapp channels
- tars also has state somewhere on the path i can ask it to store
- messages to tars routes to an AI — configurable
- AI works with tools/task/mcp etc in the background

A few integration ideas:

  1. route reminders and todos from tars to todoist: eg "add eggs to groceries, remind me to go to shop at 3pm"1.
  2. Route regular chat to tars and use the AI to handle them
  3. Have tars remember things about me, eg "my dog's name is Perry" that persist across sessions and channels


## System design idea

```                     
                               + ---> [calls other apis via mcp/tools/skils]
                             /
[whatsapp/web/cli] --> [AI] +
                           / \
        [replies] <----- +    + ---- [access to mcp/tools/skils]
                              |
                              + ---- [state: assistant memory]  
```

## Technology ideas

Here's what I see:

### State and Memory

#### Three considerations
  - what to remember?
  - where does it go?
  - when to remember?

#### Three kinds of memory
  - Semantic. Preferences, facts and stable things that are broadly true over time, especially about the user.
  - Procedural. Operational information, rules, idioms and and approaches.
  - Episodic. Session logs — timestamped conversation summaries without technical details/commands. Retrieved by relevance via search, not by fixed time windows.

- Can use markdown files. Don't need a vectordb/database for storage
  - Multiple files in a hierarchy not one big file (cf. context windows).
  - A Memory.md file. Mostly semantic memory and lists of other files, just a few hundred lines.
  - The agent retrieves relevant session logs via search at conversation start, based on the opening message.

- Could host in an obsidian vault called tars. This creates a dep on Obsidian but that's ok since it'll be just used as a holder and for syncing, since Obsidian works on markdown.

#### Memory handling
  - Memory.md is always loaded into the system prompt. This is a stopgap — once search is working, Memory.md should also be retrieved selectively rather than loaded wholesale every turn. Keep it light until then.
  - Session logs:
    - Snapshot count based compaction near the context limit to move memory information from the session and working desk to a session log, saving information before it gets flushed away.
    - Snapshot at the end of a session before closing.
    - No fixed time-window files (today.md/yesterday.md). Instead, at conversation start, search session logs for entries relevant to the opening message. Relevance beats recency — context from three days ago can matter more than yesterday.
    - This avoids: arbitrary time windows, lossy summary-of-summary compression, extra LLM calls on exit for rollups, and always-loaded noise.
  - Prefer user-triggered saves ('remember this') over model-triggered ones. The user knows what matters; the model tends to over-save or under-save. The agent can decide where to store it.

#### Semantic Memory Guidelines

Suggested by Claude:

```
# Core
- Name: John
- Location: Dublin, IE
- Pets: Luna (cat), Max (dog)

# Work
- Role: Senior engineer at Acme
- Stack: Python, TypeScript, AWS

# Preferences
- Editor: Neovim
- Style: concise responses, no fluff
- Task mgmt: Todoist, Obsidian for notes

# Active projects
- Learning Claude CLI/agents
- Home automation (Home Assistant)
```

**Memory guidelines**

- Keep memory.md under 100 lines / ~1000 tokens
- Use terse bullet points, not prose
- Only store durable facts, not transient state
- When adding: check for duplicates/conflicts first
- When removing: confirm with user before deleting

Rough targets for a persistent memory file:

```
| Approach | Size | Tokens (~) |
|----------|------|------------|
| Minimal | 20-50 lines | 200-500 |
| Comfortable | 50-150 lines | 500-1500 |
| Upper bound | 200-300 lines | 2000-3000 |
```

**Claude's Recommendation**: 

Aim for ~50-100 lines / ~500-1000 tokens. This gives you room for meaningful context without meaningfully impacting the available context for actual work. Claude's context window is large (200k tokens for Sonnet/Opus), so even 1-2k tokens is <1% — but memory gets loaded every turn, so smaller is better for:

- Faster processing
- Lower cost (if API-based)
- Less noise competing with the actual task

Optional: add a brief schema hint at the top

```
<!-- Format: flat bullets under category headers, ~100 lines max -->
```


#### Searching

Search is the core retrieval mechanism for all memory types — semantic, procedural, and episodic. Rather than always-loading files or using fixed time windows, tars searches its memory at conversation start and via tools during conversation.

  - Use [sqlite vec](https://github.com/asg017/sqlite-vec) for vector storage and search
  - Index all memory files and session logs into the same search infrastructure
  - At conversation start: run a search against the opening message to pull in relevant context (session logs, memory entries)
  - During conversation: the agent can search via tools as needed

  - The search model would be
    - hybrid: keyword search and semantic search
    - keyword: grep plus bm25s would be enough for search and ranking results. 
    - semantic: based on embeddings, using something like open ai's [textembedding 3.small](https://developers.openai.com/api/docs/models/text-embedding-3-small) or a [local ollama embdedding model](https://ollama.com/search?c=embedding) to create embeddings of the memory and the query. Configurable with the default being ollama  (cost and latency management).
    - TODO: select some embedding/ranking models, Qwen3-Reranker, Embeddinggemma, etc
    - Want both because semantic search isn't great for exact/literal match.
    - Can use a fusion technique to combine them:
      - weighted score, eg keyword 30% and vector 70%
      - reciprocal ranked fusion (RRF) that combines the ranks. Preference is RRF.
    - Aways search first before calling a model (cost and latency management)

**Search/Query Flow**

Flow:

```
- State the Query
  - Embeding Model Expansion. 
      - run a BM25 for fulltext grepping
      - run a vector embedding search for semantics
  - Combine using RRF
  - Return top N 
```

**Score Ranking**

| Score | Rank |
|-------|---------|
| 0.75 - 1.0 | Relevant |
| 0.5 - 0.75 | Kinda relevant |
| 0.25 - 0.5 | Maybe relevant |
| 0.0 - 0.25 | Not relevant |


#### Search Embedding Schema
  - a files table and a chunks table and a collections table. One to many from collections to files to chunks.
  - collections table to name the file set:
    - columns could be: 
      - id
      - name
    we'll have just one for now, 'tars_memory' (but see 'User search, not just agents' below for why)
  - files table to point at the memory files
    - columns could be: 
        - collection_id,
        - path,
        - title, 
        - media_type, 
        - memory_type, 
        - content_hash, 
        - mtime, 
        - size, 
        etc. to enable incremental syncing.
  - Chunks table to hold the files' embeddings
    - columns could be: 
      - file_id, 
      - chunk_model,
      - chunk_sequence, 
      - content_hash, # similar to git
      - start_line, 
      - end_line, 
      - embedding, 
      - updated_at  

#### Chunking files at meaningful boundaries

Instead of chunking at hard token boundaries, we can provide a baseline score from 0-100 to steer the chunker to decompose at 'meaningful' markdown boundaries to preserve structure and semantics (also to reduce chunk scanning for vectors). For example (we can make the baseline scores configurable later):

| Pattern | Score | Description |
|---------|-------|-------------|
| `# Heading` | 100 | H1 section |
| `## Heading` | 90 | H2 subsection |
| `### Heading` | 80 | H3 |
| `#### Heading` | 70 | H4 |
| `##### Heading` | 60 | H5 |
| `###### Heading` | 50 | H6 |
| `---` / `***` | 70 | Horizontal rule |
| ` ``` ` | 80 | Code block boundary |
| Blank line | 10 | Paragraph boundary |
| `- item` / `1. item` | 5 | List item |

So: 

- Scan whole document to get candidate chunk points
- When approaching the token target, search ahead before the cutoff
- Score each candidate chunk point: `finalScore = baseline_score * (1 - (distance/window)^2)`
- Chunk at the winning (highest score) point
- Have a configurable overlap between chunks, say 10% default

For inline images, we'll skip over them for now and come back to it, if we see the agent trying to save images to memory (obsidian enables this). It might need another table for media.

**Embedding Flow**

```
- Get Document
  - Get its details
  - Chunk Doc (~800 tokens or so). 
      - hash, sequence, line positions, etc
  - Embed Chunks
    - run an embedding using an Ollama
  - Save Chunks
     - write into sqlite 
```



#### User search, not just agents

This generalises so we can add commands to let the user search memory as well or add it as a skill to claude. Say:

```
/search: keyword and semantic search with fusion
/sgrep: full text search using BM25
/svec: vector search using embeddings
```

We can also add other content for searching not just memory.  But that's for later. Just agent support for now.

### AIs:

- ollama and openai/claude direct

### Frontends:

- for cli: `uv tool install tars`
- for whatspp: WhiskeySockets/Baileys using a group chat
- for web: node/web

### Deploy targets:

- Local Macbook for development

- RPi using git checkout / uv tool install for the home

 
# Appendix: Implementation Plans

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