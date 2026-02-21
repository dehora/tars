# tars

A personal AI assistant with CLI, web, and email channels. Routes messages to configurable AI providers (ollama, claude), manages persistent memory in an Obsidian vault, and integrates with Todoist, weather APIs, and Obsidian daily notes. Uses uv for package management with Python 3.14.

## What it does

tars is a conversational assistant that remembers things across sessions, manages your tasks, checks the weather, captures web pages, and writes to your daily notes. It stores memory in Obsidian markdown files so everything stays readable and editable outside of tars.

**Channels:**
- **CLI** — REPL with streaming, tab completion, and slash commands
- **Web UI** — browser-based chat with SSE streaming
- **Email** — IMAP polling + SMTP reply via Gmail, with slash commands in subject or body

**Tools:**
- **Todoist** — add, list, complete tasks via natural language
- **Weather** — current conditions and hourly forecasts (Open-Meteo)
- **Memory** — persistent facts and preferences in an Obsidian vault
- **Daily notes** — append thoughts to today's Obsidian journal entry
- **Search** — hybrid keyword + semantic search across all memory types
- **Web read** — fetch and extract text content from web pages for discussion
- **Capture** — save web pages to your Obsidian vault with optional AI summarization

**Memory types:**
- **Semantic** (`Memory.md`) — facts, preferences, things you've told tars to remember
- **Procedural** (`Procedural.md`) — behavioural rules learned from feedback
- **Episodic** (`sessions/`) — session logs of past conversations

**Multi-model routing:**

When `TARS_ESCALATION_MODEL` is set, tars uses the primary model (e.g. a local ollama model) for chat and automatically escalates to the stronger model (e.g. Claude) when tool use is detected. If the escalation model is unavailable (rate limit, billing, outage), it falls back gracefully to the primary model.

**Feedback loop:**
- `/w` flags a bad response, `/r` flags a good one
- `/review` distills corrections into procedural rules
- `/tidy` cleans up stale or duplicate memory entries
- Procedural rules feed back into every future response

## How it works

```
[CLI / Web / Email] → [conversation.py] → [core.py] → ollama / claude
                                               ↕
                                         [tools.py] → todoist, weather, memory, notes, search, web
                                               ↕
                              [memory.py] ← obsidian vault (TARS_MEMORY_DIR)
                              [notes.py]  ← obsidian vault (TARS_NOTES_DIR)
                              [search.py] ← sqlite-vec + FTS5 (tars.db)
                              [router.py] → multi-model routing + fallback
```

- **Providers**: Claude (Anthropic API) or ollama (local models). Set via `TARS_MODEL`.
- **Routing**: keyword-based pre-routing detects tool intent and escalates to a stronger model when needed. Falls back on API errors.
- **Search**: markdown-aware chunking → ollama embeddings → sqlite-vec for KNN, FTS5 for keyword, fused with Reciprocal Rank Fusion.
- **Indexing**: incremental via content hash — only re-indexes changed files.
- **Streaming**: CLI and web UI stream responses token-by-token.
- **Sessions**: conversations are summarised and logged to the vault. Compaction keeps context manageable during long sessions.

## Usage

```bash
# REPL
tars

# Single message
tars "what's the weather like?"

# Web UI
tars serve
# or: tars-serve

# Email channel
tars email
# or: tars-email

# Rebuild search index
tars index
# or: tars-index
```

### CLI commands

| Command | Description |
|---------|-------------|
| `/todoist add <text> [--due D] [--project P] [--priority N]` | Add a task |
| `/todoist today` | List today's tasks |
| `/todoist upcoming [days]` | List upcoming tasks |
| `/todoist complete <ref>` | Complete a task |
| `/weather` | Current conditions |
| `/forecast` | Today's hourly forecast |
| `/memory` | Show persistent memory |
| `/remember <semantic\|procedural> <text>` | Save to memory |
| `/note <text>` | Append to today's daily note |
| `/capture <url> [--raw]` | Capture web page to vault |
| `/search <query>` | Hybrid keyword + semantic search |
| `/sgrep <query>` | Keyword search (FTS5/BM25) |
| `/svec <query>` | Semantic search (vector KNN) |
| `/sessions` | List recent sessions |
| `/session <query>` | Search session logs |
| `/brief` | Daily digest (tasks + weather) |
| `/w [note]` | Flag last response as wrong |
| `/r [note]` | Flag last response as good |
| `/review` | Review corrections and apply learnings |
| `/tidy` | Clean up memory (duplicates, junk) |

### Email commands

Slash commands work in the email subject line or body:

| Command | Description |
|---------|-------------|
| `/todoist add <text>` | Add a task |
| `/todoist today` | List today's tasks |
| `/weather` | Current conditions |
| `/forecast` | Today's hourly forecast |
| `/memory` | Show persistent memory |
| `/remember <section> <text>` | Save to memory |
| `/note <text>` | Append to daily note |
| `/read <url>` | Fetch and return page content |
| `/capture <url> [--raw]` | Capture web page to vault |

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `TARS_MODEL` | `claude:sonnet` | Provider and model (`provider:model`) |
| `TARS_ESCALATION_MODEL` | — | Escalation model for tool calls (`provider:model`) |
| `TARS_MEMORY_DIR` | — | Path to tars Obsidian vault |
| `TARS_NOTES_DIR` | — | Path to personal Obsidian vault (daily notes, captures) |
| `TARS_MAX_TOKENS` | `1024` | Max tokens for Anthropic responses |
| `ANTHROPIC_API_KEY` | — | Required for Claude provider |
| `TARS_EMAIL_ADDRESS` | — | Gmail address for email channel |
| `TARS_EMAIL_PASSWORD` | — | Gmail app password |
| `TARS_EMAIL_ALLOW` | — | Comma-separated allowed sender addresses |
| `TARS_EMAIL_POLL_INTERVAL` | `60` | Seconds between inbox checks |
| `DEFAULT_LAT` / `DEFAULT_LON` | — | Weather location coordinates |

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo> && cd tars
uv sync
cp .env.example .env  # add your API keys and vault paths
tars
```

## Tests

```bash
uv run python -m unittest discover -s tests -v
```
