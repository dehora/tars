# tars

A personal AI assistant with CLI, web, email, and Telegram channels. Routes messages to configurable AI providers (ollama, claude), manages persistent memory in an Obsidian vault, and integrates with Todoist, weather APIs, and Obsidian daily notes. Uses uv for package management with Python 3.14.

## What it does

tars is a conversational assistant that remembers things across sessions, manages your tasks, checks the weather, captures web pages, and writes to your daily notes. It stores memory in Obsidian markdown files so everything stays readable and editable outside of tars.

**Channels:**
- **CLI** — REPL with streaming, tab completion, and slash commands
- **Web UI** — browser-based chat with SSE streaming
- **Email** — IMAP polling + SMTP reply via Gmail, with slash commands in subject or body
- **Telegram** — bot polling with persistent reply keyboard and slash commands

**Tools:**
- **Todoist** — add, list, complete tasks via natural language
- **Weather** — current conditions and hourly forecasts (Open-Meteo)
- **Memory** — persistent facts and preferences in an Obsidian vault
- **Daily notes** — append thoughts to today's Obsidian journal entry
- **Search** — hybrid keyword + semantic search across all memory types
- **Web read** — fetch and extract text content from web pages for discussion
- **Capture** — save web pages to your Obsidian vault with AI summarization (context-aware when mid-conversation)

**Memory types:**
- **Semantic** (`Memory.md`) — facts, preferences, things you've told tars to remember
- **Procedural** (`Procedural.md`) — behavioural rules learned from feedback
- **Episodic** (`sessions/`) — session logs of past conversations

**Multi-model routing:**

When `TARS_REMOTE_MODEL` (or legacy `TARS_ESCALATION_MODEL`) is set, tars uses the primary model (from `TARS_DEFAULT_MODEL` or legacy `TARS_MODEL`) for chat and automatically escalates to the remote model when tool use is detected. If the remote model is unavailable (rate limit, outage, transient errors), it falls back to the primary model.

**Feedback loop:**
- `/w` flags a bad response, `/r` flags a good one
- `/review` distills corrections into procedural rules
- `/tidy` cleans up stale or duplicate memory entries
- Procedural rules feed back into every future response

## How it works

```
[CLI / Web / Email / Telegram] → [conversation.py] → [core.py] → ollama / claude
                                               ↕
                                         [tools.py] → todoist, weather, memory, notes, search, web
                                               ↕
                              [memory.py] ← obsidian vault (TARS_MEMORY_DIR)
                              [notes.py]  ← obsidian vault (TARS_NOTES_DIR)
                              [search.py] ← sqlite-vec + FTS5 (tars.db)
                              [router.py] → multi-model routing + fallback
```

- **Providers**: Claude (Anthropic API) or ollama (local models). Set via `TARS_DEFAULT_MODEL` (or legacy `TARS_MODEL`).
- **Routing**: keyword-based pre-routing detects tool intent and escalates to a remote model when configured. Falls back on transient API errors.
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

# Send daily brief via email
tars email-brief

# Telegram bot
tars telegram
# or: tars-telegram

# Send daily brief via Telegram
tars telegram-brief

# Schedule daily email brief
tars schedule add email-brief email-brief --hour 8 --minute 0

# Schedule vault reindex on file change
tars schedule add notes-reindex notes-index --watch "$TARS_NOTES_DIR"

# List installed schedules
tars schedule list

# Test-run a schedule (uses baked OS environment)
tars schedule test email-brief

# Remove a schedule
tars schedule remove email-brief

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
| `/stats` | Memory and index health |
| `/model` | Show active model configuration |
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

### Telegram commands

Slash commands work in the bot chat. A persistent reply keyboard provides one-tap access to common commands.

| Command | Description |
|---------|-------------|
| `/todoist add <text>` | Add a task |
| `/todoist today` | List today's tasks |
| `/weather` | Current conditions |
| `/forecast` | Today's hourly forecast |
| `/memory` | Show persistent memory |
| `/remember <section> <text>` | Save to memory |
| `/note <text>` | Append to daily note |
| `/capture <url> [--raw]` | Capture web page to vault |
| `/brief` | Daily briefing digest |
| `/search <query>` | Hybrid search over memory |
| `/find <query>` | Search personal notes vault |
| `/sessions` | List recent sessions |
| `/clear` | Reset conversation |

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `TARS_MODEL` | `claude:sonnet` | Legacy primary model (`provider:model`) |
| `TARS_DEFAULT_MODEL` | `claude:sonnet` | Primary model (`provider:model`) |
| `TARS_ESCALATION_MODEL` | — | Legacy remote model for tool calls (`provider:model`) |
| `TARS_REMOTE_MODEL` | — | Remote model for tool calls (`provider:model`, explicit versions recommended; set to `none` to disable) |
| `TARS_ROUTING_POLICY` | `tool` | Routing policy (only `tool` supported) |
| `TARS_MEMORY_DIR` | — | Path to tars Obsidian vault |
| `TARS_NOTES_DIR` | — | Path to personal Obsidian vault (daily notes, captures) |
| `TARS_MAX_TOKENS` | `1024` | Max tokens for Anthropic responses |
| `ANTHROPIC_API_KEY` | — | Required for Claude provider |
| `TARS_EMAIL_ADDRESS` | — | Gmail address for email channel |
| `TARS_EMAIL_PASSWORD` | — | Gmail app password |
| `TARS_EMAIL_ALLOW` | — | Comma-separated allowed sender addresses |
| `TARS_EMAIL_TO` | — | Recipient address for email brief |
| `TARS_EMAIL_POLL_INTERVAL` | `60` | Seconds between inbox checks |
| `TARS_TELEGRAM_TOKEN` | — | Telegram bot API token from BotFather |
| `TARS_TELEGRAM_ALLOW` | — | Comma-separated Telegram user IDs |
| `TARS_API_TOKEN` | — | Optional bearer token for API auth |
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
