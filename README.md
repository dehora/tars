# tars

A personal AI assistant with CLI, web, email, and Telegram channels. Routes messages to configurable AI providers (ollama, claude), manages persistent memory in an Obsidian vault, and integrates with Todoist, weather, Strava, and Obsidian daily notes. Uses uv for package management with Python 3.14.

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
- **Search** — hybrid keyword + semantic search across all memory types, with windowed context retrieval
- **Strava** — activities, summaries, training analysis, HR zones, routes, and gear via Strava API
- **Web read** — fetch and extract text content from web pages for discussion
- **Capture** — save web pages to your Obsidian vault with AI summarization (context-aware when mid-conversation)
- **MCP** — extend with external tool servers via the Model Context Protocol (fetch, GitHub, filesystem, etc.)

**Memory:** Four-tier system (semantic, procedural, daily, episodic) stored as plain markdown in the Obsidian vault. Automatic fact extraction promotes good entries to permanent memory via `/review`. See [docs/memory.md](docs/memory.md).

**Multi-model routing:** When `TARS_MODEL_REMOTE` is set, tars escalates tool-intent messages to the remote model and falls back on transient errors. See [docs/routing.md](docs/routing.md).

**Search:** Hybrid FTS5 keyword + sqlite-vec KNN, fused with Reciprocal Rank Fusion. Query rewriting and HyDE for expanded retrieval. Two-pass context packing for automatic search on first message. See [docs/search.md](docs/search.md).

**Scheduling:** In-process scheduler for recurring tasks within long-lived processes, plus OS-level scheduling via launchd/systemd. See [docs/scheduling.md](docs/scheduling.md).

**MCP:** Extend with external tool servers via the [Model Context Protocol](https://modelcontextprotocol.io/). Configured, not coded. See [docs/mcp.md](docs/mcp.md).

**Feedback loop:** `/w` flags bad responses, `/r` flags good ones, `/review` distills corrections into procedural rules, `/tidy` cleans up stale entries. See [docs/memory.md](docs/memory.md#feedback-loop).

## How it works

```
[CLI / Web / Email / Telegram] → [conversation.py] → [core.py] → ollama / claude
                                               ↕
                                         [tools.py] → todoist, weather, memory, notes, search, web, strava
                                               ↕                                    ↕
                              [memory.py] ← obsidian vault (TARS_MEMORY_DIR)   [mcp.py] → external MCP servers
                              [notes.py]  ← obsidian vault (TARS_NOTES_DIR)
                              [search.py] ← sqlite-vec + FTS5 (tars.db)
                              [router.py] → multi-model routing + fallback
```

- **Providers**: Claude (Anthropic API) or ollama (local models). Set via `TARS_MODEL_DEFAULT`.
- **Routing**: keyword pre-routing escalates to remote model on tool intent, falls back on errors. See [docs/routing.md](docs/routing.md).
- **Search**: hybrid FTS5 + sqlite-vec KNN with RRF fusion, query rewriting, and two-pass context packing. See [docs/search.md](docs/search.md).
- **Memory**: four-tier system with automatic extraction and feedback loop. See [docs/memory.md](docs/memory.md).
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
| `/mcp` | List connected MCP servers and tools |
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
| `TARS_MODEL_DEFAULT` | `claude:sonnet` | Primary model (`provider:model`) |
| `TARS_MODEL_REMOTE` | — | Remote model for tool calls (`provider:model`, explicit versions recommended; set to `none` to disable) |
| `TARS_MODEL_EMBEDDING` | `qwen3-embedding:8b` | Embedding model for indexing and search |
| `TARS_MODEL_RETRIEVAL` | `gemma3:4b` | Local model for query rewriting and HyDE |
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
| `TARS_AUTO_EXTRACT` | `true` | Enable automatic fact extraction on session save/compact |
| `TARS_API_TOKEN` | — | Optional bearer token for API auth |
| `TARS_SCHEDULES` | — | JSON array of scheduled tasks (alternative to `schedules.json`) |
| `TARS_MCP_SERVERS` | — | JSON object of MCP server configs (alternative to `mcp_servers.json`) |
| `TARS_STRAVA_CLIENT_ID` | — | Strava OAuth app client ID |
| `TARS_STRAVA_CLIENT_SECRET` | — | Strava OAuth app client secret |
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
