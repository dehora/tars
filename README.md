# tars

A personal AI assistant with CLI, web, and (future) WhatsApp channels. Routes messages to configurable AI providers, manages persistent memory in an Obsidian vault, and integrates with Todoist, weather APIs, and Obsidian daily notes.

## What it does

tars is a conversational assistant that remembers things across sessions, manages your tasks, checks the weather, and writes to your daily notes. It stores memory in Obsidian markdown files so everything stays readable and editable outside of tars.

**Tools:**
- **Todoist** — add, list, complete tasks via natural language
- **Weather** — current conditions and hourly forecasts (Open-Meteo)
- **Memory** — persistent facts and preferences in an Obsidian vault
- **Daily notes** — append thoughts to today's Obsidian journal entry
- **Search** — hybrid keyword + semantic search across all memory types

**Memory types:**
- **Semantic** (`Memory.md`) — facts, preferences, things you've told tars to remember
- **Procedural** (`Procedural.md`) — behavioural rules learned from feedback
- **Episodic** (`sessions/`) — session logs of past conversations

**Feedback loop:**
- `/w` flags a bad response, `/r` flags a good one
- `/review` distills corrections into procedural rules
- `/tidy` cleans up stale or duplicate memory entries
- Procedural rules feed back into every future response

## How it works

```
[CLI / Web UI] → [conversation.py] → [core.py] → ollama / claude
                                          ↕
                                    [tools.py] → todoist, weather, memory, notes, search
                                          ↕
                             [memory.py] ← obsidian vault (TARS_MEMORY_DIR)
                             [notes.py]  ← obsidian vault (TARS_NOTES_DIR)
                             [search.py] ← sqlite-vec + FTS5 (tars.db)
```

- **Providers**: Claude (Anthropic API) or ollama (local models). Set via `TARS_MODEL`.
- **Search**: markdown-aware chunking → ollama embeddings → sqlite-vec for KNN, FTS5 for keyword, fused with Reciprocal Rank Fusion.
- **Indexing**: incremental via content hash — only re-indexes changed files.
- **Streaming**: CLI and web UI stream responses token-by-token.
- **Sessions**: conversations are summarised and logged to the vault. Compaction keeps context manageable during long sessions.

## Usage

```bash
# REPL
uv run tars

# Single message
uv run tars "what's the weather like?"

# Web UI
uv run tars serve

# Rebuild search index
uv run tars index
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

All commands also work in the web UI.

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `TARS_MODEL` | `claude:sonnet` | Provider and model (`provider:model`) |
| `TARS_MEMORY_DIR` | — | Path to tars Obsidian vault |
| `TARS_NOTES_DIR` | — | Path to personal Obsidian vault (daily notes) |
| `TARS_MAX_TOKENS` | `1024` | Max tokens for Anthropic responses |
| `ANTHROPIC_API_KEY` | — | Required for Claude provider |
| `DEFAULT_LAT` / `DEFAULT_LON` | — | Weather location coordinates |

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo> && cd tars
uv sync
cp .env.example .env  # add your API keys and vault paths
uv run tars
```

## Tests

```bash
uv run python -m unittest discover -s tests -v
```
