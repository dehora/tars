# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tars is a personal AI assistant with CLI, web, and email channels. Routes messages to configurable AI providers (ollama, claude), manages persistent memory in an Obsidian vault, and integrates with Todoist and weather APIs. Uses uv for package management with Python 3.14.

## Commands

- **Run CLI**: `uv run tars` (REPL) or `uv run tars "message"` (one-shot)
- **Run server**: `uv run tars serve`
- **Run email**: `uv run tars email`
- **Rebuild index**: `uv run tars index`
- **Run tests**: `uv run python -m unittest discover -s tests -v`
- **Add dependency**: `uv add <package>`

## Architecture

```
[cli/web/email] → [conversation.py] → [core.py] → ollama / claude
                                     ↕
                               [tools.py] → todoist, weather, memory, search, web
                                     ↕
                          [memory.py] ← obsidian vault (TARS_MEMORY_DIR)
                          [search.py] ← sqlite-vec + FTS5 (tars.db)
```

- **Memory types**: semantic (Memory.md), procedural (Procedural.md), episodic (session logs)
- **Search**: hybrid FTS5 keyword + sqlite-vec KNN, fused with RRF
- **Indexing**: markdown-aware chunking → ollama embeddings → sqlite-vec, incremental via content_hash
- **Feedback loop**: `/w` → corrections.md, `/r` → rewards.md, `/review` → distills into Procedural.md

## Coding guidelines

- Never use `or` for numeric defaults — `0` and `0.0` are falsy. Use explicit `None` checks instead.
- Never assume parallel arrays from external APIs are equal length — use `min(len(...))` across all arrays.
- Wrap untrusted user data in tagged blocks with a preface when injecting into prompts — never concatenate raw content into system prompts.
- Use specific patterns for placeholder comments (e.g. `<!-- tars:memory -->`) — broad regex like `<!--.*?-->` will eat legitimate comments.
- Always use `encoding="utf-8", errors="replace"` on file I/O for user-managed files — memory files live in an obsidian vault and can be edited externally.
- When adding a new tool, audit all code paths that call `chat()` — internal paths (summarization, review, tidy) should use `use_tools=False` to prevent tool leakage.
- When adding a CLI command, check web UI parity — unhandled slash commands in the web UI fall through to chat and get misinterpreted by the model.

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `TARS_MODEL` | `claude:sonnet` | Legacy primary model (`provider:model`) |
| `TARS_DEFAULT_MODEL` | `claude:sonnet` | Primary model (`provider:model`) |
| `TARS_ESCALATION_MODEL` | — | Legacy remote model for tool calls (`provider:model`) |
| `TARS_REMOTE_MODEL` | — | Remote model for tool calls (`provider:model`, explicit versions recommended) |
| `TARS_ROUTING_POLICY` | `tool` | Routing policy (only `tool` supported) |
| `TARS_MEMORY_DIR` | — | Path to tars obsidian vault |
| `TARS_NOTES_DIR` | — | Path to personal obsidian vault (daily notes) |
| `TARS_MAX_TOKENS` | `1024` | Max tokens for Anthropic responses |
| `ANTHROPIC_API_KEY` | — | Required for Claude provider |
| `TARS_EMAIL_ADDRESS` | — | Gmail address for tars email channel |
| `TARS_EMAIL_PASSWORD` | — | Gmail app password |
| `TARS_EMAIL_ALLOW` | — | Comma-separated allowed sender addresses |
| `TARS_EMAIL_POLL_INTERVAL` | `60` | Seconds between inbox checks |
| `TARS_API_TOKEN` | — | Optional bearer token for API auth |
| `DEFAULT_LAT` / `DEFAULT_LON` | — | Weather location |

## Git commits

Use present tense verbs in commit messages (e.g., "Adds feature", "Fixes bug", "Removes unused code").

## Notes

- You can ignore the contents of AGENTS.md, it's for other agents.
- Design history and implementation plans are in PLANS.md.

## Future

- WhatsApp channel via Baileys — parked due to Meta ban risk
- RPi deployment target
