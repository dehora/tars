# Roadmap

## Next

### 30. QoL: CLI

- Tab completion for slash commands
- Inline help for tool arguments (e.g. `/capture` shows usage on empty args)
- Friendlier error messages (model errors, network failures, missing config)

### 31. QoL: Web UI

- Conversation history browser (list past conversations, click to view)
- Model indicator (which model handled each message)
- Streaming progress indicator

### 33. QoL: Cross-channel visibility

- Shared view of conversations and state across channels (CLI, web, email, Telegram)
- Currently each channel feels isolated — no way to see what happened in another

## Fixes

- tars/tools.py _clean_args strips all empty strings. Suggested fix: Consider tool-specific sanitization or only stripping empty strings for parameters.

<details>
<summary>Parked</summary>

### 16. Conversation export

Dump a conversation or search results as markdown.

### 19. Proactive nudges

Pattern detection over session history for suggestions.

</details>

<details>
<summary>Done</summary>

### 1. Memory hygiene command (`/tidy`)

Loads Memory.md and Procedural.md, sends them to the model with "find duplicates, stale entries, test data, and contradictions — propose removals", lets the user approve. Same interactive pattern as `/review` but for memory quality instead of corrections.

### 2. Session browsing (`/sessions`, `/session <query>`)

Session logs exist but there's no way to browse them. `/sessions` shows recent session summaries (date + topic line), `/session <query>` searches them. Makes episodic memory visible — "what did we talk about on Tuesday?"

### 3. Memory deduplication guard

`memory_remember` currently appends blindly. Before writing, search existing entries for near-duplicates (exact substring match is enough). Procedural.md rule says "use memory_recall before adding" but the tool doesn't enforce it.

### 4. Daily briefing (`/brief`)

Runs todoist_today + weather_now + weather_forecast in parallel, formats into a single digest. One command instead of three.

### 5. Prompt routing confidence

Instead of binary right/wrong routing, the model emits a confidence signal. When uncertain whether a request is a tool call or chat, tars asks before acting. Primarily a system prompt / procedural rule change.

### 6. Web UI search

The `GET /search` endpoint exists but the web UI doesn't use it. Add a `/search` command in web chat that renders results inline, matching CLI parity.

### 8. Obsidian daily note integration (`/note`)

Appends to today's Obsidian daily note (or creates it). "Note: interesting idea about X" → appends to `2026-02-20.md` in the vault. Bridges tars and existing note-taking.

### 10. Email channel (`tars email`)

IMAP polling + SMTP reply via dedicated Gmail account with app password. Polls inbox every 60s, processes emails from whitelisted senders through `conversation.py`, replies in-thread. Supports slash commands for reliable tool execution. Python stdlib only.

### 11. Multi-model routing

Keyword-based pre-routing: local model for chat, Claude for tool calls. Falls back gracefully on API errors (rate limits, billing, outages). Logs routing decisions to stderr for observability.

### 15. Memory stats and health (`/stats`)

`/stats` command showing DB size, file count, chunk count, embedding model/dimensions, and session count. Available in CLI, web UI, and API (`GET /stats`).

### 18. RAG over full Obsidian vault

Point the indexer at the full personal Obsidian vault (not just the tars vault). Tars can answer questions about your own notes, surface relevant context from past thinking. Implemented as `tars notes-index` (builds `notes.db` in `TARS_NOTES_DIR`) and `/find <query>` REPL command.

### 20. Web reading (`web_read` tool)

Fetch and extract text content from web pages. Model calls `web_read` when a user shares a URL. HTML stripped to text via stdlib parser, truncated to 12k chars. Available as `/read` in email.

### 21. Web capture (`/capture`)

Fetch a web page, optionally summarize with the model (stripping boilerplate), save to `TARS_NOTES_DIR/17 tars captures/` with YAML frontmatter. Available in CLI and email.

### 24. Conversation-aware captures

When `/capture` is called mid-conversation, recent conversation context is passed to the summarization prompt so the summary emphasizes aspects relevant to what's being discussed. Email captures remain contextless (standalone by nature).

### 25. Tool parameter sanitization

`_clean_args()` strips empty strings and None values from tool args at the `run_tool()` boundary. Fixes the class of issues where ollama models fill in every optional parameter with empty strings.

### 27. Vault search command (`/find`, `tars notes-index`)

Add a `/find <query>` REPL command and `tars notes-index` CLI subcommand that searches the personal Obsidian vault (`TARS_NOTES_DIR`) via a separate `notes.db` index. Same chunk → embed → sqlite-vec pipeline as tars memory, with hidden directories (`.obsidian`, `.trash`) and inline base64 images excluded.

### 9. Scheduled / recurring commands

`tars schedule` CLI manages OS-level timers (launchd on macOS, systemd on Linux) for any tars subcommand. Supports calendar triggers (`--hour`/`--minute`) and file watchers (`--watch`). Includes `tars schedule test` to dry-run with baked environment. Replaces `bin/tars-schedule-{mac,linux}` scripts.

### 12. Email digest (`tars email-brief`)

`/brief` results emailed on a schedule via `tars email-brief` subcommand. Wired to OS scheduling via `tars schedule add`.

### 28. Telegram channel (`tars telegram`)

Telegram bot polling channel via `python-telegram-bot`. Persistent reply keyboard for one-tap commands, slash command dispatch, conversation support, daily brief sender (`tars telegram-brief`). User filtering via `TARS_TELEGRAM_ALLOW` user IDs.

### 29. Chunker & indexer quality improvements

Heading context breadcrumbs on chunks (`H1 > H2 > H3`), reduced chunk size (800 → 400 tokens), list cohesion (score 5 → 1), and context-aware embeddings. Heading context is prepended to embed input only — stored content and FTS unchanged. Tagged `find-v1` before changes. Requires reindex.

### 7. Context-aware tool suggestions

RouteResult with tool hints and procedural memory injected into system prompt. When Memory.md says "I use Todoist for todos" and the user says "remind me to buy eggs", route confidently to todoist.

### 26. Procedural rule auto-ingest

`_apply_review()` in `cli.py` calls `build_index()` after writing rules to Procedural.md. Incremental indexing detects the changed content_hash and re-indexes only the modified file.

### 32. QoL: Email reliability

Processing failures now retry instead of immediately sending error replies. Slash dispatch wrapped in try/except. Empty body emails get a reply instead of silent Seen flag. Max-retry path logs clearly. BODY.PEEK ensures messages stay unseen until successfully processed and replied to.

### 23. Capture enrichment

Enhance `/capture` with metadata extraction: author, publish date, tags, reading time. Store in YAML frontmatter. Could also extract and save key quotes or generate a TL;DR alongside the full summary. Makes captures more useful as Obsidian notes.

</details>
