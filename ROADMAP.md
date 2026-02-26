# Roadmap

## Next

### 13. Email search (`/search` over email)

Add `/search <query>` as a slash command over email. The search infrastructure and email slash command dispatch both exist. (Note: `/search` is already available via Telegram.)

Why: parity with Telegram channel. Very low effort given existing pieces.

### 7. Context-aware tool suggestions

A lightweight intent pre-filter (few-shot prompt or regex patterns) before the main model to improve tool routing. When Memory.md says "I use Todoist for todos" and the user says "remind me to buy eggs", route confidently to todoist.

Why: better routing = fewer `/w` corrections = better Procedural.md rules = better routing. Virtuous cycle.

### 14. Inbound webhooks

A `/webhook` endpoint that accepts POST from IFTTT/Zapier/GitHub and routes through conversation. "GitHub issue assigned to you" → tars creates a todoist task. Turns tars into a personal automation hub.

Why: connects tars to external event sources. Medium effort — needs auth, payload parsing, and action mapping.

### 16. Conversation export

Dump a conversation or search results as markdown. Useful for pulling tars knowledge back into Obsidian manually or sharing context.

Why: closes the loop between tars conversations and the PKM vault.

### 17. Voice notes via email

Whisper (local via ollama or API) transcribing voice memos sent as email attachments. Email an audio note from your phone → tars transcribes → processes as text. The email infra already handles attachments.

Why: lowest friction input method. Big UX jump but depends on reliable transcription.

### 19. Proactive nudges

Tars notices patterns in session history ("you add groceries every Thursday") and suggests things. Requires scheduled commands (item 9) plus lightweight pattern detection over sessions.

Why: moves tars from tool to partner. Highest ambition item — depends on several other pieces landing first.

### 22. Safari capture extension

Browser extension (Safari Web Extension) that adds a "Send to tars" button. Captures the current page URL and sends it to `/capture` via the API. Could use the existing `POST /tool` endpoint or a dedicated `POST /capture` route. Lets you capture articles while browsing without switching to the CLI or email.

Why: lowest friction capture path — one click from the browser. Safari Web Extensions use the same WebExtension API as Chrome/Firefox, so it could be portable later.

### 26. Procedural rule auto-ingest

When `/review` produces new rules, automatically re-index the procedural file so the rules are immediately searchable. Currently requires a manual `tars index` after review.

Why: closes a gap in the feedback loop. Rules should be live the moment they're accepted.

## Fixes

- tars/cli.py, tars/email.py, and tars/telegram.py: Slash command handling is duplicated across channels as if/elif chains. As more commands are added this will become unwieldy. Suggested fix: extract shared dispatch to `tars/commands.py` with a dispatch table mapping command names to handler functions.

- tars/tools.py _clean_args strips all empty strings. Suggested fix: Consider tool-specific sanitization or only stripping empty strings for parameters.

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

### 23. Capture enrichment

Enhance `/capture` with metadata extraction: author, publish date, tags, reading time. Store in YAML frontmatter. Could also extract and save key quotes or generate a TL;DR alongside the full summary. Makes captures more useful as Obsidian notes.

</details>
