# Roadmap

## Done

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

### 20. Web reading (`web_read` tool)

Fetch and extract text content from web pages. Model calls `web_read` when a user shares a URL. HTML stripped to text via stdlib parser, truncated to 12k chars. Available as `/read` in email.

### 21. Web capture (`/capture`)

Fetch a web page, optionally summarize with the model (stripping boilerplate), save to `TARS_NOTES_DIR/17 tars captures/` with YAML frontmatter. Available in CLI and email.

## Active

### 12. Email digest (`tars email-brief`)

`/brief` results emailed on a schedule. Cron job or simple timer that runs the briefing and sends it to you. Both pieces exist (email channel + `/brief`), just wiring them together. Natural first step toward scheduled commands.

Why: turns tars proactive for the most common use case. Minimal new code.

### 13. Email search (`/search` over email)

Add `/search <query>` as a slash command over email. Search memory and sessions from your phone. The search infrastructure and email slash command dispatch both exist.

Why: makes the knowledge system accessible from mobile. Very low effort given existing pieces.

## Backlog

### 7. Context-aware tool suggestions

A lightweight intent pre-filter (few-shot prompt or regex patterns) before the main model to improve tool routing. When Memory.md says "I use Todoist for todos" and the user says "remind me to buy eggs", route confidently to todoist.

Why: better routing = fewer `/w` corrections = better Procedural.md rules = better routing. Virtuous cycle.

### 9. Scheduled / recurring commands

Cron-like `/schedule` that runs `/brief` at 8am or `/todoist today` at a set time, pushing results to a notification channel. This is where the RPi deployment target starts to make sense — always-on, running scheduled tasks.

Why: turns tars from reactive (you ask) to proactive (it tells you). Significant architecture change (daemon mode, notifications).

### 14. Inbound webhooks

A `/webhook` endpoint that accepts POST from IFTTT/Zapier/GitHub and routes through conversation. "GitHub issue assigned to you" → tars creates a todoist task. Turns tars into a personal automation hub.

Why: connects tars to external event sources. Medium effort — needs auth, payload parsing, and action mapping.

### 15. Memory stats and health

Dashboard showing memory count, chunk count, embedding dimensions, last index time, session count, DB size. Useful for knowing if the system is healthy, especially on RPi.

Why: operational visibility. You can't fix what you can't see.

### 16. Conversation export

Dump a conversation or search results as markdown. Useful for pulling tars knowledge back into Obsidian manually or sharing context.

Why: closes the loop between tars conversations and the PKM vault.

### 17. Voice notes via email

Whisper (local via ollama or API) transcribing voice memos sent as email attachments. Email an audio note from your phone → tars transcribes → processes as text. The email infra already handles attachments.

Why: lowest friction input method. Big UX jump but depends on reliable transcription.

### 18. RAG over full Obsidian vault

Point the indexer at the full personal Obsidian vault (not just the tars vault). Tars can answer questions about your own notes, surface relevant context from past thinking.

Why: the ultimate personal knowledge assistant play. The indexer and search already work — just needs a broader scope and careful chunking for diverse note formats.

### 19. Proactive nudges

Tars notices patterns in session history ("you add groceries every Thursday") and suggests things. Requires scheduled commands (item 9) plus lightweight pattern detection over sessions.

Why: moves tars from tool to partner. Highest ambition item — depends on several other pieces landing first.

### 22. Safari capture extension

Browser extension (Safari Web Extension) that adds a "Send to tars" button. Captures the current page URL and sends it to `/capture` via the API. Could use the existing `POST /tool` endpoint or a dedicated `POST /capture` route. Lets you capture articles while browsing without switching to the CLI or email.

Why: lowest friction capture path — one click from the browser. Safari Web Extensions use the same WebExtension API as Chrome/Firefox, so it could be portable later.

### 23. Capture enrichment

Enhance `/capture` with metadata extraction: author, publish date, tags, reading time. Store in YAML frontmatter. Could also extract and save key quotes or generate a TL;DR alongside the full summary. Makes captures more useful as Obsidian notes.

Why: makes captured notes first-class PKM citizens instead of raw dumps.

### 24. Conversation-aware captures

When you `/capture` a URL mid-conversation, tars summarizes it in the context of what you're discussing. "We were talking about AI routing and you captured this article about industrial automation" — the summary focuses on the relevant angle rather than being generic.

Why: context-aware summaries are dramatically more useful than generic ones. The conversation history is already there.

### 25. Tool parameter sanitization

Strip empty strings from optional tool parameters before dispatch. Ollama models fill in every parameter (`'due': '', 'project': ''`) even when they should be omitted. Clean this at the `run_tool()` boundary so tools see clean input regardless of which model generated it.

Why: fixes the class of todoist/tool issues seen with smaller ollama models. One fix point, benefits all tools.

### 26. Procedural rule auto-ingest

When `/review` produces new rules, automatically re-index the procedural file so the rules are immediately searchable. Currently requires a manual `tars index` after review.

Why: closes a gap in the feedback loop. Rules should be live the moment they're accepted.

## Fixes

- tars/cli.py: The user's message content is mutated by prepending an instruction. Suggested fix: Pass the hint as system/metadata (e.g., an extra system prompt) rather than modifying content.

- bin/tars uv run tars depends on the current working directory to locate pyproject.toml. If bin/tars is invoked from outside the repo uv may fail to find the project. Suggested fix: Use `uv --directory /path/to/repo run tars "$@"`

- tars/conversation.py (process_message_stream fallback block). If the escalated stream errors after emitting some deltas, those partial tokens have already been yielded to the client. The fallback stream then yields a second response, resulting in mixed/duplicated output in the UI. Suggested fix: buffer escalation deltas until the stream completes or avoid streaming for escalation and only stream after a successful first chunk.
