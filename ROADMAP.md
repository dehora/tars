# Roadmap

## Active

### 1. Memory hygiene command (`/tidy`)

Loads Memory.md and Procedural.md, sends them to the model with "find duplicates, stale entries, test data, and contradictions — propose removals", lets the user approve. Same interactive pattern as `/review` but for memory quality instead of corrections.

Why: the feedback loop only works if the memory it feeds into is clean. Garbage in Procedural.md degrades every future response.

### 3. Memory deduplication guard

`memory_remember` currently appends blindly. Before writing, search existing entries for near-duplicates (exact substring match is enough). The lorem ipsum entries in Memory.md happened because tars didn't check first. Procedural.md rule 12 says "use memory_recall before adding" but the tool doesn't enforce it.

Why: prevents drift. Cheap to implement — string search before `_append_to_file`.

### 4. Daily briefing (`/brief`)

Runs todoist_today + weather_now + weather_forecast in parallel, formats into a single digest, optionally pulls relevant memory context. One command instead of three.

Why: compound command saves time, natural morning routine. Also a template for other compound commands.

## Backlog

### 2. Session browsing (`/sessions`, `/session <query>`)

Session logs exist but there's no way to browse them. `/sessions` shows recent session summaries (date + topic line), `/session <query>` searches them. Makes episodic memory visible — "what did we talk about on Tuesday?"

Why: the payoff of a personal knowledge system is being able to recall past conversations by topic.

### 5. Prompt routing confidence

Instead of binary right/wrong routing, the model emits a confidence signal. When uncertain whether a request is a tool call or chat, tars asks before acting. Primarily a system prompt / procedural rule change, not a code change.

Why: cheaper than fixing misroutes after the fact. The `/w` data shows misrouting is a real problem.

### 6. Web UI search

The `GET /search` endpoint exists but the web UI doesn't use it. Add a `/search` command in web chat that renders results inline, matching CLI parity.

Why: if using the web UI, not having search is a gap.

### 7. Context-aware tool suggestions

A lightweight intent pre-filter (few-shot prompt or regex patterns) before the main model to improve tool routing. When Memory.md says "I use Todoist for todos" and the user says "remind me to buy eggs", route confidently to todoist.

Why: better routing = fewer `/w` corrections = better Procedural.md rules = better routing. Virtuous cycle.

### 8. Obsidian daily note integration (`/note`)

Appends to today's Obsidian daily note (or creates it). "Note: interesting idea about X" → appends to `2026-02-20.md` in the vault. Bridges tars and existing note-taking.

Why: tars writing to daily notes means AI conversations feed back into PKM without manual effort.

### 9. Scheduled / recurring commands

Cron-like `/schedule` that runs `/brief` at 8am or `/todoist today` at a set time, pushing results to a notification channel. This is where the RPi deployment target starts to make sense — always-on, running scheduled tasks.

Why: turns tars from reactive (you ask) to proactive (it tells you). Significant architecture change (daemon mode, notifications).
