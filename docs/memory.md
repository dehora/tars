# Memory Architecture

tars has a four-tier memory system. Each tier has different granularity, persistence, and write patterns. All tiers live in the Obsidian vault (`TARS_MEMORY_DIR`) as plain markdown, readable and editable outside of tars.

## Tiers

| Tier | Files | Granularity | Persistence | Written by |
|------|-------|-------------|-------------|------------|
| Semantic | `Memory.md` | Distilled facts | Permanent | User (`/remember`), `/review` promotion |
| Procedural | `Procedural.md` | Behavioural rules | Permanent | `/review` distillation |
| Daily | `YYYY-MM-DD.md` | Notable events per day | Rolling window | Auto (tools, compaction, extraction) |
| Episodic | `sessions/*.md` | Full conversation summaries | Archived | Auto (compaction, session save) |

## Memory flow

```
[conversation]
    ↓ tool calls, corrections, compaction events
[daily file] ← append_daily() writes timestamped entries
    ↓ automatic fact extraction on compaction/save
[daily file] ← [extracted] facts appended
    ↓ /review promotes good entries
[Memory.md / Procedural.md] ← permanent memory
```

## Automatic fact extraction

After a conversation compacts or ends, the model extracts discrete reusable facts (preferences, project names, decisions, corrections) and writes them to the daily file tagged as `[extracted]`. Bad extractions age out naturally. Good ones get promoted to permanent memory via `/review`. Controlled by `TARS_AUTO_EXTRACT` (default: enabled). Skips trivial conversations (fewer than 3 user messages).

## What goes where

- **Semantic memory** — "user prefers dark mode", "project uses Python 3.14", "dog's name is Max"
- **Procedural memory** — "route 'remind me' requests to todoist", "use metric units for weather"
- **Daily memory** — "10:15 added 'buy milk' to todoist", "14:30 session compacted — discussed deployment", "14:30 [extracted] user is deploying to fly.io"
- **Episodic memory** — full session summaries with topics discussed, tools used, decisions made

## System prompt assembly

Memory is assembled into the system prompt in layers:

1. `Memory.md` and `Procedural.md` are always loaded (small, high-signal)
2. Today's daily file loads as `<daily-context>` (cross-session awareness within the day)
3. Hybrid search (FTS5 + vector KNN) auto-retrieves relevant context on first message using [two-pass packing](search.md#two-pass-context-packing) — anchors for breadth, then windowed expansion for depth, under a token budget

All memory blocks are wrapped with an untrusted-data preface to prevent prompt injection from user-edited vault files.

## Feedback loop

- `/w` flags a bad response — saves to `corrections.md`
- `/r` flags a good response — saves to `rewards.md`
- `/review` distills corrections into procedural rules and promotes good extractions to permanent memory
- `/tidy` cleans up stale or duplicate memory entries
- Procedural rules feed back into every future response via the system prompt
