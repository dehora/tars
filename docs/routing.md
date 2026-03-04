# Multi-Model Routing and Tool Calling

tars routes messages between a primary (typically local) model and an optional remote model. The router detects tool intent in user messages and escalates to the remote model when tools are likely needed. If the remote model is unavailable, it falls back to the primary.

## Model configuration

Two model env vars control routing:

| Env var | Purpose |
|---------|---------|
| `TARS_MODEL_DEFAULT` | Primary model for chat (default `claude:sonnet`) |
| `TARS_MODEL_REMOTE` | Remote model for tool calls (optional, set to `none` to disable) |

Both use `provider:model` format (e.g. `ollama:qwen3.5:27b`, `claude:claude-sonnet-4-5-20250929`).

`config.py` loads and validates these at startup. Remote Claude models must use explicit versioned IDs — aliases like `claude:sonnet` are rejected to prevent silent model changes.

## Router

`router.py` decides which model handles each message.

### Tool-intent detection

The router checks user input against two layers:

1. **Literal tool names** — direct mentions of internal tool names (e.g. `todoist_add_task`, `weather_now`, `memory_search`) always escalate. MCP tool names are added at startup via `update_tool_names()`.

2. **Keyword patterns** — 18+ compiled regex patterns that suggest tool intent:

| Category | Example patterns |
|----------|-----------------|
| Todoist | `\btodo\b`, `\bremind\s+me\b`, `\bbuy\b`, `\bgroceries\b` |
| Weather | `\bweather\b`, `\bforecast\b`, `\bwill\s+it\s+rain\b` |
| Memory | `\bremember\b`, `\bforget\b`, `\brecall\b` |
| Notes | `\bnote:`, `\bjot\s+down\b`, `\bmy\s+notes?\b`, `\bobsidian\b` |
| Search | `\bsearch\s+for\b`, `\blook\s+up\b` |
| Web | `https?://`, `\bread\s+this\b` |

### Routing decision

```
user message
    ↓
has tool intent? ──yes──→ remote model configured? ──yes──→ escalate to remote
    │                           │
    no                         no
    ↓                           ↓
primary model              primary model
```

The router returns a `RouteResult(provider, model, tool_hints)` — tool hints tell downstream code which tools the user likely wants.

If default and remote are the same model, no escalation occurs (avoids a no-op switch).

## Tool-calling loop

All three code paths (`chat_anthropic`, `chat_ollama`, `stream_response`) use a bounded tool-calling loop:

```python
for _round in range(_MAX_TOOL_ROUNDS):  # _MAX_TOOL_ROUNDS = 10
    response = model.chat(messages, tools)
    if no tool calls in response:
        break
    execute tool calls
    append results to messages
```

The loop exits when the model returns no tool calls or after 10 rounds. No unbounded `while True` anywhere.

## Fallback

When the remote model is a Claude API model, transient errors trigger a fallback to the primary model:

- `anthropic.APIStatusError` — covers HTTP 400 (billing), 429 (rate limit), 500/529 (overload)
- Fallback only applies when escalated — if the primary model is Claude and it errors, the error re-raises (no fallback to itself)

This means a local ollama primary + Claude remote setup degrades gracefully to local-only when the API is unavailable.
