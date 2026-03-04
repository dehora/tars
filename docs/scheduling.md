# Scheduling

tars supports two complementary scheduling mechanisms: in-process scheduling for recurring tasks within a running process, and OS-level scheduling for external invocations.

## In-process scheduling

Long-lived processes (`tars serve`, `tars telegram`, `tars email`) run an in-process scheduler that fires configured tasks as slash commands through the existing dispatch.

Configure via `schedules.json` in the memory dir or the `TARS_SCHEDULES` env var:

```json
[
  {"name": "morning_brief", "schedule": "08:00", "action": "/brief", "deliver": "email"},
  {"name": "todoist_check", "schedule": "*/60", "action": "/todoist today", "deliver": "daily"}
]
```

| Field | Format | Description |
|-------|--------|-------------|
| `name` | string | Unique schedule identifier |
| `schedule` | `"HH:MM"` or `"*/N"` | Daily at time, or every N minutes |
| `action` | string | Slash command to execute |
| `deliver` | `"daily"`, `"email"`, `"telegram"` | Where to send results (default: `daily`) |

Use `/schedule` to show both OS-level and in-process schedules.

## OS-level scheduling

`tars schedule` manages launchd plists (macOS) and systemd timer/path units (Linux).

```bash
# Schedule daily email brief at 8am
tars schedule add email-brief email-brief --hour 8 --minute 0

# Schedule vault reindex on file change
tars schedule add notes-reindex notes-index --watch "$TARS_NOTES_DIR"

# List installed schedules
tars schedule list

# Test-run with baked environment (no live env vars)
tars schedule test email-brief

# Remove a schedule
tars schedule remove email-brief
```

### Environment capture

`scheduler.py` captures known tars env vars from both `.env` (via python-dotenv) and the current environment, baking them into the generated plist/unit files. This means scheduled jobs run with consistent config regardless of the shell environment. Generated config files use `chmod 0o600` since they contain secrets.

### How they complement each other

- **OS scheduling** — for external invocations: daily briefs, periodic reindexing, anything that should run even when no tars process is active
- **In-process scheduling** — for recurring tasks within a running process: periodic checks, scheduled messages through existing channels
