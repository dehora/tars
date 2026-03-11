from __future__ import annotations

import json
import os
import subprocess
import shutil
from typing import TYPE_CHECKING

from tars.debug import verbose
from tars.memory import _run_memory_tool
from tars.notes import _run_note_tool
from tars.search import _run_notes_search_tool, _run_search_tool
from tars.weather import _run_weather_tool
from tars.strava import _run_strava_tool
from tars.web import _run_web_tool

if TYPE_CHECKING:
    from tars.mcp import MCPClient

ANTHROPIC_TOOLS = [
    {
        "name": "todoist_add_task",
        "description": "Add a task to Todoist",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Task description"},
                "due": {"type": "string", "description": "Due date/time, e.g. 'today', 'tomorrow 3pm', '2024-03-15'"},
                "project": {"type": "string", "description": "Project name"},
                "priority": {"type": "integer", "description": "Priority 1-4 (4 is urgent)", "enum": [1, 2, 3, 4]},
            },
            "required": ["content"],
        },
    },
    {
        "name": "todoist_today",
        "description": "List tasks due today",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "todoist_upcoming",
        "description": "List upcoming tasks for the next N days",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of days to look ahead (default 7)"},
            },
        },
    },
    {
        "name": "todoist_complete_task",
        "description": "Complete/close a task by reference (task ID or search text)",
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Task ID or search text to identify the task"},
            },
            "required": ["ref"],
        },
    },
    {
        "name": "weather_now",
        "description": "Get current weather conditions and precipitation forecast for the next few hours. Use for questions like 'will it rain?', 'do I need an umbrella?', 'what's the temperature?'",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude (defaults to user's location)"},
                "lon": {"type": "number", "description": "Longitude (defaults to user's location)"},
            },
        },
    },
    {
        "name": "weather_forecast",
        "description": "Get today's full hourly weather forecast. Use for questions like 'what's the weather today?', 'forecast for today'",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude (defaults to user's location)"},
                "lon": {"type": "number", "description": "Longitude (defaults to user's location)"},
            },
        },
    },
    {
        "name": "memory_remember",
        "description": "Save information to persistent memory. Use when the user shares facts, preferences, or rules they want remembered across sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The information to remember (a single line)"},
                "section": {
                    "type": "string",
                    "description": "Which memory section: 'semantic' for facts/preferences, 'procedural' for rules/patterns, 'pinned' for persistent brief items",
                    "enum": ["semantic", "procedural", "pinned"],
                },
            },
            "required": ["content", "section"],
        },
    },
    {
        "name": "memory_update",
        "description": "Update an existing memory entry. Use when new information contradicts or supersedes something already in memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "old_content": {"type": "string", "description": "The existing memory line to replace (exact match)"},
                "new_content": {"type": "string", "description": "The replacement text"},
            },
            "required": ["old_content", "new_content"],
        },
    },
    {
        "name": "memory_forget",
        "description": "Remove an entry from persistent memory. Use when the user asks to forget something or when information is no longer relevant.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The exact memory line to remove"},
                "section": {
                    "type": "string",
                    "description": "Limit removal to a specific section (semantic, procedural, pinned). If omitted, searches all sections.",
                    "enum": ["semantic", "procedural", "pinned"],
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_recall",
        "description": "Read current persistent memory. Use before adding new memories to check for duplicates or contradictions.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Search persistent memory (semantic, procedural, and episodic) using "
            "hybrid keyword + semantic search. Use this to find relevant context, "
            "past conversations, preferences, or facts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5)",
                },
                "window": {
                    "type": "integer",
                    "description": "Number of neighboring chunks to include for context (default 1)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "notes_search",
        "description": (
            "Search the user's personal notes vault (Obsidian) using hybrid keyword + "
            "semantic search. Use this when the user asks about their own notes, daily "
            "journals, or personal knowledge base — NOT for tars memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in the user's notes",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5)",
                },
                "window": {
                    "type": "integer",
                    "description": "Number of neighboring chunks to include for context (default 2)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "note_daily",
        "description": "Append a note to today's Obsidian daily note. Use when the user wants to jot something down or save a thought.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The note content to append"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "web_read",
        "description": "Fetch and read a web page. Use when the user shares a URL or asks you to read/discuss a link.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "strava_activities",
        "description": (
            "Fetch Strava activities with optional filters. Use when the user asks about "
            "runs, rides, workouts, training, or exercise activities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Time period: '7d', '30d', '3m', 'this-week', 'last-week', 'this-month', 'last-month', 'this-year', 'ytd'",
                },
                "type": {
                    "type": "string",
                    "description": "Activity type filter: 'Run', 'Ride', 'Swim', 'Walk', 'Hike', etc.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20, max 100)",
                },
                "id": {
                    "type": "integer",
                    "description": "Fetch a single activity by ID (overrides other params)",
                },
                "sort": {
                    "type": "string",
                    "description": "Sort order: 'recent' (default) or 'oldest'",
                    "enum": ["recent", "oldest"],
                },
            },
        },
    },
    {
        "name": "strava_summary",
        "description": (
            "Aggregate Strava activities for a single period into per-type summaries: "
            "totals (distance, time, elevation), averages (pace, HR, cadence), "
            "and effort scores. Use for simple period breakdowns. "
            "For trend comparison, use strava_analysis. "
            "Results are capped at 200 activities per period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Time period: '7d', '30d', '3m', 'this-week', 'last-week', 'this-month', 'last-month', 'this-year', 'ytd' (default: 'this-month')",
                },
                "type": {
                    "type": "string",
                    "description": "Optional activity type filter: 'Run', 'Ride', 'Swim', etc.",
                },
            },
        },
    },
    {
        "name": "strava_user",
        "description": (
            "Fetch Strava athlete profile, stats, zones, and gear. Use when the user asks "
            "about their running/cycling stats, fitness profile, or training totals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["profile", "stats", "zones", "gear"],
                    },
                    "description": "Sections to include (default: ['profile', 'stats'])",
                },
            },
        },
    },
    {
        "name": "strava_compare",
        "description": (
            "Compare Strava activity summaries across two periods. Use for period-over-period "
            "analysis like 'how does this month compare to last month?' or 'am I running more "
            "than last year?'. Auto-derives comparison period if not specified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period_a": {
                    "type": "string",
                    "description": "The period to analyze: '7d', '30d', '3m', 'this-week', 'last-week', 'this-month', 'last-month', 'this-year', 'ytd'",
                },
                "period_b": {
                    "type": "string",
                    "description": "Comparison period (auto-derived if omitted — e.g. this-month auto-compares to last-month)",
                },
                "type": {
                    "type": "string",
                    "description": "Optional activity type filter: 'Run', 'Ride', 'Swim', etc.",
                },
            },
            "required": ["period_a"],
        },
    },
    {
        "name": "strava_analysis",
        "description": (
            "Analyse Strava training for a period with automatic trend comparison "
            "to the previous period. Returns overall totals across all activity types, "
            "per-type breakdowns, and period-over-period deltas. Use for 'analyse my "
            "training this week', 'how does this month compare?', or weekly summaries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Time period: '7d', '30d', '3m', 'this-week', 'last-week', 'this-month', 'last-month', 'this-year', 'ytd' (default: 'this-week')",
                },
                "compare_period": {
                    "type": "string",
                    "description": "Comparison period (auto-derived if omitted — e.g. this-week auto-compares to last-week)",
                },
                "type": {
                    "type": "string",
                    "description": "Optional activity type filter: 'Run', 'Ride', 'Swim', etc.",
                },
            },
        },
    },
    {
        "name": "strava_routes",
        "description": (
            "Browse Strava routes and starred segments. Use when the user asks about "
            "saved routes, route details, or starred/favourite segments."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "What to do: 'list' routes, 'detail' for a specific route, 'starred' for starred segments",
                    "enum": ["list", "detail", "starred"],
                },
                "id": {
                    "type": "integer",
                    "description": "Route ID — required for action 'detail'",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20, max 50)",
                },
            },
            "required": ["action"],
        },
    },
]

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in ANTHROPIC_TOOLS
]

_mcp_client: MCPClient | None = None


def set_mcp_client(client: MCPClient | None) -> None:
    global _mcp_client
    _mcp_client = client


def get_mcp_client() -> MCPClient | None:
    return _mcp_client


def _to_ollama_format(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


def get_all_tools(mcp_client: MCPClient | None = None) -> tuple[list, list]:
    """Return (anthropic_tools, ollama_tools) with native + MCP tools merged."""
    client = mcp_client or _mcp_client
    anthropic = list(ANTHROPIC_TOOLS)
    if client:
        anthropic.extend(client.discover_tools())
    ollama = [_to_ollama_format(t) for t in anthropic]
    return anthropic, ollama


_TOOL_REQUIRED: dict[str, list[str]] = {
    t["name"]: t["input_schema"].get("required", [])
    for t in ANTHROPIC_TOOLS
}


def _clean_args(args: dict) -> dict:
    """Strip empty-string and None optional params that models fill in needlessly."""
    return {k: v for k, v in args.items() if v is not None and v != ""}


def _resolve_td() -> str | None:
    """Resolve the todoist CLI binary, honoring TARS_TD if set."""
    td_env = os.environ.get("TARS_TD", "").strip()
    if td_env:
        if os.path.isfile(td_env) and os.access(td_env, os.X_OK):
            return td_env
        return None
    td_bin = shutil.which("td")
    if td_bin:
        return td_bin
    # Check fnm stable paths (Node.js version manager)
    fnm_base = os.path.expanduser("~/.local/share/fnm/node-versions")
    if os.path.isdir(fnm_base):
        try:
            for ver in sorted(os.listdir(fnm_base), reverse=True):
                fnm_td = os.path.join(fnm_base, ver, "installation", "bin", "td")
                if os.path.isfile(fnm_td) and os.access(fnm_td, os.X_OK):
                    return fnm_td
        except OSError:
            pass
    for candidate in (
        os.path.expanduser("~/.local/bin/td"),
        "/opt/homebrew/bin/td",
        "/usr/local/bin/td",
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def run_tool(name: str, args: dict, *, quiet: bool = False) -> str:
    args = _clean_args(args)
    missing = [f for f in _TOOL_REQUIRED.get(name, []) if f not in args]
    if missing:
        return json.dumps({"error": f"missing required field(s) for {name}: {', '.join(missing)}"})
    if not quiet:
        verbose(f"  [tool] {name}({args})")
    try:
        if name in ("memory_remember", "memory_recall", "memory_update", "memory_forget"):
            return _run_memory_tool(name, args)
        if name == "memory_search":
            return _run_search_tool(name, args)
        if name == "notes_search":
            return _run_notes_search_tool(name, args)
        if name in ("weather_now", "weather_forecast"):
            return _run_weather_tool(name, args)
        if name == "note_daily":
            return _run_note_tool(name, args)
        if name == "web_read":
            return _run_web_tool(name, args)
        if name in ("strava_activities", "strava_user", "strava_summary", "strava_compare", "strava_analysis", "strava_routes"):
            return _run_strava_tool(name, args)
        if name.startswith("todoist_"):
            td_bin = _resolve_td()
            if not td_bin:
                return json.dumps({"error": "td CLI not found — install with: pip install todoist-cli or set TARS_TD to its path"})
        if name == "todoist_add_task":
            cmd = [td_bin, "task", "add", args["content"]]
            if due := args.get("due"):
                cmd.extend(["--due", due])
            if project := args.get("project"):
                cmd.extend(["--project", project])
            if priority := args.get("priority"):
                cmd.extend(["--priority", str(priority)])
        elif name == "todoist_today":
            cmd = [td_bin, "today", "--json"]
        elif name == "todoist_upcoming":
            days = args.get("days", 7)
            cmd = [td_bin, "upcoming", str(days), "--json"]
        elif name == "todoist_complete_task":
            cmd = [td_bin, "task", "complete", args["ref"]]
        else:
            if _mcp_client and "." in name:
                return _mcp_client.call_tool(name, args)
            return json.dumps({"error": f"Unknown tool: {name}"})

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return json.dumps({"error": result.stderr.strip() or f"td exited with code {result.returncode}"})
        return result.stdout.strip() or json.dumps({"ok": True})
    except FileNotFoundError:
        return json.dumps({"error": "td CLI not found — install with: pip install todoist-cli or set TARS_TD to its path"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "td command timed out"})
