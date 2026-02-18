import json
import subprocess
import sys

from tars.memory import _run_memory_tool
from tars.weather import _run_weather_tool

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
                    "description": "Which memory section: 'semantic' for facts/preferences, 'procedural' for rules/patterns",
                    "enum": ["semantic", "procedural"],
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
        "name": "memory_recall",
        "description": "Read current persistent memory. Use before adding new memories to check for duplicates or contradictions.",
        "input_schema": {
            "type": "object",
            "properties": {},
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


def run_tool(name: str, args: dict) -> str:
    print(f"  [tool] {name}({args})", file=sys.stderr)
    try:
        if name in ("memory_remember", "memory_recall", "memory_update"):
            return _run_memory_tool(name, args)
        if name in ("weather_now", "weather_forecast"):
            return _run_weather_tool(name, args)
        if name == "todoist_add_task":
            cmd = ["td", "task", "add", args["content"]]
            if due := args.get("due"):
                cmd.extend(["--due", due])
            if project := args.get("project"):
                cmd.extend(["--project", project])
            if priority := args.get("priority"):
                cmd.extend(["--priority", str(priority)])
        elif name == "todoist_today":
            cmd = ["td", "today", "--json"]
        elif name == "todoist_upcoming":
            days = args.get("days", 7)
            cmd = ["td", "upcoming", str(days), "--json"]
        elif name == "todoist_complete_task":
            cmd = ["td", "task", "complete", args["ref"]]
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return json.dumps({"error": result.stderr.strip() or f"td exited with code {result.returncode}"})
        return result.stdout.strip() or json.dumps({"ok": True})
    except FileNotFoundError:
        return json.dumps({"error": "td CLI not found — install with: pip install todoist-cli"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "td command timed out"})
