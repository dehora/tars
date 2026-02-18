import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import anthropic
import ollama
from dotenv import load_dotenv

load_dotenv()


CLAUDE_MODELS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
    "opus": "claude-opus-4-6",
}

DEFAULT_MODEL = "claude:sonnet"

SYSTEM_PROMPT = """\
You are tars, a helpful AI assistant. You have Todoist, weather, and memory tools \
available. You MUST call the appropriate tool when the user asks about tasks, \
reminders, todos, or their schedule. NEVER pretend to have taken an action — \
always use the tool and report the actual result.

When adding tasks, extract the content, due date, project, and priority from \
the user's message. When listing tasks, summarise the results conversationally. \
When completing tasks, confirm what was completed.

For weather questions, use weather_now for current conditions and short-term \
precipitation ("will it rain?"), or weather_forecast for the full day outlook. \
Summarise the data conversationally — include temperature, precipitation chance, \
and conditions. If the user mentions a specific location, pass its coordinates.

You have persistent memory in Memory.md. Use memory_remember to save facts, \
preferences, and rules the user shares. Use memory_recall to check existing \
memory before adding — avoid duplicates. When new information contradicts \
existing memory, update the existing entry rather than appending. \
Sections: "semantic" for facts/preferences, "procedural" for rules/patterns."""

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


def _memory_dir() -> Path | None:
    d = os.environ.get("TARS_MEMORY_DIR")
    if not d:
        return None
    p = Path(d)
    return p if p.is_dir() else None


_MEMORY_FILES = {
    "semantic": "Memory.md",
    "procedural": "Procedural.md",
}


def _memory_file(section: str) -> Path | None:
    d = _memory_dir()
    if d is None:
        return None
    filename = _MEMORY_FILES.get(section)
    return d / filename if filename else None


def _load_memory() -> str:
    """Load Memory.md (semantic) — always included in system prompt."""
    p = _memory_file("semantic")
    if p is None or not p.exists():
        return ""
    return p.read_text()


def _build_system_prompt() -> str:
    memory = _load_memory()
    if not memory:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n---\n\n{memory}"


def _append_to_file(p: Path, content: str) -> None:
    """Append a list item to a memory file, replacing comment placeholders."""
    text = p.read_text() if p.exists() else ""
    # Remove comment placeholders
    text = re.sub(r"<!--.*?-->\n?", "", text)
    text = text.rstrip() + f"\n- {content}\n"
    p.write_text(text)


def _run_memory_tool(name: str, args: dict) -> str:
    if name == "memory_recall":
        d = _memory_dir()
        if d is None:
            return json.dumps({"error": "Memory not configured (TARS_MEMORY_DIR not set)"})
        result = {}
        for section, filename in _MEMORY_FILES.items():
            p = d / filename
            if p.exists():
                result[section] = p.read_text()
        if not result:
            return json.dumps({"error": "No memory files found"})
        return json.dumps(result)

    if name == "memory_update":
        # Search both files for the old entry
        d = _memory_dir()
        if d is None:
            return json.dumps({"error": "Memory not configured (TARS_MEMORY_DIR not set)"})
        old_line = f"- {args['old_content'].strip()}"
        new_line = f"- {args['new_content'].strip()}"
        for filename in _MEMORY_FILES.values():
            p = d / filename
            if not p.exists():
                continue
            text = p.read_text()
            if old_line in text:
                text = text.replace(old_line, new_line, 1)
                p.write_text(text)
                return json.dumps({"ok": True, "old": args["old_content"], "new": args["new_content"]})
        return json.dumps({"error": f"Could not find existing entry: {args['old_content']}"})

    # memory_remember
    section = args["section"]
    p = _memory_file(section)
    if p is None:
        return json.dumps({"error": "Memory not configured (TARS_MEMORY_DIR not set)"})
    content = args["content"].strip()
    _append_to_file(p, content)
    return json.dumps({"ok": True, "section": section, "content": content})


WMO_WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def _fetch_weather(lat: float, lon: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,precipitation,weather_code,wind_speed_10m"
        f"&hourly=temperature_2m,precipitation_probability,precipitation"
        f"&forecast_days=1&timezone=auto"
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _resolve_coords(args: dict) -> tuple[float | None, float | None]:
    lat = args["lat"] if "lat" in args and args["lat"] is not None else None
    lon = args["lon"] if "lon" in args and args["lon"] is not None else None

    if lat is None:
        env_lat = os.environ.get("DEFAULT_LAT")
        lat = float(env_lat) if env_lat not in (None, "") else None
    if lon is None:
        env_lon = os.environ.get("DEFAULT_LON")
        lon = float(env_lon) if env_lon not in (None, "") else None

    return lat, lon


def _run_weather_tool(name: str, args: dict) -> str:
    lat, lon = _resolve_coords(args)
    if lat is None or lon is None:
        return json.dumps({"error": "No location provided and DEFAULT_LAT/DEFAULT_LON not set in .env"})
    try:
        data = _fetch_weather(lat, lon)
    except Exception as e:
        return json.dumps({"error": f"Weather API request failed: {e}"})

    if name == "weather_now":
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        # Next 6 hours of precipitation data
        precip_probs = hourly.get("precipitation_probability", [])[:6]
        precip_amounts = hourly.get("precipitation", [])[:6]
        temps = hourly.get("temperature_2m", [])[:6]
        times = hourly.get("time", [])[:6]
        series_len = min(
            len(times),
            len(temps),
            len(precip_probs),
            len(precip_amounts),
        )
        weather_code = current.get("weather_code", 0)
        return json.dumps({
            "current": {
                "temperature_c": current.get("temperature_2m"),
                "precipitation_mm": current.get("precipitation"),
                "conditions": WMO_WEATHER_CODES.get(weather_code, f"Code {weather_code}"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
            },
            "next_hours": [
                {
                    "time": times[i] if i < len(times) else None,
                    "temp_c": temps[i] if i < len(temps) else None,
                    "precip_prob_pct": precip_probs[i] if i < len(precip_probs) else None,
                    "precip_mm": precip_amounts[i] if i < len(precip_amounts) else None,
                }
                for i in range(series_len)
            ],
            "location": {"lat": lat, "lon": lon},
        })

    # weather_forecast — full day hourly
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precip_probs = hourly.get("precipitation_probability", [])
    precip_amounts = hourly.get("precipitation", [])
    series_len = min(len(times), len(temps), len(precip_probs), len(precip_amounts))
    return json.dumps({
        "hourly": [
            {
                "time": times[i],
                "temp_c": temps[i] if i < len(temps) else None,
                "precip_prob_pct": precip_probs[i] if i < len(precip_probs) else None,
                "precip_mm": precip_amounts[i] if i < len(precip_amounts) else None,
            }
            for i in range(series_len)
        ],
        "location": {"lat": lat, "lon": lon},
    })


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


def parse_model(model_str: str) -> tuple[str, str]:
    provider, _, model = model_str.partition(":")
    if not model:
        raise ValueError(f"Invalid model format '{model_str}', expected provider:model")
    return provider, model


def chat_anthropic(messages: list[dict], model: str) -> str:
    resolved = CLAUDE_MODELS.get(model, model)
    client = anthropic.Anthropic()
    local_messages = [m.copy() for m in messages]

    while True:
        response = client.messages.create(
            model=resolved,
            max_tokens=1024,
            system=_build_system_prompt(),
            messages=local_messages,
            tools=ANTHROPIC_TOOLS,
        )

        if response.stop_reason != "tool_use":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # Process tool calls
        assistant_content = list(response.content)
        local_messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        local_messages.append({"role": "user", "content": tool_results})


def chat_ollama(messages: list[dict], model: str) -> str:
    local_messages = [{"role": "system", "content": _build_system_prompt()}]
    local_messages.extend(m.copy() for m in messages)

    while True:
        response = ollama.chat(model=model, messages=local_messages, tools=OLLAMA_TOOLS)

        if not response.message.tool_calls:
            return response.message.content or ""

        # Append the assistant message with tool calls
        local_messages.append(response.message)

        # Execute each tool call and feed results back
        for tool_call in response.message.tool_calls:
            result = run_tool(tool_call.function.name, tool_call.function.arguments)
            local_messages.append({
                "role": "tool",
                "content": result,
            })


def chat(messages: list[dict], provider: str, model: str) -> str:
    if provider == "claude":
        return chat_anthropic(messages, model)
    if provider == "ollama":
        return chat_ollama(messages, model)
    raise ValueError(f"Unknown provider: {provider}")


def repl(provider: str, model: str):
    messages = []
    print(f"tars [{provider}:{model}] (ctrl-d to quit)")
    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue
        messages.append({"role": "user", "content": user_input})
        reply = chat(messages, provider, model)
        messages.append({"role": "assistant", "content": reply})
        print(f"tars> {reply}")


def main():
    parser = argparse.ArgumentParser(prog="tars", description="tars AI assistant")
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help="provider:model (e.g. ollama:gemma3:12b, claude:sonnet)",
    )
    parser.add_argument("message", nargs="*", help="message for single-shot mode")
    args = parser.parse_args()

    provider, model = parse_model(args.model)

    if args.message:
        message = " ".join(args.message)
        print(chat([{"role": "user", "content": message}], provider, model))
    else:
        repl(provider, model)


if __name__ == "__main__":
    main()
