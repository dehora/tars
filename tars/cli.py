import argparse
import json
import subprocess
import sys

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
You are tars, a helpful AI assistant. You have Todoist tools available. \
You MUST call the appropriate tool when the user asks about tasks, reminders, \
todos, or their schedule. NEVER pretend to have taken an action — always use \
the tool and report the actual result.

When adding tasks, extract the content, due date, project, and priority from \
the user's message. When listing tasks, summarise the results conversationally. \
When completing tasks, confirm what was completed."""

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
            system=SYSTEM_PROMPT,
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
    local_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
