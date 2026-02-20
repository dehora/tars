import argparse
import readline
import sys
from pathlib import Path

from dotenv import load_dotenv

from tars.conversation import Conversation, process_message, save_session
from tars.core import DEFAULT_MODEL, parse_model
from tars.indexer import build_index
from tars.memory import save_correction, save_reward
from tars.search import search
from tars.sessions import _session_path
from tars.tools import run_tool

load_dotenv()

_SLASH_SEARCH = {"/search": "hybrid", "/sgrep": "fts", "/svec": "vec"}


def _preview_lines(content: str, max_lines: int = 3) -> list[str]:
    """Extract preview lines, skipping leading headings and blanks."""
    lines = content.strip().splitlines()
    start = 0
    for j, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            start = j
            break
    return lines[start : start + max_lines]


def _print_search_results(results, mode: str) -> None:
    """Format and print search results to stdout."""
    if not results:
        print(f"  no results ({mode})")
        return
    for i, r in enumerate(results, 1):
        source = r.file_title or r.file_path
        print(f"  {i}. [{r.score:.3f}] {source}:{r.start_line}-{r.end_line} ({r.memory_type})")
        for line in _preview_lines(r.content):
            print(f"     {line}")
        if i < len(results):
            print()


def _handle_slash_search(user_input: str) -> bool:
    """Handle /search, /sgrep, /svec commands. Returns True if handled."""
    parts = user_input.strip().split(None, 1)
    cmd = parts[0] if parts else ""
    if cmd not in _SLASH_SEARCH:
        return False
    query = parts[1] if len(parts) > 1 else ""
    if not query:
        print(f"  usage: {cmd} <query>")
        return True
    mode = _SLASH_SEARCH[cmd]
    try:
        results = search(query, mode=mode, limit=10)
        _print_search_results(results, mode)
    except Exception as e:
        print(f"  [error] search failed: {e}", file=sys.stderr)
    return True


def _parse_todoist_add(tokens: list[str]) -> dict:
    """Parse '/todoist add content --due D --project P --priority N' into args dict."""
    args: dict = {}
    content_parts: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--due" and i + 1 < len(tokens):
            args["due"] = tokens[i + 1]
            i += 2
        elif t == "--project" and i + 1 < len(tokens):
            args["project"] = tokens[i + 1]
            i += 2
        elif t == "--priority" and i + 1 < len(tokens):
            args["priority"] = int(tokens[i + 1])
            i += 2
        else:
            content_parts.append(t)
            i += 1
    args["content"] = " ".join(content_parts)
    return args


def _handle_slash_tool(user_input: str) -> bool:
    """Handle direct tool commands. Returns True if handled."""
    parts = user_input.strip().split()
    cmd = parts[0] if parts else ""

    if cmd == "/todoist":
        sub = parts[1] if len(parts) > 1 else ""
        if sub == "add" and len(parts) > 2:
            args = _parse_todoist_add(parts[2:])
            result = run_tool("todoist_add_task", args)
        elif sub == "today":
            result = run_tool("todoist_today", {})
        elif sub == "upcoming":
            days = int(parts[2]) if len(parts) > 2 else 7
            result = run_tool("todoist_upcoming", {"days": days})
        elif sub == "complete" and len(parts) > 2:
            result = run_tool("todoist_complete_task", {"ref": " ".join(parts[2:])})
        else:
            print("  usage: /todoist add|today|upcoming|complete ...")
            return True
        print(f"  {result}")
        return True

    if cmd == "/weather":
        print(f"  {run_tool('weather_now', {})}")
        return True

    if cmd == "/forecast":
        print(f"  {run_tool('weather_forecast', {})}")
        return True

    if cmd == "/memory":
        print(f"  {run_tool('memory_recall', {})}")
        return True

    if cmd == "/remember":
        if len(parts) < 3:
            print("  usage: /remember <semantic|procedural> <content>")
            return True
        section = parts[1]
        content = " ".join(parts[2:])
        print(f"  {run_tool('memory_remember', {'section': section, 'content': content})}")
        return True

    return False


def repl(provider: str, model: str):
    conv = Conversation(id="repl", provider=provider, model=model)
    session_file = _session_path()
    history_file = Path.home() / ".tars_history"
    try:
        readline.read_history_file(history_file)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)
    print(f"tars [{provider}:{model}] (ctrl-d to quit)")
    try:
        while True:
            try:
                user_input = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input.strip():
                continue
            if user_input.strip() == "/help":
                print("  tools:")
                print("    /todoist add <text> [--due D] [--project P] [--priority N]")
                print("    /todoist today|upcoming [days]|complete <ref>")
                print("    /weather         current conditions")
                print("    /forecast        today's hourly forecast")
                print("    /memory          show persistent memory")
                print("    /remember <semantic|procedural> <text>")
                print("  search:")
                print("    /search <query>  hybrid keyword + semantic")
                print("    /sgrep <query>   keyword (FTS5/BM25)")
                print("    /svec <query>    semantic (vector KNN)")
                print("  feedback:")
                print("    /w [note]        flag last response as wrong")
                print("    /r [note]        flag last response as good")
                print("  /help              show this help")
                continue
            if user_input.strip().startswith(("/w", "/r")):
                parts = user_input.strip().split(None, 1)
                cmd = parts[0]
                if cmd in ("/w", "/r"):
                    if len(conv.messages) < 2:
                        print("  nothing to flag yet")
                    else:
                        note = parts[1] if len(parts) > 1 else ""
                        fn = save_correction if cmd == "/w" else save_reward
                        result = fn(
                            conv.messages[-2]["content"],
                            conv.messages[-1]["content"],
                            note,
                        )
                        print(f"  {result}")
                    continue
            if _handle_slash_tool(user_input):
                continue
            if _handle_slash_search(user_input):
                continue
            reply = process_message(conv, user_input, session_file)
            print(f"tars> {reply}")
    finally:
        try:
            readline.write_history_file(history_file)
        except OSError:
            pass
        save_session(conv, session_file)


def _run_index(embedding_model: str) -> None:
    """Index memory files and print stats."""
    try:
        stats = build_index(model=embedding_model)
    except RuntimeError as e:
        print(f"  [warning] index update failed ({type(e).__name__}): {e}", file=sys.stderr)
        return
    print(
        f"index: {stats['indexed']} indexed, "
        f"{stats['skipped']} skipped, "
        f"{stats['deleted']} deleted, "
        f"{stats['chunks']} chunks"
    )


def _startup_index() -> None:
    """Run incremental index at startup, silently skip on failure."""
    try:
        build_index()
    except Exception as e:
        print(
            f"  [warning] index update failed ({type(e).__name__}): {e}",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(prog="tars", description="tars AI assistant")
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help="provider:model (e.g. ollama:gemma3:12b, claude:sonnet)",
    )
    sub = parser.add_subparsers(dest="command")
    idx = sub.add_parser("index", help="rebuild memory search index")
    idx.add_argument(
        "--embedding-model",
        default="qwen3-embedding:0.6b",
        help="ollama embedding model to use",
    )
    for cmd, mode in [("search", "hybrid"), ("sgrep", "fts"), ("svec", "vec")]:
        sp = sub.add_parser(cmd, help=f"{mode} search over memory index")
        sp.add_argument("query", nargs="+", help="search query")
        sp.add_argument("-n", "--limit", type=int, default=10, help="max results")
    srv = sub.add_parser("serve", help="start HTTP API server")
    srv.add_argument("--host", default="127.0.0.1", help="bind address")
    srv.add_argument("--port", type=int, default=8180, help="port number")
    parser.add_argument("message", nargs="*", help="message for single-shot mode")
    args = parser.parse_args()

    if args.command == "index":
        _run_index(args.embedding_model)
        return

    if args.command in ("search", "sgrep", "svec"):
        mode = {"search": "hybrid", "sgrep": "fts", "svec": "vec"}[args.command]
        query = " ".join(args.query)
        try:
            results = search(query, mode=mode, limit=args.limit)
            _print_search_results(results, mode)
        except Exception as e:
            print(f"  [error] search failed: {e}", file=sys.stderr)
        return

    if args.command == "serve":
        import uvicorn

        uvicorn.run("tars.api:app", host=args.host, port=args.port)
        return

    provider, model = parse_model(args.model)

    _startup_index()

    if args.message:
        message = " ".join(args.message)
        conv = Conversation(id="oneshot", provider=provider, model=model)
        session_file = _session_path()
        reply = process_message(conv, message, session_file)
        print(reply)
    else:
        repl(provider, model)


if __name__ == "__main__":
    main()
