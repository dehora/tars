import argparse
import logging
import os
import readline
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from tars.colors import _ENABLED as _COLORS_ENABLED
from tars.colors import bold, cyan, dim, green, link, red, yellow
from tars.commands import command_names, dispatch
from tars.config import apply_cli_overrides, load_model_config, model_summary
from tars.conversation import Conversation, process_message, process_message_stream, save_session
from tars.embeddings import DEFAULT_EMBEDDING_MODEL
from tars.indexer import build_index, build_notes_index
from tars.memory import _append_to_file, _memory_file, archive_feedback
from tars.search import search
from tars.sessions import _session_path

load_dotenv()


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
        title = bold(link(f"file://{r.file_path}", source))
        location = dim(f":{r.start_line}-{r.end_line}")
        meta = dim(f"[{r.score:.3f}] ({r.memory_type})")
        print(f"  {i}. {title}{location} {meta}")
        for line in _preview_lines(r.content):
            print(f"     {line}")
        if i < len(results):
            print()


def _print_schedule_list(schedules: list[dict]) -> None:
    """Format and print schedule list to stdout."""
    if not schedules:
        print("  no schedules installed")
        return
    # Header
    print(f"  {'name':<20} {'trigger':<25} {'last run':<40} {'log'}")
    for s in schedules:
        name = s.get("name", "")
        trigger = s.get("trigger", "")
        last_run = s.get("last_run", "")
        log = s.get("log", "")
        # Shorten home dir in log path
        log = log.replace(str(Path.home()), "~")
        print(f"  {name:<20} {trigger:<25} {last_run:<40} {log}")


_COMMAND_NAMES = command_names() | {"?"}

_SLASH_COMPLETIONS = sorted(
    c + " " if c in {"/todoist", "/remember", "/search", "/sgrep", "/svec",
                      "/find", "/session", "/w", "/r", "/note", "/capture",
                      "/read"} else c
    for c in command_names()
)
_TODOIST_SUBS = ["add ", "today", "upcoming ", "complete "]
_REMEMBER_SUBS = ["semantic ", "procedural "]


def _completer(text: str, state: int) -> str | None:
    """Readline tab completer for slash commands."""
    buf = readline.get_line_buffer()
    if buf.startswith("/todoist "):
        sub = buf[9:]
        matches = [s for s in _TODOIST_SUBS if s.startswith(sub)]
        options = [f"/todoist {m}" for m in matches]
    elif buf.startswith("/remember "):
        sub = buf[10:]
        matches = [s for s in _REMEMBER_SUBS if s.startswith(sub)]
        options = [f"/remember {m}" for m in matches]
    else:
        options = [c for c in _SLASH_COMPLETIONS if c.startswith(text)]
    return options[state] if state < len(options) else None


def _recolor_input(user_input: str) -> None:
    stripped = user_input.strip()
    cmd = stripped.split()[0] if stripped else ""
    if cmd not in _COMMAND_NAMES:
        return
    if not _COLORS_ENABLED:
        return
    colored = user_input.replace(cmd, cyan(cmd), 1)
    print(f"\033[A\033[2K{bold(green('you> '))}{colored}")


_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class _Spinner:
    """Braille spinner for CLI thinking indicator."""

    def __init__(self) -> None:
        self._spinning = False
        self._thread: threading.Thread | None = None

    @property
    def spinning(self) -> bool:
        return self._spinning

    def start(self) -> None:
        self._spinning = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        i = 0
        while self._spinning:
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            sys.stdout.write(f"\r  {frame} {dim('thinking...')}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)

    def stop(self) -> None:
        self._spinning = False
        if self._thread:
            self._thread.join(timeout=0.2)
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()


_LOGO = r"""
  _
 | |_ __ _ _ __ ___
 | __/ _` | '__/ __|
 | || (_| | |  \__ \
  \__\__,_|_|  |___/
"""


def _welcome(config) -> None:
    print(cyan(_LOGO.rstrip()))
    print(dim(f"  [{config.primary_provider}:{config.primary_model}] ctrl-d to quit"))
    print(dim("  ? for shortcuts, /help for commands"))


def _apply_review(result: str) -> None:
    """Parse rules from /review output and apply with user approval."""
    rules = [
        line.strip()[2:].strip()
        for line in result.splitlines()
        if line.strip().startswith("- ")
    ]
    if not rules:
        return

    try:
        answer = input("  apply? (y/n) ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if answer != "y":
        print("  skipped")
        return

    p = _memory_file("procedural")
    if p is None:
        print("  no memory dir configured")
        return
    for rule in rules:
        _append_to_file(p, rule)
    archive_feedback()
    print(f"  {len(rules)} rules added to Procedural.md")
    try:
        build_index()
        print("  index updated")
    except Exception as e:
        print(f"  {yellow('[warning]')} reindex failed: {e}", file=sys.stderr)


def _apply_tidy(result: str) -> None:
    """Parse removals from /tidy output and apply with user approval."""
    removals: list[tuple[str, str]] = []
    for line in result.splitlines():
        stripped = line.strip()
        if stripped.startswith("[semantic] "):
            removals.append(("semantic", stripped[11:].strip()))
        elif stripped.startswith("[procedural] "):
            removals.append(("procedural", stripped[13:].strip()))
    if not removals:
        return

    try:
        answer = input("  apply? (y/n) ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if answer != "y":
        print("  skipped")
        return

    removed = 0
    for section, content in removals:
        p = _memory_file(section)
        if p is None or not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        target = f"- {content}\n"
        if target in text:
            text = text.replace(target, "", 1)
            p.write_text(text, encoding="utf-8", errors="replace")
            removed += 1
    print(f"  {removed} entries removed")


def repl(config):
    conv = Conversation(
        id="repl",
        provider=config.primary_provider,
        model=config.primary_model,
        remote_provider=config.remote_provider,
        remote_model=config.remote_model,
        routing_policy=config.routing_policy,
        channel="cli",
    )
    session_file = _session_path(channel="cli")
    history_file = Path.home() / ".tars_history"
    try:
        readline.read_history_file(history_file)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)
    readline.set_completer(_completer)
    readline.set_completer_delims("")
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    _welcome(config)
    try:
        while True:
            try:
                user_input = input(bold(green("you> ")))
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input.strip():
                continue
            _recolor_input(user_input)

            stripped = user_input.strip()
            ctx = {"channel": "cli", "config": config}
            cmd = stripped.split()[0] if stripped else ""

            # Commands that need a spinner (slow I/O or model calls)
            if cmd in ("/capture", "/review", "/tidy", "/brief"):
                spinner = _Spinner()
                spinner.start()
                result = dispatch(
                    user_input, config.primary_provider, config.primary_model,
                    conv=conv, context=ctx,
                )
                spinner.stop()
                if result is not None:
                    for line in result.splitlines():
                        print(f"  {line}")
                    if cmd == "/review":
                        _apply_review(result)
                    elif cmd == "/tidy":
                        _apply_tidy(result)
                continue

            result = dispatch(
                user_input, config.primary_provider, config.primary_model,
                conv=conv, context=ctx,
            )
            if result is not None:
                if result == "__clear__":
                    try:
                        save_session(conv, session_file)
                    except Exception:
                        pass
                    conv.messages.clear()
                    conv.msg_count = 0
                    conv.last_compaction = 0
                    conv.last_compaction_index = 0
                    conv.cumulative_summary = ""
                    conv.search_context = ""
                    session_file = _session_path(channel="cli")
                    print("  conversation cleared")
                else:
                    for line in result.splitlines():
                        print(f"  {line}")
                continue

            # Chat — stream response
            spinner = _Spinner()
            spinner.start()
            got_output = False
            for delta in process_message_stream(conv, user_input, session_file):
                if spinner.spinning:
                    spinner.stop()
                if not got_output:
                    sys.stdout.write(bold(cyan("tars> ")))
                    got_output = True
                sys.stdout.write(delta)
                sys.stdout.flush()
                time.sleep(0.016)
            if got_output:
                print()  # final newline
            elif spinner.spinning:
                spinner.stop()
    finally:
        try:
            readline.write_history_file(history_file)
        except OSError:
            pass
        try:
            save_session(conv, session_file)
        except KeyboardInterrupt:
            pass


def _run_index(embedding_model: str) -> None:
    """Index memory files and print stats."""
    try:
        stats = build_index(model=embedding_model)
    except RuntimeError as e:
        print(f"  {yellow('[warning]')} index update failed ({type(e).__name__}): {e}", file=sys.stderr)
        return
    print(
        f"index: {stats['indexed']} indexed, "
        f"{stats['skipped']} skipped, "
        f"{stats['deleted']} deleted, "
        f"{stats['chunks']} chunks"
    )


def _run_notes_index(embedding_model: str) -> None:
    """Index personal vault files and print stats."""
    try:
        stats = build_notes_index(model=embedding_model)
    except RuntimeError as e:
        print(f"  {yellow('[warning]')} notes index failed ({type(e).__name__}): {e}", file=sys.stderr)
        return
    print(
        f"notes-index: {stats['indexed']} indexed, "
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
            f"  {yellow('[warning]')} index update failed ({type(e).__name__}): {e}",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(prog="tars", description="tars AI assistant")
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="provider:model (e.g. ollama:gemma3:12b, claude:sonnet)",
    )
    parser.add_argument(
        "--remote-model",
        default=None,
        help="provider:model for escalation (e.g. claude:claude-sonnet-4-5-20250929)",
    )
    sub = parser.add_subparsers(dest="command")
    idx = sub.add_parser("index", help="rebuild memory search index")
    idx.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="ollama embedding model to use",
    )
    nidx = sub.add_parser("notes-index", help="rebuild personal notes/vault search index")
    nidx.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="ollama embedding model to use",
    )
    for cmd, mode in [("search", "hybrid"), ("sgrep", "fts"), ("svec", "vec")]:
        sp = sub.add_parser(cmd, help=f"{mode} search over memory index")
        sp.add_argument("query", nargs="+", help="search query")
        sp.add_argument("-n", "--limit", type=int, default=10, help="max results")
    srv = sub.add_parser("serve", help="start HTTP API server")
    srv.add_argument("--host", default="127.0.0.1", help="bind address")
    srv.add_argument("--port", type=int, default=8180, help="port number")
    sub.add_parser("email", help="start email polling channel")
    sub.add_parser("email-brief", help="send the daily brief via email")
    sub.add_parser("telegram", help="start Telegram bot channel")
    sub.add_parser("telegram-brief", help="send the daily brief via Telegram")

    sched = sub.add_parser("schedule", help="manage scheduled commands")
    sched_sub = sched.add_subparsers(dest="schedule_command")
    sched_add = sched_sub.add_parser("add", help="add a scheduled command")
    sched_add.add_argument("schedule_name", help="schedule name (e.g. email-brief)")
    sched_add.add_argument("schedule_cmd", help="tars subcommand to run")
    sched_add.add_argument("schedule_args", nargs="*", help="subcommand arguments")
    sched_add.add_argument("--hour", type=int, default=8)
    sched_add.add_argument("--minute", type=int, default=0)
    sched_add.add_argument("--watch", help="watch a directory instead of using a timer")
    sched_sub.add_parser("list", help="show installed schedules")
    sched_rm = sched_sub.add_parser("remove", help="remove a scheduled command")
    sched_rm.add_argument("schedule_name", help="schedule name to remove")
    sched_test = sched_sub.add_parser("test", help="test-run a schedule in its OS environment")
    sched_test.add_argument("schedule_name", help="schedule name to test")

    # Detect one-shot messages before argparse sees them — argparse subparsers
    # greedily match the first positional arg as a subcommand, so
    # `tars "hello"` fails with "invalid choice".  If argv[1] isn't a known
    # subcommand or flag, treat everything after flags as a message.
    _subcommands = {"index", "notes-index", "search", "sgrep", "svec", "serve", "email", "email-brief", "telegram", "telegram-brief", "schedule"}
    raw_args = sys.argv[1:]
    message_args: list[str] = []
    # Skip leading flags (-m, --model, --remote-model and their values)
    i = 0
    while i < len(raw_args) and raw_args[i].startswith("-"):
        i += 1  # flag
        if raw_args[i - 1] in ("-m", "--model", "--remote-model") and i < len(raw_args):
            i += 1  # flag value
    if i < len(raw_args) and raw_args[i] not in _subcommands:
        message_args = raw_args[i:]
        raw_args = raw_args[:i]

    args = parser.parse_args(raw_args)
    args.message = message_args

    if args.command == "index":
        _run_index(args.embedding_model)
        return

    if args.command == "notes-index":
        _run_notes_index(args.embedding_model)
        return

    if args.command in ("search", "sgrep", "svec"):
        mode = {"search": "hybrid", "sgrep": "fts", "svec": "vec"}[args.command]
        query = " ".join(args.query)
        try:
            results = search(query, mode=mode, limit=args.limit)
            _print_search_results(results, mode)
        except Exception as e:
            print(f"  {red('[error]')} search failed: {e}", file=sys.stderr)
        return

    if args.command == "serve":
        import uvicorn

        if not os.environ.get("TARS_API_TOKEN", ""):
            if args.host not in ("127.0.0.1", "::1", "localhost"):
                logging.getLogger("tars").error(
                    "TARS_API_TOKEN is not set and server is binding to %s "
                    "— all endpoints are publicly accessible", args.host
                )
        uvicorn.run("tars.api:app", host=args.host, port=args.port)
        return

    config = apply_cli_overrides(load_model_config(), args.model, args.remote_model)
    provider = config.primary_provider
    model = config.primary_model

    if args.command == "email":
        from tars.email import run_email

        run_email(config)
        return

    if args.command == "email-brief":
        from tars.email import send_brief_email

        try:
            send_brief_email()
        except Exception as e:
            print(f"  {red('[error]')} email brief failed: {e}", file=sys.stderr)
        return

    if args.command == "telegram":
        from tars.telegram import run_telegram

        run_telegram(config)
        return

    if args.command == "telegram-brief":
        from tars.telegram import send_brief_telegram_sync

        try:
            send_brief_telegram_sync()
        except Exception as e:
            print(f"  {red('[error]')} telegram brief failed: {e}", file=sys.stderr)
        return

    if args.command == "schedule":
        from tars.scheduler import ScheduleEntry, schedule_add, schedule_list, schedule_remove, schedule_test

        if args.schedule_command == "add":
            entry = ScheduleEntry(
                name=args.schedule_name,
                command=args.schedule_cmd,
                args=args.schedule_args or [],
                hour=args.hour if not args.watch else None,
                minute=args.minute if not args.watch else None,
                watch_path=args.watch,
            )
            print(schedule_add(entry))
        elif args.schedule_command == "list":
            _print_schedule_list(schedule_list())
        elif args.schedule_command == "remove":
            print(schedule_remove(args.schedule_name))
        elif args.schedule_command == "test":
            print(schedule_test(args.schedule_name))
        else:
            print("usage: tars schedule {add,list,remove,test}")
        return

    _startup_index()

    from tars.mcp import MCPClient, _load_mcp_config
    from tars.tools import set_mcp_client

    mcp_client = None
    mcp_config = _load_mcp_config()
    if mcp_config:
        mcp_client = MCPClient(mcp_config)
        mcp_client.start()
        set_mcp_client(mcp_client)
        from tars.router import update_tool_names
        update_tool_names({t["name"] for t in mcp_client.discover_tools()})

    try:
        if args.message:
            message = " ".join(args.message)
            conv = Conversation(
                id="oneshot",
                provider=provider,
                model=model,
                remote_provider=config.remote_provider,
                remote_model=config.remote_model,
                routing_policy=config.routing_policy,
                channel="cli",
            )
            conv.search_context = "[one-shot message, no follow-up possible — act immediately on any tool requests]"
            session_file = _session_path(channel="cli")
            reply = process_message(conv, message, session_file)
            print(reply)
        else:
            repl(config)
    finally:
        if mcp_client:
            mcp_client.stop()
            set_mcp_client(None)


def main_serve():
    """Convenience entrypoint for `tars-serve`."""
    import uvicorn

    sys.argv = ["tars", "serve"] + sys.argv[1:]
    main()


def main_email():
    """Convenience entrypoint for `tars-email`."""
    sys.argv = ["tars", "email"] + sys.argv[1:]
    main()


def main_index():
    """Convenience entrypoint for `tars-index`."""
    sys.argv = ["tars", "index"] + sys.argv[1:]
    main()


def main_telegram():
    """Convenience entrypoint for `tars-telegram`."""
    sys.argv = ["tars", "telegram"] + sys.argv[1:]
    main()


if __name__ == "__main__":
    main()
