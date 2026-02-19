import argparse
import readline
import sys
from pathlib import Path

from dotenv import load_dotenv

from tars.core import DEFAULT_MODEL, _search_relevant_context, chat, parse_model
from tars.indexer import build_index
from tars.search import search
from tars.sessions import (
    SESSION_COMPACTION_INTERVAL,
    _save_session,
    _session_path,
    _summarize_session,
)

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


def repl(provider: str, model: str):
    messages = []
    session_file = _session_path()
    msg_count = 0
    last_compaction = 0
    last_compaction_message_index = 0
    cumulative_summary = ""
    search_context = ""
    history_file = Path.home() / ".tars_history"
    try:
        readline.read_history_file(history_file)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)
    print(f"tars [{provider}:{model}] (ctrl-d to quit)")
    def _merge_summary(existing: str, new: str) -> str:
        if not existing:
            return new
        if not new:
            return existing
        return f"{existing.rstrip()}\n{new.lstrip()}"
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
                print("  /search <query>  hybrid keyword + semantic search")
                print("  /sgrep <query>   keyword search (FTS5/BM25)")
                print("  /svec <query>    semantic search (vector KNN)")
                print("  /help            show this help")
                continue
            if _handle_slash_search(user_input):
                continue
            # Search on first message only
            if not messages and not search_context:
                try:
                    search_context = _search_relevant_context(user_input)
                except Exception as e:
                    print(f"  [warning] startup search failed: {e}", file=sys.stderr)
            messages.append({"role": "user", "content": user_input})
            reply = chat(messages, provider, model, search_context=search_context)
            messages.append({"role": "assistant", "content": reply})
            print(f"tars> {reply}")
            msg_count += 1
            if session_file and msg_count - last_compaction >= SESSION_COMPACTION_INTERVAL:
                try:
                    new_messages = messages[last_compaction_message_index:]
                    summary = _summarize_session(
                        new_messages, provider, model, previous_summary=cumulative_summary,
                    )
                    cumulative_summary = _merge_summary(cumulative_summary, summary)
                    _save_session(session_file, cumulative_summary, is_compaction=True)
                    last_compaction = msg_count
                    last_compaction_message_index = len(messages)
                except Exception as e:
                    print(f"  [warning] session compaction failed: {e}", file=sys.stderr)
    finally:
        try:
            readline.write_history_file(history_file)
        except OSError:
            pass
        # Save final session on exit
        if session_file and messages and msg_count > last_compaction:
            try:
                new_messages = messages[last_compaction_message_index:]
                summary = _summarize_session(
                    new_messages, provider, model, previous_summary=cumulative_summary,
                )
                cumulative_summary = _merge_summary(cumulative_summary, summary)
                _save_session(session_file, cumulative_summary)
            except Exception as e:
                print(f"  [warning] session save failed: {e}", file=sys.stderr)


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

    provider, model = parse_model(args.model)

    _startup_index()

    if args.message:
        message = " ".join(args.message)
        search_context = ""
        try:
            search_context = _search_relevant_context(message)
        except Exception as e:
            print(f"  [warning] startup search failed: {e}", file=sys.stderr)
        messages = [{"role": "user", "content": message}]
        reply = chat(messages, provider, model, search_context=search_context)
        print(reply)
        session_file = _session_path()
        if session_file:
            try:
                messages.append({"role": "assistant", "content": reply})
                summary = _summarize_session(messages, provider, model)
                _save_session(session_file, summary)
            except Exception as e:
                print(f"  [warning] session save failed: {e}", file=sys.stderr)
    else:
        repl(provider, model)


if __name__ == "__main__":
    main()
