import argparse
import sys

from dotenv import load_dotenv

from tars.core import DEFAULT_MODEL, chat, parse_model
from tars.indexer import build_index
from tars.sessions import (
    SESSION_COMPACTION_INTERVAL,
    _save_session,
    _session_path,
    _summarize_session,
)

load_dotenv()


def repl(provider: str, model: str):
    messages = []
    session_file = _session_path()
    msg_count = 0
    last_compaction = 0
    last_compaction_message_index = 0
    cumulative_summary = ""
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
            messages.append({"role": "user", "content": user_input})
            reply = chat(messages, provider, model)
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
    stats = build_index(model=embedding_model)
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
        print(f"  [warning] index update failed: {e}", file=sys.stderr)


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
    parser.add_argument("message", nargs="*", help="message for single-shot mode")
    args = parser.parse_args()

    if args.command == "index":
        _run_index(args.embedding_model)
        return

    provider, model = parse_model(args.model)

    _startup_index()

    if args.message:
        message = " ".join(args.message)
        messages = [{"role": "user", "content": message}]
        reply = chat(messages, provider, model)
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
