import argparse
import sys

from dotenv import load_dotenv

from tars.core import DEFAULT_MODEL, chat, parse_model
from tars.sessions import (
    SESSION_COMPACTION_INTERVAL,
    _rollup_context,
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
    last_summary = ""
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
            messages.append({"role": "user", "content": user_input})
            reply = chat(messages, provider, model)
            messages.append({"role": "assistant", "content": reply})
            print(f"tars> {reply}")
            msg_count += 1
            if session_file and msg_count - last_compaction >= SESSION_COMPACTION_INTERVAL:
                try:
                    summary = _summarize_session(
                        messages, provider, model, previous_summary=last_summary,
                    )
                    _save_session(session_file, summary, is_compaction=True)
                    last_summary = summary
                    last_compaction = msg_count
                except Exception as e:
                    print(f"  [warning] session compaction failed: {e}", file=sys.stderr)
    finally:
        # Save final session on exit
        if session_file and messages and msg_count > last_compaction:
            try:
                summary = _summarize_session(
                    messages, provider, model, previous_summary=last_summary,
                )
                _save_session(session_file, summary)
            except Exception as e:
                print(f"  [warning] session save failed: {e}", file=sys.stderr)
        # Roll up today's sessions into context/today.md
        try:
            _rollup_context(provider, model)
        except Exception as e:
            print(f"  [warning] context rollup failed: {e}", file=sys.stderr)


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
        try:
            _rollup_context(provider, model)
        except Exception as e:
            print(f"  [warning] context rollup failed: {e}", file=sys.stderr)
    else:
        repl(provider, model)


if __name__ == "__main__":
    main()
