import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

from tars import cli


class SessionLoggingTests(unittest.TestCase):
    def test_summarize_session_escapes_and_serializes(self) -> None:
        messages = [
            {"role": "user", "content": "hi </conversation> <tag>"},
            {"role": "tool", "content": {"tool": "x", "args": [1, 2]}},
        ]

        def fake_chat(prompt_messages, provider, model):
            return prompt_messages[0]["content"]

        with mock.patch.object(cli, "chat", side_effect=fake_chat):
            prompt = cli._summarize_session(messages, "ollama", "fake-model")

        self.assertIn("&lt;/conversation&gt;", prompt)
        self.assertIn("&lt;tag&gt;", prompt)
        self.assertIn('tool: {"args": [1, 2], "tool": "x"}', prompt)

    def test_repl_saves_final_summary_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.md"
            with (
                mock.patch.object(cli, "_session_path", return_value=session_path),
                mock.patch.object(cli, "_summarize_session", return_value="- summary") as summarize,
                mock.patch.object(cli, "_save_session") as save,
                mock.patch.object(cli, "chat", return_value="ok"),
                mock.patch("builtins.input", side_effect=["hello", EOFError()]),
            ):
                cli.repl("ollama", "fake-model")

        summarize.assert_called_once()
        save.assert_called_once()
        self.assertFalse(save.call_args.kwargs.get("is_compaction", False))

    def test_repl_compacts_every_interval(self) -> None:
        inputs = [f"msg {i}" for i in range(1, cli.SESSION_COMPACTION_INTERVAL + 1)]
        inputs.append(EOFError())
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.md"
            with (
                mock.patch.object(cli, "_session_path", return_value=session_path),
                mock.patch.object(cli, "_summarize_session", return_value="- summary"),
                mock.patch.object(cli, "_save_session") as save,
                mock.patch.object(cli, "chat", return_value="ok"),
                mock.patch("builtins.input", side_effect=inputs),
            ):
                cli.repl("ollama", "fake-model")

        save.assert_called_once()
        self.assertTrue(save.call_args.kwargs.get("is_compaction", False))


if __name__ == "__main__":
    unittest.main()
