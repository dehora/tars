import sys
import tempfile
import unittest
from datetime import datetime, timedelta
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


class ContextRollupTests(unittest.TestCase):
    def test_rollup_creates_today_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            today = datetime.now().strftime("%Y-%m-%d")
            (sessions_dir / f"{today}T10-00-00.md").write_text("# Session\n- talked about weather")

            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(cli, "chat", return_value="- discussed weather"),
            ):
                cli._rollup_context("ollama", "fake-model")

            today_path = Path(tmpdir) / "context" / "today.md"
            self.assertTrue(today_path.exists())
            text = today_path.read_text()
            self.assertIn(f"<!-- tars:date {today} -->", text)
            self.assertIn("- discussed weather", text)

    def test_rollup_rotates_yesterday(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            (sessions_dir / f"{today}T10-00-00.md").write_text("# Session\n- new stuff")

            context_dir = Path(tmpdir) / "context"
            context_dir.mkdir()
            (context_dir / "today.md").write_text(
                f"<!-- tars:date {yesterday} -->\n# Context {yesterday}\n\n- old stuff\n"
            )

            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(cli, "chat", return_value="- new summary"),
            ):
                cli._rollup_context("ollama", "fake-model")

            yesterday_path = context_dir / "yesterday.md"
            self.assertTrue(yesterday_path.exists())
            self.assertIn(f"<!-- tars:date {yesterday} -->", yesterday_path.read_text())
            today_path = context_dir / "today.md"
            self.assertIn(f"<!-- tars:date {today} -->", today_path.read_text())

    def test_rollup_deletes_stale_yesterday(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            today = datetime.now().strftime("%Y-%m-%d")
            old_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            (sessions_dir / f"{today}T10-00-00.md").write_text("# Session\n- stuff")

            context_dir = Path(tmpdir) / "context"
            context_dir.mkdir()
            (context_dir / "yesterday.md").write_text(
                f"<!-- tars:date {old_date} -->\n# Context {old_date}\n\n- stale\n"
            )

            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(cli, "chat", return_value="- summary"),
            ):
                cli._rollup_context("ollama", "fake-model")

            self.assertFalse((context_dir / "yesterday.md").exists())

    def test_rollup_skips_without_memory_dir(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            # Should not raise
            cli._rollup_context("ollama", "fake-model")

    def test_rollup_skips_without_today_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(cli, "chat") as mock_chat,
            ):
                cli._rollup_context("ollama", "fake-model")
            mock_chat.assert_not_called()

    def test_load_context_returns_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context_dir = Path(tmpdir) / "context"
            context_dir.mkdir()
            (context_dir / "today.md").write_text("today content")
            (context_dir / "yesterday.md").write_text("yesterday content")

            with mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}):
                result = cli._load_context()

            self.assertIn("today content", result)
            self.assertIn("yesterday content", result)

    def test_load_context_empty_without_dir(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(cli._load_context(), "")

    def test_build_system_prompt_includes_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context_dir = Path(tmpdir) / "context"
            context_dir.mkdir()
            (context_dir / "today.md").write_text("today rollup")

            with mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}):
                prompt = cli._build_system_prompt()

            self.assertIn("<context>", prompt)
            self.assertIn("today rollup", prompt)


if __name__ == "__main__":
    unittest.main()
