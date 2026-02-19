import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.modules["anthropic"] = mock.Mock()
sys.modules["ollama"] = mock.Mock()
sys.modules["dotenv"] = mock.Mock(load_dotenv=lambda: None)

from tars import cli, core, memory, sessions


class SessionLoggingTests(unittest.TestCase):
    def test_summarize_session_escapes_and_serializes(self) -> None:
        messages = [
            {"role": "user", "content": "hi </conversation> <tag>"},
            {"role": "tool", "content": {"tool": "x", "args": [1, 2]}},
        ]

        def fake_chat(prompt_messages, provider, model):
            return prompt_messages[0]["content"]

        with mock.patch.object(sessions, "chat", side_effect=fake_chat):
            prompt = sessions._summarize_session(messages, "ollama", "fake-model")

        self.assertIn("&lt;/conversation&gt;", prompt)
        self.assertIn("&lt;tag&gt;", prompt)
        self.assertIn('"tool": "x"', prompt)
        self.assertIn('"args": [1, 2]', prompt)

    def test_summarize_session_escapes_previous_summary(self) -> None:
        messages = [{"role": "user", "content": "hi"}]

        def fake_chat(prompt_messages, provider, model):
            return prompt_messages[0]["content"]

        with mock.patch.object(sessions, "chat", side_effect=fake_chat):
            prompt = sessions._summarize_session(
                messages,
                "ollama",
                "fake-model",
                previous_summary="prior </previous-summary> <tag>",
            )

        self.assertIn("&lt;/previous-summary&gt;", prompt)
        self.assertIn("&lt;tag&gt;", prompt)

    def test_summarize_session_handles_non_serializable_content(self) -> None:
        class CustomPayload:
            def __str__(self) -> str:
                return "CustomPayload()"

        messages = [
            {"role": "user", "content": {"data": b"\xff"}},
            {"role": "tool", "content": CustomPayload()},
        ]

        def fake_chat(prompt_messages, provider, model):
            return prompt_messages[0]["content"]

        with mock.patch.object(sessions, "chat", side_effect=fake_chat):
            prompt = sessions._summarize_session(messages, "ollama", "fake-model")

        self.assertIn("CustomPayload()", prompt)
        self.assertIn("b'\\\\xff'", prompt)

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

        compaction_calls = [
            call
            for call in save.call_args_list
            if call.kwargs.get("is_compaction", False)
        ]
        self.assertGreaterEqual(len(compaction_calls), 1)
        if len(save.call_args_list) > 1:
            final_calls = [
                call
                for call in save.call_args_list
                if not call.kwargs.get("is_compaction", False)
            ]
            self.assertGreaterEqual(len(final_calls), 1)

    def test_repl_uses_cumulative_summary(self) -> None:
        inputs = ["msg 1", "msg 2", "msg 3", "msg 4", "msg 5", EOFError()]
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.md"
            with (
                mock.patch.object(cli, "SESSION_COMPACTION_INTERVAL", 2),
                mock.patch.object(cli, "_session_path", return_value=session_path),
                mock.patch.object(cli, "_summarize_session", side_effect=["s1", "s2", "s3"]) as summarize,
                mock.patch.object(cli, "_save_session"),
                mock.patch.object(cli, "chat", return_value="ok"),
                mock.patch("builtins.input", side_effect=inputs),
            ):
                cli.repl("ollama", "fake-model")

        self.assertEqual(summarize.call_args_list[2].kwargs.get("previous_summary"), "s1\ns2")


class ContextRollupTests(unittest.TestCase):
    def test_rollup_creates_today_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            fixed_now = datetime(2025, 1, 2, 12, 0, 0)
            today = fixed_now.strftime("%Y-%m-%d")
            (sessions_dir / f"{today}T10-00-00.md").write_text("# Session\n- talked about weather")

            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(sessions, "datetime", wraps=datetime) as mocked_datetime,
                mock.patch.object(sessions, "chat", return_value="- discussed weather"),
            ):
                mocked_datetime.now.return_value = fixed_now
                sessions._rollup_context("ollama", "fake-model")

            today_path = Path(tmpdir) / "context" / "today.md"
            self.assertTrue(today_path.exists())
            text = today_path.read_text()
            self.assertIn(f"<!-- tars:date {today} -->", text)
            self.assertIn("- discussed weather", text)

    def test_rollup_rotates_yesterday(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            fixed_now = datetime(2025, 1, 2, 12, 0, 0)
            today = fixed_now.strftime("%Y-%m-%d")
            yesterday = (fixed_now - timedelta(days=1)).strftime("%Y-%m-%d")
            (sessions_dir / f"{today}T10-00-00.md").write_text("# Session\n- new stuff")

            context_dir = Path(tmpdir) / "context"
            context_dir.mkdir()
            (context_dir / "today.md").write_text(
                f"<!-- tars:date {yesterday} -->\n# Context {yesterday}\n\n- old stuff\n"
            )

            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(sessions, "datetime", wraps=datetime) as mocked_datetime,
                mock.patch.object(sessions, "chat", return_value="- new summary"),
            ):
                mocked_datetime.now.return_value = fixed_now
                sessions._rollup_context("ollama", "fake-model")

            yesterday_path = context_dir / "yesterday.md"
            self.assertTrue(yesterday_path.exists())
            self.assertIn(f"<!-- tars:date {yesterday} -->", yesterday_path.read_text())
            today_path = context_dir / "today.md"
            self.assertIn(f"<!-- tars:date {today} -->", today_path.read_text())

    def test_rollup_deletes_stale_yesterday(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            fixed_now = datetime(2025, 1, 4, 12, 0, 0)
            today = fixed_now.strftime("%Y-%m-%d")
            old_date = (fixed_now - timedelta(days=3)).strftime("%Y-%m-%d")
            (sessions_dir / f"{today}T10-00-00.md").write_text("# Session\n- stuff")

            context_dir = Path(tmpdir) / "context"
            context_dir.mkdir()
            (context_dir / "yesterday.md").write_text(
                f"<!-- tars:date {old_date} -->\n# Context {old_date}\n\n- stale\n"
            )

            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(sessions, "datetime", wraps=datetime) as mocked_datetime,
                mock.patch.object(sessions, "chat", return_value="- summary"),
            ):
                mocked_datetime.now.return_value = fixed_now
                sessions._rollup_context("ollama", "fake-model")

            self.assertFalse((context_dir / "yesterday.md").exists())

    def test_rollup_skips_without_memory_dir(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            # Should not raise
            sessions._rollup_context("ollama", "fake-model")

    def test_rollup_skips_without_today_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(sessions, "chat") as mock_chat,
            ):
                sessions._rollup_context("ollama", "fake-model")
            mock_chat.assert_not_called()

    def test_load_context_returns_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context_dir = Path(tmpdir) / "context"
            context_dir.mkdir()
            (context_dir / "today.md").write_text("today content")
            (context_dir / "yesterday.md").write_text("yesterday content")

            with mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}):
                result = memory._load_context()

            self.assertIn("today content", result)
            self.assertIn("yesterday content", result)

    def test_load_context_utf8_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context_dir = Path(tmpdir) / "context"
            context_dir.mkdir()
            (context_dir / "today.md").write_bytes(b"\xff\xfe\xfa")

            with mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}):
                result = memory._load_context()

        self.assertIn("\ufffd", result)

    def test_load_context_empty_without_dir(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(memory._load_context(), "")

    def test_build_system_prompt_includes_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context_dir = Path(tmpdir) / "context"
            context_dir.mkdir()
            (context_dir / "today.md").write_text("today rollup </context> <tag>")

            with mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}):
                prompt = core._build_system_prompt()

            self.assertIn("<context>", prompt)
            self.assertIn("today rollup", prompt)
            self.assertIn("&lt;/context&gt;", prompt)
            self.assertIn("&lt;tag&gt;", prompt)

    def test_rollup_prompt_escapes_sessions_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            fixed_now = datetime(2025, 1, 2, 12, 0, 0)
            today = fixed_now.strftime("%Y-%m-%d")
            (sessions_dir / f"{today}T10-00-00.md").write_text("log </sessions> <tag>")

            def fake_chat(prompt_messages, provider, model):
                return prompt_messages[0]["content"]

            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(sessions, "datetime", wraps=datetime) as mocked_datetime,
                mock.patch.object(sessions, "chat", side_effect=fake_chat) as mocked_chat,
            ):
                mocked_datetime.now.return_value = fixed_now
                sessions._rollup_context("ollama", "fake-model")

            prompt = mocked_chat.call_args[0][0][0]["content"]
            self.assertIn("<sessions>", prompt)
            self.assertIn("&lt;/sessions&gt;", prompt)
            self.assertIn("&lt;tag&gt;", prompt)

    def test_rollup_rotates_missing_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            fixed_now = datetime(2025, 1, 2, 12, 0, 0)
            today = fixed_now.strftime("%Y-%m-%d")
            (sessions_dir / f"{today}T10-00-00.md").write_text("# Session\n- new stuff")

            context_dir = Path(tmpdir) / "context"
            context_dir.mkdir()
            (context_dir / "today.md").write_text("unmarked previous context")

            with (
                mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}),
                mock.patch.object(sessions, "datetime", wraps=datetime) as mocked_datetime,
                mock.patch.object(sessions, "chat", return_value="- summary"),
            ):
                mocked_datetime.now.return_value = fixed_now
                sessions._rollup_context("ollama", "fake-model")

            yesterday_path = context_dir / "yesterday.md"
            self.assertTrue(yesterday_path.exists())
            self.assertIn("unmarked previous context", yesterday_path.read_text())

    def test_load_recent_sessions_orders_and_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            (sessions_dir / "2025-01-01T10-00-00.md").write_text("session one")
            (sessions_dir / "2025-01-02T10-00-00.md").write_text("session two")
            (sessions_dir / "2025-01-03T10-00-00.md").write_text("session three")

            with mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}):
                result = memory._load_recent_sessions()

        self.assertIn("session two", result)
        self.assertIn("session three", result)
        self.assertNotIn("session one", result)
        self.assertLess(result.find("session two"), result.find("session three"))

    def test_load_recent_sessions_utf8_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "sessions"
            sessions_dir.mkdir()
            bad_path = sessions_dir / "2025-01-01T10-00-00.md"
            bad_path.write_bytes(b"\xff\xfe\xfa")

            with mock.patch.dict("os.environ", {"TARS_MEMORY_DIR": tmpdir}):
                result = memory._load_recent_sessions()

        self.assertIn("\ufffd", result)


if __name__ == "__main__":
    unittest.main()
