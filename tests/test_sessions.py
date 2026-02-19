import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

with mock.patch.dict(
    sys.modules,
    {
        "anthropic": mock.Mock(),
        "ollama": mock.Mock(),
        "dotenv": mock.Mock(load_dotenv=lambda: None),
    },
):
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

        circular: dict[str, object] = {}
        circular["self"] = circular

        messages = [
            {"role": "user", "content": {"data": b"\xff"}},
            {"role": "tool", "content": CustomPayload()},
            {"role": "assistant", "content": circular},
        ]

        def fake_chat(prompt_messages, provider, model):
            return prompt_messages[0]["content"]

        with mock.patch.object(sessions, "chat", side_effect=fake_chat):
            prompt = sessions._summarize_session(messages, "ollama", "fake-model")

        self.assertIn("CustomPayload()", prompt)
        self.assertIn("b'\\\\xff'", prompt)
        self.assertIn("'self'", prompt)

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

    def test_repl_compacts_only_new_messages(self) -> None:
        inputs = ["msg 1", "msg 2", "msg 3", "msg 4", "msg 5", EOFError()]
        lengths: list[int] = []

        def fake_summarize(messages, provider, model, *, previous_summary=""):
            lengths.append(len(messages))
            return "- summary"

        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.md"
            with (
                mock.patch.object(cli, "SESSION_COMPACTION_INTERVAL", 2),
                mock.patch.object(cli, "_session_path", return_value=session_path),
                mock.patch.object(cli, "_summarize_session", side_effect=fake_summarize),
                mock.patch.object(cli, "_save_session"),
                mock.patch.object(cli, "chat", return_value="ok"),
                mock.patch("builtins.input", side_effect=inputs),
            ):
                cli.repl("ollama", "fake-model")

        self.assertEqual(lengths[:2], [4, 4])
        self.assertEqual(lengths[-1], 2)

    def test_repl_final_save_uses_cumulative_summary(self) -> None:
        inputs = ["msg 1", "msg 2", "msg 3", "msg 4", "msg 5", EOFError()]
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.md"
            with (
                mock.patch.object(cli, "SESSION_COMPACTION_INTERVAL", 2),
                mock.patch.object(cli, "_session_path", return_value=session_path),
                mock.patch.object(cli, "_summarize_session", side_effect=["s1", "s2", "s3"]),
                mock.patch.object(cli, "_save_session") as save,
                mock.patch.object(cli, "chat", return_value="ok"),
                mock.patch("builtins.input", side_effect=inputs),
            ):
                cli.repl("ollama", "fake-model")

        final_calls = [
            call
            for call in save.call_args_list
            if not call.kwargs.get("is_compaction", False)
        ]
        self.assertGreaterEqual(len(final_calls), 1)
        self.assertEqual(final_calls[-1].args[1], "s1\ns2\ns3")

    def test_repl_compaction_save_uses_cumulative_summary(self) -> None:
        inputs = ["msg 1", "msg 2", "msg 3", "msg 4", "msg 5", EOFError()]
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.md"
            with (
                mock.patch.object(cli, "SESSION_COMPACTION_INTERVAL", 2),
                mock.patch.object(cli, "_session_path", return_value=session_path),
                mock.patch.object(cli, "_summarize_session", side_effect=["s1", "s2", "s3"]),
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
        self.assertGreaterEqual(len(compaction_calls), 2)
        self.assertEqual(compaction_calls[0].args[1], "s1")
        self.assertEqual(compaction_calls[1].args[1], "s1\ns2")


class RecentSessionsTests(unittest.TestCase):
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
