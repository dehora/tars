"""Tests for tars.scheduler module."""

import os
import plistlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tars.scheduler import (
    LABEL_PREFIX,
    ScheduleEntry,
    _capture_env,
    _generate_plist,
    _generate_systemd_path,
    _generate_systemd_service,
    _generate_systemd_timer,
    _is_linux,
    _is_macos,
    _load_dotenv_values,
    _read_last_log_line,
    _schedule_test_linux,
    schedule_list,
)


class TestScheduleEntry(unittest.TestCase):
    def test_defaults(self):
        e = ScheduleEntry(name="test", command="index")
        self.assertEqual(e.name, "test")
        self.assertEqual(e.command, "index")
        self.assertEqual(e.args, [])
        self.assertIsNone(e.hour)
        self.assertIsNone(e.minute)
        self.assertIsNone(e.watch_path)


class TestCaptureEnv(unittest.TestCase):
    @mock.patch("tars.scheduler._load_dotenv_values", return_value={})
    def test_picks_up_tars_vars(self, _mock_dotenv):
        with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": "/tmp/mem", "TARS_MAX_TOKENS": "2048"}, clear=False):
            env = _capture_env()
            self.assertEqual(env["TARS_MEMORY_DIR"], "/tmp/mem")
            self.assertEqual(env["TARS_MAX_TOKENS"], "2048")

    @mock.patch("tars.scheduler._load_dotenv_values", return_value={})
    def test_captures_model_vars(self, _mock_dotenv):
        env_vars = {
            "TARS_MODEL_DEFAULT": "ollama:qwen3.5:27b",
            "TARS_MODEL_REMOTE": "claude:claude-sonnet-4-5-20250929",
            "TARS_MODEL_EMBEDDING": "qwen3-embedding:8b",
            "TARS_MODEL_RETRIEVAL": "gemma3:4b",
        }
        with mock.patch.dict(os.environ, env_vars, clear=False):
            env = _capture_env()
        for key, val in env_vars.items():
            self.assertEqual(env[key], val, f"{key} not captured")

    @mock.patch("tars.scheduler._load_dotenv_values", return_value={})
    def test_ignores_unknown_vars(self, _mock_dotenv):
        with mock.patch.dict(os.environ, {"RANDOM_VAR": "nope"}, clear=False):
            env = _capture_env()
            self.assertNotIn("RANDOM_VAR", env)

    @mock.patch("tars.scheduler._load_dotenv_values", return_value={})
    def test_empty_values_excluded(self, _mock_dotenv):
        with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": ""}, clear=False):
            env = _capture_env()
            self.assertNotIn("TARS_MEMORY_DIR", env)

    def test_env_overlays_dotenv(self):
        """os.environ takes precedence over .env values."""
        with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": "/from/env"}, clear=False):
            with mock.patch("tars.scheduler._load_dotenv_values", return_value={"TARS_MEMORY_DIR": "/from/dotenv"}):
                env = _capture_env()
                self.assertEqual(env["TARS_MEMORY_DIR"], "/from/env")


class TestGeneratePlist(unittest.TestCase):
    def _make_entry(self, **kwargs):
        defaults = dict(name="test-brief", command="email-brief", args=[], hour=8, minute=30, watch_path=None)
        defaults.update(kwargs)
        return ScheduleEntry(**defaults)

    def test_calendar_interval(self):
        entry = self._make_entry()
        plist = _generate_plist(entry, {}, "/usr/bin/uv", Path("/repo"))
        self.assertEqual(plist["Label"], f"{LABEL_PREFIX}-test-brief")
        self.assertIn("StartCalendarInterval", plist)
        cal = plist["StartCalendarInterval"]
        self.assertEqual(cal["Hour"], 8)
        self.assertEqual(cal["Minute"], 30)
        self.assertNotIn("WatchPaths", plist)

    def test_watch_paths(self):
        entry = self._make_entry(hour=None, minute=None, watch_path="/vault")
        plist = _generate_plist(entry, {}, "/usr/bin/uv", Path("/repo"))
        self.assertIn("WatchPaths", plist)
        self.assertEqual(plist["WatchPaths"], ["/vault"])
        self.assertNotIn("StartCalendarInterval", plist)

    def test_includes_env(self):
        env = {"TARS_MEMORY_DIR": "/mem", "ANTHROPIC_API_KEY": "sk-test"}
        entry = self._make_entry()
        plist = _generate_plist(entry, env, "/usr/bin/uv", Path("/repo"))
        self.assertIn("EnvironmentVariables", plist)
        self.assertEqual(plist["EnvironmentVariables"]["TARS_MEMORY_DIR"], "/mem")
        self.assertEqual(plist["EnvironmentVariables"]["ANTHROPIC_API_KEY"], "sk-test")

    def test_no_env_omits_key(self):
        entry = self._make_entry()
        plist = _generate_plist(entry, {}, "/usr/bin/uv", Path("/repo"))
        self.assertNotIn("EnvironmentVariables", plist)

    def test_program_arguments(self):
        entry = self._make_entry(command="index", args=["--embedding-model", "qwen3"])
        plist = _generate_plist(entry, {}, "/home/user/.local/bin/uv", Path("/repo"))
        prog = plist["ProgramArguments"]
        self.assertEqual(prog, ["/home/user/.local/bin/uv", "run", "tars", "index", "--embedding-model", "qwen3"])

    def test_working_directory(self):
        entry = self._make_entry()
        plist = _generate_plist(entry, {}, "/usr/bin/uv", Path("/my/repo"))
        self.assertEqual(plist["WorkingDirectory"], "/my/repo")

    def test_log_paths(self):
        entry = self._make_entry()
        plist = _generate_plist(entry, {}, "/usr/bin/uv", Path("/repo"))
        self.assertIn("tars-test-brief.log", plist["StandardOutPath"])
        self.assertIn("tars-test-brief.err.log", plist["StandardErrorPath"])

    def test_plist_is_serializable(self):
        """Generated plist can round-trip through plistlib."""
        entry = self._make_entry()
        env = {"TARS_MEMORY_DIR": "/mem"}
        plist = _generate_plist(entry, env, "/usr/bin/uv", Path("/repo"))
        data = plistlib.dumps(plist)
        loaded = plistlib.loads(data)
        self.assertEqual(loaded["Label"], plist["Label"])


class TestGenerateSystemd(unittest.TestCase):
    def _make_entry(self, **kwargs):
        defaults = dict(name="test-brief", command="email-brief", args=[], hour=8, minute=30, watch_path=None)
        defaults.update(kwargs)
        return ScheduleEntry(**defaults)

    def test_service_content(self):
        entry = self._make_entry()
        env = {"TARS_MEMORY_DIR": "/mem"}
        content = _generate_systemd_service(entry, env, "/usr/bin/uv", Path("/repo"))
        self.assertIn("[Service]", content)
        self.assertIn("Type=oneshot", content)
        self.assertIn("WorkingDirectory=/repo", content)
        self.assertIn("uv run tars email-brief", content)
        self.assertIn('Environment="TARS_MEMORY_DIR=/mem"', content)

    def test_timer_content(self):
        entry = self._make_entry()
        content = _generate_systemd_timer(entry)
        self.assertIn("[Timer]", content)
        self.assertIn("OnCalendar=*-*-* 08:30:00", content)
        self.assertIn("Persistent=true", content)
        self.assertIn("[Install]", content)

    def test_path_unit(self):
        entry = self._make_entry(watch_path="/vault/notes")
        content = _generate_systemd_path(entry)
        self.assertIn("[Path]", content)
        self.assertIn("PathModified=/vault/notes", content)
        self.assertIn("[Install]", content)

    def test_service_with_args(self):
        entry = self._make_entry(command="index", args=["--embedding-model", "qwen3"])
        content = _generate_systemd_service(entry, {}, "/usr/bin/uv", Path("/repo"))
        self.assertIn("uv run tars index --embedding-model qwen3", content)


class TestReadLastLogLine(unittest.TestCase):
    def test_empty_path(self):
        self.assertEqual(_read_last_log_line(""), "")

    def test_nonexistent_file(self):
        result = _read_last_log_line("/nonexistent/path/log.txt")
        self.assertEqual(result, "(never)")

    def test_reads_last_line(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("line 1\nline 2\nlast line\n")
            f.flush()
            result = _read_last_log_line(f.name)
            self.assertEqual(result, "last line")
            os.unlink(f.name)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("")
            f.flush()
            result = _read_last_log_line(f.name)
            self.assertEqual(result, "(never)")
            os.unlink(f.name)


class TestBuildPath(unittest.TestCase):
    def test_preserves_existing_path(self):
        from tars.scheduler import _build_path
        original_path = "/custom/bin:/another/bin"
        with mock.patch.dict(os.environ, {"PATH": original_path}):
            result = _build_path()
            self.assertIn("/custom/bin", result)
            self.assertIn("/another/bin", result)

    def test_uses_fallback_when_no_path(self):
        from tars.scheduler import _build_path
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _build_path()
            self.assertIn("/usr/local/bin", result)
            self.assertIn("/usr/bin", result)


class TestScheduleListEmpty(unittest.TestCase):
    def test_empty_list_macos(self):
        with mock.patch("tars.scheduler._is_macos", return_value=True):
            with mock.patch("tars.scheduler._is_linux", return_value=False):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with mock.patch("tars.scheduler.Path.home", return_value=Path(tmpdir)):
                        result = schedule_list()
                        self.assertEqual(result, [])

    def test_empty_list_linux(self):
        with mock.patch("tars.scheduler._is_macos", return_value=False):
            with mock.patch("tars.scheduler._is_linux", return_value=True):
                with tempfile.TemporaryDirectory() as tmpdir:
                    with mock.patch("tars.scheduler._systemd_dir", return_value=Path(tmpdir)):
                        result = schedule_list()
                        self.assertEqual(result, [])


class TestScheduleListMacosDiscovery(unittest.TestCase):
    def test_discovers_installed_plists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            agents_dir = tmppath / "Library" / "LaunchAgents"
            agents_dir.mkdir(parents=True)

            # Write a test plist
            plist = {
                "Label": f"{LABEL_PREFIX}-test-brief",
                "ProgramArguments": ["/bin/bash", "-lc", "uv run tars email-brief"],
                "WorkingDirectory": "/repo",
                "StartCalendarInterval": {"Hour": 8, "Minute": 0},
                "StandardOutPath": str(tmppath / "test.log"),
                "StandardErrorPath": str(tmppath / "test.err.log"),
            }
            plist_path = agents_dir / f"{LABEL_PREFIX}-test-brief.plist"
            with open(plist_path, "wb") as f:
                plistlib.dump(plist, f)

            with mock.patch("tars.scheduler._is_macos", return_value=True):
                with mock.patch("tars.scheduler._is_linux", return_value=False):
                    with mock.patch("tars.scheduler.Path.home", return_value=tmppath):
                        result = schedule_list()
                        self.assertEqual(len(result), 1)
                        self.assertEqual(result[0]["name"], "test-brief")
                        self.assertEqual(result[0]["trigger"], "daily 08:00")


class TestPlatformDetection(unittest.TestCase):
    def test_macos_detection(self):
        with mock.patch("tars.scheduler.platform.system", return_value="Darwin"):
            self.assertTrue(_is_macos())
            self.assertFalse(_is_linux())

    def test_linux_detection(self):
        with mock.patch("tars.scheduler.platform.system", return_value="Linux"):
            self.assertFalse(_is_macos())
            self.assertTrue(_is_linux())


class TestSystemdEnvEscaping(unittest.TestCase):
    def _make_entry(self, **kwargs):
        defaults = dict(name="test", command="email-brief", args=[], hour=8, minute=0, watch_path=None)
        defaults.update(kwargs)
        return ScheduleEntry(**defaults)

    def test_values_are_quoted(self):
        entry = self._make_entry()
        env = {"KEY": "value"}
        content = _generate_systemd_service(entry, env, "/usr/bin/uv", Path("/repo"))
        self.assertIn('Environment="KEY=value"', content)

    def test_newlines_stripped(self):
        entry = self._make_entry()
        env = {"KEY": "line1\nline2\rline3"}
        content = _generate_systemd_service(entry, env, "/usr/bin/uv", Path("/repo"))
        self.assertNotIn("\nline2", content)
        self.assertIn("line1 line2 line3", content)

    def test_quotes_escaped(self):
        entry = self._make_entry()
        env = {"KEY": 'value with "quotes"'}
        content = _generate_systemd_service(entry, env, "/usr/bin/uv", Path("/repo"))
        self.assertIn('\\"quotes\\"', content)


class TestFilePermissions(unittest.TestCase):
    def _make_entry(self, **kwargs):
        defaults = dict(name="test-perms", command="email-brief", args=[], hour=8, minute=0, watch_path=None)
        defaults.update(kwargs)
        return ScheduleEntry(**defaults)

    def test_plist_chmod_600(self):
        import stat
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            agents_dir = tmppath / "Library" / "LaunchAgents"
            agents_dir.mkdir(parents=True)
            logs_dir = tmppath / "Library" / "Logs"
            logs_dir.mkdir(parents=True)

            entry = self._make_entry()
            env = {"TARS_MEMORY_DIR": "/mem"}

            with (
                mock.patch("tars.scheduler.Path.home", return_value=tmppath),
                mock.patch("tars.scheduler.subprocess.run"),
            ):
                from tars.scheduler import _schedule_add_macos
                _schedule_add_macos(entry, env, "/usr/bin/uv", Path("/repo"))

            plist_path = agents_dir / f"{LABEL_PREFIX}-test-perms.plist"
            self.assertTrue(plist_path.exists())
            mode = plist_path.stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)

    def test_systemd_service_chmod_600(self):
        import stat
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = self._make_entry()
            env = {"TARS_MEMORY_DIR": "/mem"}

            with (
                mock.patch("tars.scheduler._systemd_dir", return_value=Path(tmpdir)),
                mock.patch("tars.scheduler.subprocess.run"),
            ):
                from tars.scheduler import _schedule_add_linux
                _schedule_add_linux(entry, env, "/usr/bin/uv", Path("/repo"))

            service_path = Path(tmpdir) / "tars-test-perms.service"
            timer_path = Path(tmpdir) / "tars-test-perms.timer"
            self.assertTrue(service_path.exists())
            self.assertEqual(service_path.stat().st_mode & 0o777, 0o600)
            self.assertTrue(timer_path.exists())
            self.assertEqual(timer_path.stat().st_mode & 0o777, 0o600)


class ScheduleTestLinuxTests(unittest.TestCase):

    def test_schedule_test_linux_shlex(self):
        """shlex.split handles quoted args with spaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = Path(tmpdir) / "tars-shlex.service"
            service.write_text(
                "[Service]\n"
                'ExecStart=/usr/bin/uv run tars "hello world"\n'
                "WorkingDirectory=/tmp\n",
                encoding="utf-8",
            )
            with mock.patch(
                "tars.scheduler._systemd_dir", return_value=Path(tmpdir),
            ):
                with mock.patch("tars.scheduler.subprocess.run") as mock_run:
                    mock_run.return_value = mock.Mock(
                        returncode=0, stdout="ok", stderr="",
                    )
                    _schedule_test_linux("shlex")
                args = mock_run.call_args[0][0]
                self.assertEqual(
                    args, ["/usr/bin/uv", "run", "tars", "hello world"],
                )

    def test_schedule_test_linux_env_quotes(self):
        """Environment= values with systemd quoting are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = Path(tmpdir) / "tars-envq.service"
            service.write_text(
                "[Service]\n"
                'Environment="TARS_MEMORY_DIR=/my dir"\n'
                "ExecStart=/usr/bin/uv run tars\n"
                "WorkingDirectory=/tmp\n",
                encoding="utf-8",
            )
            with mock.patch(
                "tars.scheduler._systemd_dir", return_value=Path(tmpdir),
            ):
                with mock.patch("tars.scheduler.subprocess.run") as mock_run:
                    mock_run.return_value = mock.Mock(
                        returncode=0, stdout="ok", stderr="",
                    )
                    _schedule_test_linux("envq")
                env = mock_run.call_args[1]["env"]
                self.assertEqual(env["TARS_MEMORY_DIR"], "/my dir")


if __name__ == "__main__":
    unittest.main()
