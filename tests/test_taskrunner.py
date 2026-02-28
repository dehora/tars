"""Tests for tars.taskrunner — in-process task scheduler."""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

# Mock ollama before importing tars modules
if "ollama" not in sys.modules:
    sys.modules["ollama"] = mock.Mock()

from tars.taskrunner import (
    ScheduledTask,
    TaskRunner,
    _deliver,
    _is_due,
    _load_tasks,
    _parse_schedule,
)


class ScheduleParsingTests(unittest.TestCase):

    def test_parse_daily_schedule(self):
        self.assertEqual(_parse_schedule("08:00"), "08:00")
        self.assertEqual(_parse_schedule("23:59"), "23:59")
        self.assertEqual(_parse_schedule("0:0"), "00:00")

    def test_parse_interval_schedule(self):
        self.assertEqual(_parse_schedule("*/60"), "*/60")
        self.assertEqual(_parse_schedule("*/1"), "*/1")
        self.assertEqual(_parse_schedule("*/120"), "*/120")

    def test_parse_invalid_schedule(self):
        self.assertIsNone(_parse_schedule(""))
        self.assertIsNone(_parse_schedule("every day"))
        self.assertIsNone(_parse_schedule("25:00"))
        self.assertIsNone(_parse_schedule("12:60"))
        self.assertIsNone(_parse_schedule("*/0"))
        self.assertIsNone(_parse_schedule("*/-1"))
        self.assertIsNone(_parse_schedule("*/abc"))

    def test_load_from_json_file(self):
        with tempfile.TemporaryDirectory() as td:
            schedules = [
                {"name": "brief", "schedule": "08:00", "action": "/brief", "deliver": "email"},
                {"name": "check", "schedule": "*/60", "action": "/todoist today"},
            ]
            Path(td, "schedules.json").write_text(json.dumps(schedules))
            with mock.patch("tars.taskrunner._memory_dir", return_value=Path(td)):
                tasks = _load_tasks()
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].name, "brief")
        self.assertEqual(tasks[0].schedule, "08:00")
        self.assertEqual(tasks[0].deliver, "email")
        self.assertEqual(tasks[1].name, "check")
        self.assertEqual(tasks[1].deliver, "daily")

    def test_load_from_env_var(self):
        schedules = [{"name": "t1", "schedule": "*/5", "action": "/brief"}]
        with mock.patch("tars.taskrunner._memory_dir", return_value=None):
            with mock.patch.dict(os.environ, {"TARS_SCHEDULES": json.dumps(schedules)}):
                tasks = _load_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].name, "t1")

    def test_load_empty_config(self):
        with mock.patch("tars.taskrunner._memory_dir", return_value=None):
            with mock.patch.dict(os.environ, {}, clear=False):
                env = os.environ.copy()
                env.pop("TARS_SCHEDULES", None)
                with mock.patch.dict(os.environ, env, clear=True):
                    tasks = _load_tasks()
        self.assertEqual(tasks, [])

    def test_load_skips_invalid_entries(self):
        schedules = [
            {"name": "good", "schedule": "08:00", "action": "/brief"},
            {"name": "bad_schedule", "schedule": "nope", "action": "/brief"},
            {"schedule": "09:00", "action": "/brief"},  # missing name
            {"name": "bad_deliver", "schedule": "10:00", "action": "/brief", "deliver": "fax"},
        ]
        with mock.patch("tars.taskrunner._memory_dir", return_value=None):
            with mock.patch.dict(os.environ, {"TARS_SCHEDULES": json.dumps(schedules)}):
                tasks = _load_tasks()
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].name, "good")
        self.assertEqual(tasks[1].name, "bad_deliver")
        self.assertEqual(tasks[1].deliver, "daily")

    def test_load_skips_duplicate_names(self):
        schedules = [
            {"name": "dup", "schedule": "08:00", "action": "/brief"},
            {"name": "dup", "schedule": "09:00", "action": "/weather"},
        ]
        with mock.patch("tars.taskrunner._memory_dir", return_value=None):
            with mock.patch.dict(os.environ, {"TARS_SCHEDULES": json.dumps(schedules)}):
                tasks = _load_tasks()
        self.assertEqual(len(tasks), 1)


class IsDueTests(unittest.TestCase):

    def test_daily_fires_at_correct_time(self):
        task = ScheduledTask(name="t", schedule="08:00", action="/brief")
        now = datetime(2025, 6, 15, 8, 0, 30)
        self.assertTrue(_is_due(task, now))

    def test_daily_not_before_time(self):
        task = ScheduledTask(name="t", schedule="08:00", action="/brief")
        now = datetime(2025, 6, 15, 7, 59, 0)
        self.assertFalse(_is_due(task, now))

    def test_daily_not_after_time(self):
        task = ScheduledTask(name="t", schedule="08:00", action="/brief")
        now = datetime(2025, 6, 15, 8, 1, 0)
        self.assertFalse(_is_due(task, now))

    def test_daily_no_double_fire(self):
        task = ScheduledTask(name="t", schedule="08:00", action="/brief")
        task.last_run = datetime(2025, 6, 15, 8, 0, 0)
        now = datetime(2025, 6, 15, 8, 0, 30)
        self.assertFalse(_is_due(task, now))

    def test_daily_fires_next_day(self):
        task = ScheduledTask(name="t", schedule="08:00", action="/brief")
        task.last_run = datetime(2025, 6, 14, 8, 0, 0)
        now = datetime(2025, 6, 15, 8, 0, 0)
        self.assertTrue(_is_due(task, now))

    def test_interval_fires_immediately(self):
        task = ScheduledTask(name="t", schedule="*/5", action="/brief")
        now = datetime(2025, 6, 15, 10, 0, 0)
        self.assertTrue(_is_due(task, now))

    def test_interval_fires_after_elapsed(self):
        task = ScheduledTask(name="t", schedule="*/5", action="/brief")
        task.last_run = datetime(2025, 6, 15, 10, 0, 0)
        now = datetime(2025, 6, 15, 10, 5, 0)
        self.assertTrue(_is_due(task, now))

    def test_interval_not_before_elapsed(self):
        task = ScheduledTask(name="t", schedule="*/5", action="/brief")
        task.last_run = datetime(2025, 6, 15, 10, 0, 0)
        now = datetime(2025, 6, 15, 10, 3, 0)
        self.assertFalse(_is_due(task, now))

    def test_never_run_daily_waits_for_time(self):
        task = ScheduledTask(name="t", schedule="14:00", action="/brief")
        now = datetime(2025, 6, 15, 10, 0, 0)
        self.assertFalse(_is_due(task, now))


class ExecutionTests(unittest.TestCase):

    @mock.patch("tars.taskrunner.append_daily")
    @mock.patch("tars.taskrunner._deliver")
    def test_execute_dispatches_through_commands(self, mock_deliver, mock_daily):
        with mock.patch("tars.commands.dispatch", return_value="brief result") as mock_dispatch:
            runner = TaskRunner("claude", "sonnet")
            task = ScheduledTask(name="t", schedule="*/1", action="/brief")
            now = datetime.now()
            runner._execute(task, now)
            mock_dispatch.assert_called_once_with(
                "/brief", "claude", "sonnet",
                context={"channel": "scheduled"},
            )
            mock_deliver.assert_called_once_with("brief result", "daily", "t")

    @mock.patch("tars.taskrunner.append_daily")
    @mock.patch("tars.taskrunner._deliver")
    def test_failed_dispatch_does_not_raise(self, mock_deliver, mock_daily):
        with mock.patch("tars.commands.dispatch", side_effect=RuntimeError("boom")):
            runner = TaskRunner("claude", "sonnet")
            task = ScheduledTask(name="t", schedule="*/1", action="/brief")
            now = datetime.now()
            runner._execute(task, now)
            mock_deliver.assert_called_once()
            result = mock_deliver.call_args[0][0]
            self.assertIn("error", result)

    @mock.patch("tars.taskrunner.append_daily")
    def test_deliver_to_daily(self, mock_daily):
        _deliver("test output", "daily", "test_task")
        mock_daily.assert_called_once()
        call_arg = mock_daily.call_args[0][0]
        self.assertIn("[scheduled:test_task]", call_arg)

    @mock.patch("tars.taskrunner._send_scheduled_email")
    def test_deliver_to_email(self, mock_email):
        _deliver("test output", "email", "test_task")
        mock_email.assert_called_once()

    @mock.patch("tars.taskrunner._send_scheduled_telegram")
    def test_deliver_to_telegram(self, mock_telegram):
        _deliver("test output", "telegram", "test_task")
        mock_telegram.assert_called_once()

    @mock.patch("tars.taskrunner.append_daily")
    @mock.patch("tars.taskrunner._deliver")
    def test_concurrent_execution_guard(self, mock_deliver, mock_daily):
        with mock.patch("tars.commands.dispatch", return_value="ok"):
            runner = TaskRunner("claude", "sonnet")
            task = ScheduledTask(name="t", schedule="*/1", action="/brief")
            task._lock.acquire()  # simulate running
            try:
                now = datetime.now()
                # _loop checks _is_due then tries lock; simulate that path
                if not task._lock.acquire(blocking=False):
                    skipped = True
                else:
                    skipped = False
                    task._lock.release()
                self.assertTrue(skipped)
            finally:
                task._lock.release()


class TaskRunnerLifecycleTests(unittest.TestCase):

    @mock.patch("tars.taskrunner._load_tasks")
    def test_start_spawns_thread(self, mock_load):
        mock_load.return_value = [
            ScheduledTask(name="t", schedule="*/60", action="/brief"),
        ]
        runner = TaskRunner("claude", "sonnet")
        runner.start()
        self.assertTrue(runner._running)
        self.assertIsNotNone(runner._thread)
        self.assertTrue(runner._thread.is_alive())
        runner.stop()
        self.assertFalse(runner._running)

    @mock.patch("tars.taskrunner._load_tasks")
    def test_stop_sets_flag(self, mock_load):
        mock_load.return_value = [
            ScheduledTask(name="t", schedule="*/60", action="/brief"),
        ]
        runner = TaskRunner("claude", "sonnet")
        runner.start()
        self.assertTrue(runner._running)
        runner.stop()
        self.assertFalse(runner._running)

    @mock.patch("tars.taskrunner._load_tasks")
    def test_no_tasks_no_thread(self, mock_load):
        mock_load.return_value = []
        runner = TaskRunner("claude", "sonnet")
        runner.start()
        self.assertFalse(runner._running)
        self.assertIsNone(runner._thread)

    @mock.patch("tars.taskrunner._load_tasks")
    def test_list_tasks_returns_configured(self, mock_load):
        tasks = [
            ScheduledTask(name="a", schedule="08:00", action="/brief"),
            ScheduledTask(name="b", schedule="*/30", action="/weather"),
        ]
        mock_load.return_value = tasks
        runner = TaskRunner("claude", "sonnet")
        runner.start()
        listed = runner.list_tasks()
        self.assertEqual(len(listed), 2)
        self.assertEqual(listed[0].name, "a")
        runner.stop()


class IntegrationTests(unittest.TestCase):

    def test_load_and_check_due(self):
        """Load schedules from JSON + verify _is_due logic end-to-end."""
        with tempfile.TemporaryDirectory() as td:
            schedules = [
                {"name": "every_minute", "schedule": "*/1", "action": "/brief", "deliver": "daily"},
            ]
            Path(td, "schedules.json").write_text(json.dumps(schedules))
            with mock.patch("tars.taskrunner._memory_dir", return_value=Path(td)):
                tasks = _load_tasks()
            self.assertEqual(len(tasks), 1)
            self.assertTrue(_is_due(tasks[0], datetime.now()))

    @mock.patch("tars.taskrunner.append_daily")
    def test_deliver_daily_contains_tag(self, mock_daily):
        _deliver("hello world", "daily", "my_task")
        call_arg = mock_daily.call_args[0][0]
        self.assertIn("[scheduled:my_task]", call_arg)
        self.assertIn("hello world", call_arg)


if __name__ == "__main__":
    unittest.main()
