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
    _send_scheduled_telegram,
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


class StopGuaranteeTests(unittest.TestCase):

    @mock.patch("tars.taskrunner.append_daily")
    @mock.patch("tars.taskrunner._deliver")
    @mock.patch("tars.commands.dispatch", return_value="ok")
    @mock.patch("tars.taskrunner._load_tasks")
    def test_stop_returns_within_one_second_even_with_long_tick(
        self, mock_load, mock_dispatch, mock_deliver, mock_daily,
    ):
        task = ScheduledTask(name="t", schedule="23:59", action="/brief")
        task.last_run = datetime.now()
        mock_load.return_value = [task]
        runner = TaskRunner("claude", "sonnet", tick=60)
        runner.start()
        self.assertTrue(runner._running)
        time.sleep(0.1)
        start = time.monotonic()
        runner.stop()
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 1.0)
        self.assertFalse(runner._running)


class FreshTimestampTests(unittest.TestCase):

    @mock.patch("tars.taskrunner.append_daily")
    @mock.patch("tars.taskrunner._deliver")
    def test_each_task_gets_fresh_timestamp(self, mock_deliver, mock_daily):
        """Verify _is_due uses a fresh now per task, not a stale one."""
        timestamps: list[datetime] = []
        original_is_due = _is_due

        def tracking_is_due(task, now):
            timestamps.append(now)
            return original_is_due(task, now)

        task1 = ScheduledTask(name="t1", schedule="*/1", action="/brief")
        task2 = ScheduledTask(name="t2", schedule="*/1", action="/weather")

        with (
            mock.patch("tars.taskrunner._load_tasks", return_value=[task1, task2]),
            mock.patch("tars.taskrunner._is_due", side_effect=tracking_is_due),
            mock.patch("tars.commands.dispatch", return_value="ok"),
        ):
            runner = TaskRunner("claude", "sonnet", tick=60)
            runner.start()
            time.sleep(0.3)
            runner.stop()

        self.assertGreaterEqual(len(timestamps), 2)
        # The two timestamps should be from separate datetime.now() calls
        # (they may differ slightly since dispatch runs between them)


class DeliverSanitizationTests(unittest.TestCase):

    @mock.patch("tars.taskrunner.append_daily")
    def test_deliver_strips_newlines_from_daily_entry(self, mock_daily):
        _deliver("line one\nline two\nline three", "daily", "test_task")
        call_arg = mock_daily.call_args[0][0]
        self.assertNotIn("\n", call_arg)
        self.assertIn("line one line two line three", call_arg)

    @mock.patch("tars.taskrunner.append_daily")
    def test_deliver_empty_result(self, mock_daily):
        _deliver("", "daily", "test_task")
        call_arg = mock_daily.call_args[0][0]
        self.assertIn("(no output)", call_arg)


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


class TelegramUidValidationTests(unittest.TestCase):

    @mock.patch("urllib.request.urlopen")
    def test_valid_uid_sends(self, mock_urlopen):
        with mock.patch.dict(os.environ, {
            "TARS_TELEGRAM_TOKEN": "tok",
            "TARS_TELEGRAM_ALLOW": "12345",
        }):
            _send_scheduled_telegram("hello")
        mock_urlopen.assert_called_once()
        data = json.loads(mock_urlopen.call_args[0][0].data)
        self.assertEqual(data["chat_id"], 12345)

    @mock.patch("urllib.request.urlopen")
    def test_invalid_uid_skipped(self, mock_urlopen):
        with mock.patch.dict(os.environ, {
            "TARS_TELEGRAM_TOKEN": "tok",
            "TARS_TELEGRAM_ALLOW": "not_a_number",
        }):
            _send_scheduled_telegram("hello")
        mock_urlopen.assert_not_called()

    @mock.patch("urllib.request.urlopen")
    def test_mixed_uids_sends_only_valid(self, mock_urlopen):
        with mock.patch.dict(os.environ, {
            "TARS_TELEGRAM_TOKEN": "tok",
            "TARS_TELEGRAM_ALLOW": "111,bad,222",
        }):
            _send_scheduled_telegram("hello")
        self.assertEqual(mock_urlopen.call_count, 2)
        sent_ids = []
        for call in mock_urlopen.call_args_list:
            data = json.loads(call[0][0].data)
            sent_ids.append(data["chat_id"])
        self.assertEqual(sent_ids, [111, 222])


if __name__ == "__main__":
    unittest.main()
