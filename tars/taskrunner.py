"""In-process task scheduler for long-lived tars processes.

Runs as a daemon thread inside `tars serve`, `tars telegram`, or `tars email`.
Fires configured tasks as slash commands through commands.dispatch() and delivers
results via daily memory, email, or telegram.

Complements (does not replace) the OS-level scheduler in tars/scheduler.py.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

from tars.memory import _memory_dir, append_daily

logger = logging.getLogger(__name__)

_TICK_SECONDS = 60


@dataclass
class ScheduledTask:
    name: str
    schedule: str  # "HH:MM" for daily or "*/N" for interval minutes
    action: str  # slash command to dispatch
    deliver: str = "daily"  # "daily" | "email" | "telegram"
    last_run: datetime | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _parse_schedule(raw: str) -> str | None:
    """Validate schedule format. Returns normalized string or None if invalid."""
    raw = raw.strip()
    if raw.startswith("*/"):
        try:
            n = int(raw[2:])
            if n < 1:
                return None
            return f"*/{n}"
        except ValueError:
            return None
    parts = raw.split(":")
    if len(parts) == 2:
        try:
            h, m = int(parts[0]), int(parts[1])
            if 0 <= h <= 23 and 0 <= m <= 59:
                return f"{h:02d}:{m:02d}"
        except ValueError:
            pass
    return None


def _load_tasks() -> list[ScheduledTask]:
    """Load scheduled tasks from schedules.json or TARS_SCHEDULES env var."""
    raw_json = None

    md = _memory_dir()
    if md is not None:
        schedules_path = md / "schedules.json"
        if schedules_path.exists():
            try:
                raw_json = schedules_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("failed to read schedules.json: %s", e)

    if raw_json is None:
        raw_json = os.environ.get("TARS_SCHEDULES")

    if not raw_json:
        return []

    try:
        entries = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning("invalid schedules JSON: %s", e)
        return []

    if not isinstance(entries, list):
        logger.warning("schedules JSON must be a list")
        return []

    tasks: list[ScheduledTask] = []
    seen_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        schedule = entry.get("schedule", "")
        action = entry.get("action", "")
        deliver = entry.get("deliver", "daily")

        if not name or not schedule or not action:
            logger.warning("skipping incomplete schedule entry: %s", entry)
            continue

        if name in seen_names:
            logger.warning("skipping duplicate schedule name: %s", name)
            continue

        parsed = _parse_schedule(schedule)
        if parsed is None:
            logger.warning("invalid schedule format '%s' for task '%s'", schedule, name)
            continue

        if deliver not in ("daily", "email", "telegram"):
            logger.warning("invalid deliver target '%s' for task '%s', defaulting to daily", deliver, name)
            deliver = "daily"

        seen_names.add(name)
        tasks.append(ScheduledTask(
            name=name,
            schedule=parsed,
            action=action,
            deliver=deliver,
        ))

    return tasks


def _is_due(task: ScheduledTask, now: datetime) -> bool:
    """Check if a task should fire at the given time."""
    if task.schedule.startswith("*/"):
        interval_minutes = int(task.schedule[2:])
        if task.last_run is None:
            return True
        elapsed = (now - task.last_run).total_seconds() / 60
        return elapsed >= interval_minutes

    # Daily "HH:MM"
    h, m = task.schedule.split(":")
    target_hour, target_minute = int(h), int(m)

    if now.hour != target_hour or now.minute != target_minute:
        return False

    if task.last_run is not None and task.last_run.date() == now.date():
        return False

    return True


def _deliver(result: str, target: str, task_name: str) -> None:
    """Deliver a scheduled task result to the configured target."""
    tag = f"[scheduled:{task_name}]"
    truncated = result[:200] if result else "(no output)"

    if target == "daily":
        try:
            append_daily(f"{tag} {truncated}")
        except Exception as e:
            logger.warning("daily delivery failed for %s: %s", task_name, e)
        return

    if target == "email":
        try:
            _send_scheduled_email(f"{tag}\n\n{result}")
        except Exception as e:
            logger.warning("email delivery failed for %s: %s", task_name, e)
        return

    if target == "telegram":
        try:
            _send_scheduled_telegram(f"{tag}\n\n{result}")
        except Exception as e:
            logger.warning("telegram delivery failed for %s: %s", task_name, e)
        return


def _send_scheduled_email(body: str) -> None:
    """Send a scheduled result via email (reuses email.py SMTP pattern)."""
    address = os.environ.get("TARS_EMAIL_ADDRESS")
    password = os.environ.get("TARS_EMAIL_PASSWORD")
    to_addr = os.environ.get("TARS_EMAIL_TO")
    if not address or not password or not to_addr:
        logger.warning("email delivery skipped: missing TARS_EMAIL_* config")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = address
    msg["To"] = to_addr
    msg["Subject"] = "tars scheduled"
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(address, password)
        smtp.send_message(msg)


def _send_scheduled_telegram(body: str) -> None:
    """Send a scheduled result via Telegram (reuses telegram.py pattern)."""
    import urllib.request

    token = os.environ.get("TARS_TELEGRAM_TOKEN")
    allow = os.environ.get("TARS_TELEGRAM_ALLOW")
    if not token or not allow:
        logger.warning("telegram delivery skipped: missing TARS_TELEGRAM_* config")
        return
    for uid in allow.split(","):
        uid = uid.strip()
        if not uid:
            continue
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({"chat_id": uid, "text": body[:4096]}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning("telegram send to %s failed: %s", uid, e)


class TaskRunner:
    """In-process scheduler that runs as a daemon thread."""

    def __init__(self, provider: str, model: str, *, tick: int | None = None) -> None:
        self._provider = provider
        self._model = model
        self._tick = tick if tick is not None else _TICK_SECONDS
        self._running = False
        self._thread: threading.Thread | None = None
        self._tasks: list[ScheduledTask] = []

    def start(self) -> None:
        self._tasks = _load_tasks()
        if not self._tasks:
            logger.info("taskrunner: no scheduled tasks configured")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="taskrunner")
        self._thread.start()
        names = ", ".join(t.name for t in self._tasks)
        logger.info("taskrunner: started with %d task(s): %s", len(self._tasks), names)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("taskrunner: stopped")

    def list_tasks(self) -> list[ScheduledTask]:
        return list(self._tasks)

    def _loop(self) -> None:
        while self._running:
            now = datetime.now()
            for task in self._tasks:
                if _is_due(task, now):
                    if not task._lock.acquire(blocking=False):
                        logger.info("taskrunner: skipping %s (still running)", task.name)
                        continue
                    try:
                        self._execute(task, now)
                    finally:
                        task._lock.release()
            time.sleep(self._tick)

    def _execute(self, task: ScheduledTask, now: datetime) -> None:
        from tars.commands import dispatch

        logger.info("taskrunner: executing %s → %s", task.name, task.action)
        task.last_run = now

        try:
            result = dispatch(
                task.action,
                self._provider,
                self._model,
                context={"channel": "scheduled"},
            )
            if result is None:
                result = "(no output)"
        except Exception as e:
            logger.error("taskrunner: %s failed: %s", task.name, e)
            result = f"(error: {e})"

        try:
            append_daily(f"[scheduled:{task.name}] ran {task.action}")
        except Exception:
            pass

        try:
            _deliver(result, task.deliver, task.name)
        except Exception as e:
            logger.error("taskrunner: delivery failed for %s: %s", task.name, e)
