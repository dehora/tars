"""OS-level schedule management for tars commands.

Creates and manages launchd plists (macOS) or systemd units (Linux) that run
tars subcommands on a timer or when a watched path changes.
"""

import os
import platform
import plistlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

LABEL_PREFIX = "com.dehora.tars"

_KNOWN_ENV_KEYS = [
    "TARS_MEMORY_DIR",
    "TARS_NOTES_DIR",
    "TARS_DEFAULT_MODEL",
    "TARS_MODEL",
    "TARS_REMOTE_MODEL",
    "TARS_ESCALATION_MODEL",
    "TARS_ROUTING_POLICY",
    "TARS_MAX_TOKENS",
    "ANTHROPIC_API_KEY",
    "TARS_EMAIL_ADDRESS",
    "TARS_EMAIL_PASSWORD",
    "TARS_EMAIL_ALLOW",
    "TARS_EMAIL_TO",
    "TARS_EMAIL_POLL_INTERVAL",
    "TARS_API_TOKEN",
    "DEFAULT_LAT",
    "DEFAULT_LON",
]


@dataclass
class ScheduleEntry:
    name: str  # e.g. "email-brief"
    command: str  # e.g. "email-brief"
    args: list[str] = field(default_factory=list)
    hour: int | None = None  # calendar trigger
    minute: int | None = None
    watch_path: str | None = None  # file watcher trigger


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _load_dotenv_values() -> dict[str, str | None]:
    """Load .env values via dotenv, returning empty dict on import failure."""
    try:
        from dotenv import dotenv_values

        return dotenv_values()
    except ImportError:
        return {}


def _capture_env() -> dict[str, str]:
    """Capture known tars env vars from .env and current environment."""
    env: dict[str, str] = {}
    # Try loading .env via dotenv
    dot = _load_dotenv_values()
    for key in _KNOWN_ENV_KEYS:
        val = dot.get(key)
        if val:
            env[key] = val
    # Overlay from current os.environ (takes precedence)
    for key in _KNOWN_ENV_KEYS:
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


def _find_uv() -> str:
    """Find the uv binary path."""
    # Check TARS_UV first
    tars_uv = os.environ.get("TARS_UV", "")
    if tars_uv and shutil.which(tars_uv):
        return tars_uv

    # Check PATH
    found = shutil.which("uv")
    if found:
        return found

    # Check known locations
    candidates = [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
        Path("/usr/bin/uv"),
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)

    raise FileNotFoundError("uv not found. Install uv or set TARS_UV to its path.")


def _find_repo() -> Path:
    """Find the tars repo root directory."""
    tars_home = os.environ.get("TARS_HOME")
    if tars_home:
        return Path(tars_home)
    # Derive from package location: tars/scheduler.py -> repo root
    return Path(__file__).resolve().parent.parent


def _label(name: str) -> str:
    return f"{LABEL_PREFIX}-{name}"


def _plist_path(name: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_label(name)}.plist"


def _log_dir_macos() -> Path:
    return Path.home() / "Library" / "Logs"


def _systemd_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


# -- plist generation (macOS) --


def _build_command_string(uv_path: str, command: str, args: list[str]) -> str:
    """Build the shell command string for the schedule."""
    parts = [f'echo "[tars-{command}] $(date -Iseconds) start"']
    cmd_parts = [uv_path, "run", "tars", command] + args
    parts.append(" ".join(cmd_parts))
    return "; ".join(parts)


def _generate_plist(
    entry: ScheduleEntry,
    env: dict[str, str],
    uv_path: str,
    repo_dir: Path,
) -> dict:
    """Generate a launchd plist dict."""
    cmd = _build_command_string(uv_path, entry.command, entry.args)
    log_dir = _log_dir_macos()

    plist: dict = {
        "Label": _label(entry.name),
        "ProgramArguments": ["/bin/bash", "-lc", cmd],
        "WorkingDirectory": str(repo_dir),
        "StandardOutPath": str(log_dir / f"tars-{entry.name}.log"),
        "StandardErrorPath": str(log_dir / f"tars-{entry.name}.err.log"),
    }

    if env:
        plist["EnvironmentVariables"] = dict(env)

    if entry.watch_path:
        plist["WatchPaths"] = [entry.watch_path]
    else:
        interval: dict = {}
        if entry.hour is not None:
            interval["Hour"] = entry.hour
        if entry.minute is not None:
            interval["Minute"] = entry.minute
        plist["StartCalendarInterval"] = interval

    return plist


# -- systemd generation (Linux) --


def _generate_systemd_service(
    entry: ScheduleEntry,
    env: dict[str, str],
    uv_path: str,
    repo_dir: Path,
) -> str:
    """Generate a systemd service unit file."""
    cmd = _build_command_string(uv_path, entry.command, entry.args)
    lines = [
        "[Unit]",
        f"Description=tars {entry.name}",
        "",
        "[Service]",
        "Type=oneshot",
        f"WorkingDirectory={repo_dir}",
        f"ExecStart=/bin/bash -lc '{cmd}'",
    ]
    for key, val in env.items():
        lines.append(f"Environment={key}={val}")
    return "\n".join(lines) + "\n"


def _generate_systemd_timer(entry: ScheduleEntry) -> str:
    """Generate a systemd timer unit file."""
    hour = entry.hour if entry.hour is not None else 0
    minute = entry.minute if entry.minute is not None else 0
    lines = [
        "[Unit]",
        f"Description=Run tars {entry.name}",
        "",
        "[Timer]",
        f"OnCalendar=*-*-* {hour:02d}:{minute:02d}:00",
        "Persistent=true",
        "",
        "[Install]",
        "WantedBy=timers.target",
    ]
    return "\n".join(lines) + "\n"


def _generate_systemd_path(entry: ScheduleEntry) -> str:
    """Generate a systemd path unit file for watch mode."""
    lines = [
        "[Unit]",
        f"Description=Watch paths for tars {entry.name}",
        "",
        "[Path]",
        f"PathModified={entry.watch_path}",
        "",
        "[Install]",
        "WantedBy=paths.target",
    ]
    return "\n".join(lines) + "\n"


# -- public API --


def schedule_add(entry: ScheduleEntry) -> str:
    """Install an OS-level schedule for a tars command."""
    uv_path = _find_uv()
    repo_dir = _find_repo()
    env = _capture_env()

    if _is_macos():
        return _schedule_add_macos(entry, env, uv_path, repo_dir)
    elif _is_linux():
        return _schedule_add_linux(entry, env, uv_path, repo_dir)
    else:
        return f"unsupported platform: {platform.system()}"


def _schedule_add_macos(
    entry: ScheduleEntry,
    env: dict[str, str],
    uv_path: str,
    repo_dir: Path,
) -> str:
    plist = _generate_plist(entry, env, uv_path, repo_dir)
    plist_path = _plist_path(entry.name)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    _log_dir_macos().mkdir(parents=True, exist_ok=True)

    # Unload existing if present
    if plist_path.exists():
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True,
        )

    with open(plist_path, "wb") as f:
        plistlib.dump(plist, f)

    result = subprocess.run(
        ["launchctl", "load", str(plist_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"launchctl load failed: {result.stderr.strip()}"

    if entry.watch_path:
        trigger = f"watch {entry.watch_path}"
    else:
        h = entry.hour if entry.hour is not None else 0
        m = entry.minute if entry.minute is not None else 0
        trigger = f"daily {h:02d}:{m:02d}"
    return f"installed: {_label(entry.name)} ({trigger})"


def _schedule_add_linux(
    entry: ScheduleEntry,
    env: dict[str, str],
    uv_path: str,
    repo_dir: Path,
) -> str:
    unit_dir = _systemd_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_name = f"tars-{entry.name}"

    service_content = _generate_systemd_service(entry, env, uv_path, repo_dir)
    service_path = unit_dir / f"{unit_name}.service"
    service_path.write_text(service_content, encoding="utf-8")

    if entry.watch_path:
        path_content = _generate_systemd_path(entry)
        path_path = unit_dir / f"{unit_name}.path"
        path_path.write_text(path_content, encoding="utf-8")
        trigger_unit = f"{unit_name}.path"
        trigger_desc = f"watch {entry.watch_path}"
    else:
        timer_content = _generate_systemd_timer(entry)
        timer_path = unit_dir / f"{unit_name}.timer"
        timer_path.write_text(timer_content, encoding="utf-8")
        trigger_unit = f"{unit_name}.timer"
        h = entry.hour if entry.hour is not None else 0
        m = entry.minute if entry.minute is not None else 0
        trigger_desc = f"daily {h:02d}:{m:02d}"

    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    result = subprocess.run(
        ["systemctl", "--user", "enable", "--now", trigger_unit],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"systemctl enable failed: {result.stderr.strip()}"

    return f"installed: {unit_name} ({trigger_desc})"


def schedule_remove(name: str) -> str:
    """Remove an installed schedule."""
    if _is_macos():
        return _schedule_remove_macos(name)
    elif _is_linux():
        return _schedule_remove_linux(name)
    else:
        return f"unsupported platform: {platform.system()}"


def _schedule_remove_macos(name: str) -> str:
    plist_path = _plist_path(name)
    if not plist_path.exists():
        return f"no schedule found: {name}"

    subprocess.run(
        ["launchctl", "unload", str(plist_path)],
        capture_output=True,
    )
    plist_path.unlink()
    return f"removed: {_label(name)}"


def _schedule_remove_linux(name: str) -> str:
    unit_dir = _systemd_dir()
    unit_name = f"tars-{name}"
    service_path = unit_dir / f"{unit_name}.service"
    timer_path = unit_dir / f"{unit_name}.timer"
    path_path = unit_dir / f"{unit_name}.path"

    if not service_path.exists():
        return f"no schedule found: {name}"

    # Disable timer or path unit
    for unit_file in [timer_path, path_path]:
        if unit_file.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", unit_file.name],
                capture_output=True,
            )
            unit_file.unlink()

    service_path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    return f"removed: {unit_name}"


def schedule_list() -> list[dict]:
    """List all installed tars schedules."""
    if _is_macos():
        return _schedule_list_macos()
    elif _is_linux():
        return _schedule_list_linux()
    return []


def _schedule_list_macos() -> list[dict]:
    agents_dir = Path.home() / "Library" / "LaunchAgents"
    if not agents_dir.is_dir():
        return []

    results = []
    for plist_file in sorted(agents_dir.glob(f"{LABEL_PREFIX}-*.plist")):
        try:
            with open(plist_file, "rb") as f:
                plist = plistlib.load(f)
        except Exception:
            continue

        label = plist.get("Label", plist_file.stem)
        name = label.removeprefix(f"{LABEL_PREFIX}-")

        # Determine trigger
        if "WatchPaths" in plist:
            paths = plist["WatchPaths"]
            trigger = f"watch {paths[0]}" if paths else "watch"
        elif "StartCalendarInterval" in plist:
            cal = plist["StartCalendarInterval"]
            h = cal.get("Hour", 0)
            m = cal.get("Minute", 0)
            trigger = f"daily {h:02d}:{m:02d}"
        else:
            trigger = "unknown"

        # Last run from log file
        log_path = plist.get("StandardOutPath", "")
        last_run = _read_last_log_line(log_path)

        results.append({
            "name": name,
            "trigger": trigger,
            "last_run": last_run,
            "log": log_path,
        })

    return results


def _schedule_list_linux() -> list[dict]:
    unit_dir = _systemd_dir()
    if not unit_dir.is_dir():
        return []

    results = []
    for service_file in sorted(unit_dir.glob("tars-*.service")):
        name = service_file.stem.removeprefix("tars-")
        timer_file = unit_dir / f"tars-{name}.timer"
        path_file = unit_dir / f"tars-{name}.path"

        # Determine trigger
        if path_file.exists():
            content = path_file.read_text(encoding="utf-8", errors="replace")
            watch = ""
            for line in content.splitlines():
                if line.startswith("PathModified="):
                    watch = line.split("=", 1)[1]
            trigger = f"watch {watch}" if watch else "watch"
        elif timer_file.exists():
            content = timer_file.read_text(encoding="utf-8", errors="replace")
            trigger = "timer"
            for line in content.splitlines():
                if line.startswith("OnCalendar="):
                    cal = line.split("=", 1)[1]
                    # Parse "*-*-* HH:MM:SS" -> "daily HH:MM"
                    parts = cal.strip().split()
                    if len(parts) >= 2:
                        time_part = parts[1].rsplit(":", 1)[0]  # drop seconds
                        trigger = f"daily {time_part}"
        else:
            trigger = "unknown"

        # Last run from journalctl
        last_run = _read_journalctl_last(f"tars-{name}")

        results.append({
            "name": name,
            "trigger": trigger,
            "last_run": last_run,
            "log": f"journalctl --user -u tars-{name}",
        })

    return results


def _read_last_log_line(log_path: str) -> str:
    """Read the last non-empty line from a log file."""
    if not log_path:
        return ""
    try:
        p = Path(log_path)
        if not p.exists() or p.stat().st_size == 0:
            return "(never)"
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = [l for l in text.strip().splitlines() if l.strip()]
        return lines[-1] if lines else "(empty)"
    except Exception:
        return "(error reading log)"


def _read_journalctl_last(unit_name: str) -> str:
    """Read the last journalctl line for a systemd unit."""
    try:
        result = subprocess.run(
            ["journalctl", "--user", "-u", unit_name, "-n", "1", "--no-pager", "-o", "short-iso"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout.strip()
        if output and "No entries" not in output and "-- No entries --" not in output:
            return output
        return "(never)"
    except Exception:
        return "(unknown)"


def schedule_test(name: str) -> str:
    """Test-run a schedule using only its baked environment."""
    if _is_macos():
        return _schedule_test_macos(name)
    elif _is_linux():
        return _schedule_test_linux(name)
    return f"unsupported platform: {platform.system()}"


def _schedule_test_macos(name: str) -> str:
    plist_path = _plist_path(name)
    if not plist_path.exists():
        return f"no schedule found: {name}"

    try:
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
    except Exception as e:
        return f"failed to read plist: {e}"

    prog_args = plist.get("ProgramArguments", [])
    if len(prog_args) < 3:
        return f"unexpected ProgramArguments: {prog_args}"

    work_dir = plist.get("WorkingDirectory", ".")
    baked_env = plist.get("EnvironmentVariables", {})

    # Build a minimal env: only baked vars + PATH for shell to work
    test_env = dict(baked_env)
    test_env.setdefault("PATH", "/usr/bin:/bin:/usr/local/bin")
    test_env.setdefault("HOME", str(Path.home()))

    try:
        result = subprocess.run(
            prog_args,
            cwd=work_dir,
            env=test_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "test timed out after 120s"
    except Exception as e:
        return f"test failed to run: {e}"

    output = []
    output.append(f"exit code: {result.returncode}")
    if result.stdout.strip():
        output.append(f"stdout:\n{result.stdout.strip()}")
    if result.stderr.strip():
        output.append(f"stderr:\n{result.stderr.strip()}")
    return "\n".join(output)


def _schedule_test_linux(name: str) -> str:
    service_path = _systemd_dir() / f"tars-{name}.service"
    if not service_path.exists():
        return f"no schedule found: {name}"

    content = service_path.read_text(encoding="utf-8", errors="replace")
    exec_start = ""
    work_dir = "."
    baked_env: dict[str, str] = {}

    for line in content.splitlines():
        if line.startswith("ExecStart="):
            exec_start = line.split("=", 1)[1]
        elif line.startswith("WorkingDirectory="):
            work_dir = line.split("=", 1)[1]
        elif line.startswith("Environment="):
            kv = line.split("=", 1)[1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                baked_env[k] = v

    if not exec_start:
        return f"no ExecStart found in {service_path}"

    test_env = dict(baked_env)
    test_env.setdefault("PATH", "/usr/bin:/bin:/usr/local/bin")
    test_env.setdefault("HOME", str(Path.home()))

    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", exec_start.lstrip("/bin/bash -lc ").strip("'")],
            cwd=work_dir,
            env=test_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "test timed out after 120s"
    except Exception as e:
        return f"test failed to run: {e}"

    output = []
    output.append(f"exit code: {result.returncode}")
    if result.stdout.strip():
        output.append(f"stdout:\n{result.stdout.strip()}")
    if result.stderr.strip():
        output.append(f"stderr:\n{result.stderr.strip()}")
    return "\n".join(output)
