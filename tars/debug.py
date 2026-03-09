"""Central verbose/debug toggle for tars."""

import os
import sys
import threading

VERBOSE = False

_lock = threading.Lock()
_active_spinner = None


def configure(*, from_env: bool = False, enable: bool = False):
    """Set VERBOSE from env and/or explicit flag. Call after load_dotenv()."""
    global VERBOSE
    if from_env and os.environ.get("TARS_VERBOSE", "").strip() == "1":
        VERBOSE = True
    if enable:
        VERBOSE = True


def set_spinner(spinner):
    global _active_spinner
    _active_spinner = spinner


def verbose(*args, **kwargs):
    if VERBOSE:
        with _lock:
            if _active_spinner and _active_spinner.spinning:
                sys.stdout.write("\r" + " " * 40 + "\r")
                sys.stdout.flush()
            print(*args, file=sys.stderr, **kwargs)
            sys.stderr.flush()
