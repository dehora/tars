"""Zero-dependency ANSI color helpers with NO_COLOR and non-TTY support."""

import os
import sys

_ENABLED = (
    "NO_COLOR" not in os.environ
    and hasattr(sys.stdout, "isatty")
    and sys.stdout.isatty()
)

RESET = "\033[0m" if _ENABLED else ""
BOLD = "\033[1m" if _ENABLED else ""
DIM = "\033[2m" if _ENABLED else ""

_CYAN = "\033[36m" if _ENABLED else ""
_GREEN = "\033[32m" if _ENABLED else ""
_YELLOW = "\033[33m" if _ENABLED else ""
_RED = "\033[31m" if _ENABLED else ""
_MAGENTA = "\033[35m" if _ENABLED else ""
_BLUE = "\033[34m" if _ENABLED else ""


def bold(s: str) -> str:
    return f"{BOLD}{s}{RESET}" if _ENABLED else s


def dim(s: str) -> str:
    return f"{DIM}{s}{RESET}" if _ENABLED else s


def cyan(s: str) -> str:
    return f"{_CYAN}{s}{RESET}" if _ENABLED else s


def green(s: str) -> str:
    return f"{_GREEN}{s}{RESET}" if _ENABLED else s


def yellow(s: str) -> str:
    return f"{_YELLOW}{s}{RESET}" if _ENABLED else s


def red(s: str) -> str:
    return f"{_RED}{s}{RESET}" if _ENABLED else s


def magenta(s: str) -> str:
    return f"{_MAGENTA}{s}{RESET}" if _ENABLED else s


def blue(s: str) -> str:
    return f"{_BLUE}{s}{RESET}" if _ENABLED else s
