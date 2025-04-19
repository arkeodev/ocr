import os
import sys
from contextlib import nullcontext
from enum import Enum
from typing import Any, ContextManager


class LogLevel(Enum):
    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3


DEFAULT_LOG_LEVEL = LogLevel.INFO

debug_env = os.environ.get("DEBUG_LEVEL", "")
try:
    DEBUG_LEVEL = LogLevel(int(debug_env)) if debug_env else DEFAULT_LOG_LEVEL
except (ValueError, TypeError):
    DEBUG_LEVEL = DEFAULT_LOG_LEVEL


def init_logger():
    print(
        f"[INFO] Terminal logging active (level: {DEBUG_LEVEL.name})", file=sys.stderr
    )


def set_log_level(level: LogLevel):
    global DEBUG_LEVEL
    DEBUG_LEVEL = level
    print(f"[INFO] Log level set to: {DEBUG_LEVEL.name}", file=sys.stderr)


def log(message: str, level: LogLevel = LogLevel.INFO):
    if level.value <= DEBUG_LEVEL.value:
        print(f"[{level.name}] {message}", file=sys.stderr)


def debug(message: str):
    log(message, level=LogLevel.DEBUG)


def info(message: str, show_spinner: bool = False):
    log(message, level=LogLevel.INFO)


def success(message: str):
    print(f"[SUCCESS] {message}", file=sys.stderr)


def warning(message: str):
    log(message, level=LogLevel.WARNING)


def error(message: str):
    log(message, level=LogLevel.ERROR)


def spinner(message: str) -> ContextManager[Any]:
    print(f"[SPINNER] {message}", file=sys.stderr)
    return nullcontext()


def hidden_spinner() -> ContextManager:
    return nullcontext()
