import os
import re
import subprocess

_VERSION_RE = re.compile(r"version:\s*(\d+)")


def _server_binary() -> str:
    return os.environ.get("LLAMA_TUI_SERVER_BIN", "llama-server")


def _current_version() -> int | None:
    try:
        out = subprocess.check_output(
            [_server_binary(), "--version"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return None
    m = _VERSION_RE.search(out)
    return int(m.group(1)) if m else None


def check_version(min_version: int) -> bool:
    v = _current_version()
    return v is not None and v >= min_version