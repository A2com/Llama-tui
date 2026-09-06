import os
import re
import subprocess

_VERSION_RE = re.compile(r"version:\s*(\S+)")


def _server_binary() -> str:
    return os.environ.get("LLAMA_TUI_SERVER_BIN", "llama-server")


def _raw_version_output() -> str | None:
    try:
        return subprocess.check_output(
            [_server_binary(), "--version"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return None


def current_version_string() -> str | None:
    out = _raw_version_output()
    if out is None:
        return None
    m = _VERSION_RE.search(out)
    return m.group(1) if m else None


def check_version(min_version: int) -> bool:
    v = current_version_string()
    if v is None:
        return False
    if v.isdigit():
        return int(v) >= min_version
    # semver build (e.g. "0.4.0") supersedes any legacy build-number scheme
    return True