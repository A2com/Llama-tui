"""Garde-fou WebUI : le binaire llama-server embarque-t-il l'interface web ?

Le bottle Homebrew est compilé sans LLAMA_BUILD_UI → GET / renvoie 404.
Ce test échoue tant que le binaire actif n'a pas d'UI embarquée.
"""
from pathlib import Path
import os
import socket

import httpx
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PORT = 8082


def _server_bin() -> str:
    return os.environ.get("LLAMA_TUI_SERVER_BIN", "llama-server")


def _server_port() -> int:
    try:
        import json
        cfg = json.loads((PROJECT_ROOT / "config" / "server.json").read_text())
        return int(cfg.get("port", DEFAULT_PORT))
    except Exception:
        return DEFAULT_PORT


def _llama_server_reachable(port: int) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def test_webui_assets_served():
    """GET / sur llama-server doit servir la WebUI (200), pas 404 File Not Found."""
    port = _server_port()
    if not _llama_server_reachable(port):
        pytest.skip("aucun llama-server en cours — test d'asset UI requiert un serveur actif")
    r = httpx.get(f"http://127.0.0.1:{port}/", timeout=5.0)
    assert r.status_code == 200, (
        f"GET / → {r.status_code} : WebUI absente du binaire {_server_bin()} "
        f"(build sans LLAMA_BUILD_UI ?)")
    assert "html" in r.headers.get("content-type", "").lower()