import os
import signal
import socket
import subprocess
from enum import Enum
from pathlib import Path

import httpx

from src.config import ServerConfig


class ServerStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"


class ServerManager:
    def __init__(self, config_file: Path, pid_file: Path = None, log_dir: Path = None):
        self._config = ServerConfig.from_file(config_file)
        self._process: subprocess.Popen | None = None
        project_root = config_file.parent
        self.pid_file = pid_file or (project_root / "llama-server.pid")
        self._log_dir = log_dir or (project_root / "logs")

    # ── Port helpers ───────────────────────────────────────────────────────

    def _port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", self._config.port)) == 0

    def _pid_on_port(self) -> int | None:
        """Retourne le PID du process écoutant sur le port, via lsof."""
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f":{self._config.port}"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            return int(out.splitlines()[0]) if out else None
        except Exception:
            return None

    def _pid_is_ours(self, pid: int) -> bool:
        """True si pid est un llama-server vivant (pas un PID recyclé par un autre process)."""
        try:
            os.kill(pid, 0)
            comm = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "comm="],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:  # ProcessLookupError, PermissionError, ps en échec
            return False
        return "llama-server" in comm or Path(comm).name == Path(getattr(self._config, "server_bin", None) or "llama-server").name

    # ── Status ─────────────────────────────────────────────────────────────

    def status(self) -> ServerStatus:
        # Process qu'on a lancé toujours actif
        if self._process is not None and self._process.poll() is None:
            return ServerStatus.RUNNING
        # Process lancé par une session précédente (pidfile)
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
            except ValueError:
                pid = None
            if pid is not None and self._pid_is_ours(pid):
                return ServerStatus.RUNNING
            self.pid_file.unlink(missing_ok=True)  # pidfile périmé
        # Détection par port (llama-server lancé manuellement)
        if self._port_in_use():
            return ServerStatus.RUNNING
        return ServerStatus.STOPPED

    # ── Start ──────────────────────────────────────────────────────────────

    def start(self) -> int:
        if self.status() == ServerStatus.RUNNING:
            raise RuntimeError("Server already running")

        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_handle = open(self._log_dir / "server.log", "a")

        # Priorité : env var (tests side-by-side) > server.json (persisté, indépendant
        # de l'env du process TUI) > PATH
        binary = os.environ.get("LLAMA_TUI_SERVER_BIN") or getattr(self._config, "server_bin", None) or "llama-server"
        cmd = [binary] + self._config.to_cli_args()
        self._process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        self.pid_file.write_text(str(self._process.pid))
        return self._process.pid

    # ── Stop ───────────────────────────────────────────────────────────────

    def stop(self) -> None:
        # Tuer notre process géré
        if self._process is not None:
            if self._process.poll() is None:
                self._process.send_signal(signal.SIGTERM)
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None

        # Tuer via pidfile (session précédente)
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                if self._pid_is_ours(pid):
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
            self.pid_file.unlink(missing_ok=True)

        # Tuer via port (lancé manuellement)
        if self._port_in_use():
            pid = self._pid_on_port()
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass

    # ── Health & info ──────────────────────────────────────────────────────

    def health_check(self) -> bool:
        if self.status() == ServerStatus.STOPPED:
            return False
        try:
            resp = httpx.get(
                f"http://127.0.0.1:{self._config.port}/health",
                timeout=2.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_info(self) -> dict:
        running = self.status() == ServerStatus.RUNNING
        pid = None
        if running:
            if self._process:
                pid = self._process.pid
            elif self.pid_file.exists():
                try:
                    pid = int(self.pid_file.read_text().strip())
                except ValueError:
                    pass
            if pid is None:
                pid = self._pid_on_port()
        return {
            "status": self.status().value,
            "pid": pid,
            "host": self._config.host,
            "port": self._config.port,
            "model": self._config.model,
            "ctx_size": self._config.ctx_size,
            "n_gpu_layers": self._config.n_gpu_layers,
            "parallel": self._config.parallel,
            "flash_attn": self._config.flash_attn,
        }
