import os
import signal
import subprocess
from enum import Enum
from pathlib import Path

import httpx


class ProxyStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"


class ProxyManager:
    def __init__(self, config_file: Path, port: int = 8001,
                 backend: str = "litellm", server_port: int = 8082,
                 thinking: bool = False,
                 pid_file: Path = None, log_dir: Path = None):
        self._config_file = config_file
        self._port = port
        self._backend = backend
        self._server_port = server_port
        self._thinking = thinking
        self._process: subprocess.Popen | None = None
        project_root = config_file.parent.parent
        self.pid_file = pid_file or (project_root / "proxy.pid")
        self._log_dir = log_dir or (project_root / "logs")
        self._proxy_script = project_root / "src" / "fast_proxy.ts"

    def _port_responding(self) -> bool:
        try:
            resp = httpx.get(f"http://127.0.0.1:{self._port}/health", timeout=1.0)
            return resp.status_code < 500
        except Exception:
            return False

    def status(self) -> ProxyStatus:
        if self._process is not None and self._process.poll() is None:
            return ProxyStatus.RUNNING
        if self.pid_file.exists() and self._process is not None:
            self.pid_file.unlink()
        if self._port_responding():
            return ProxyStatus.RUNNING
        return ProxyStatus.STOPPED

    def start(self) -> int:
        if self.status() == ProxyStatus.RUNNING:
            raise RuntimeError("Proxy already running")

        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_handle = open(self._log_dir / "proxy.log", "a")

        if self._backend == "bun":
            cmd = ["bun", "run", str(self._proxy_script)]
            env = {
                **os.environ,
                "UPSTREAM": f"http://127.0.0.1:{self._server_port}/v1",
                "PORT": str(self._port),
                "ENABLE_THINKING": "1" if self._thinking else "0",
            }
        else:
            cmd = ["litellm", "--config", str(self._config_file), "--port", str(self._port)]
            env = None
        self._process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        self.pid_file.write_text(str(self._process.pid))
        return self._process.pid

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None
        if self.pid_file.exists():
            self.pid_file.unlink()

    def health_check(self) -> bool:
        if self.status() == ProxyStatus.STOPPED:
            return False
        try:
            resp = httpx.get(f"http://localhost:{self._port}/health", timeout=2.0)
            return resp.status_code < 500
        except Exception:
            return False

    def get_info(self) -> dict:
        running = self.status() == ProxyStatus.RUNNING
        pid = None
        if running:
            if self._process:
                pid = self._process.pid
            elif self.pid_file.exists():
                try:
                    pid = int(self.pid_file.read_text().strip())
                except ValueError:
                    pid = None
        return {
            "status": self.status().value,
            "pid": pid,
            "port": self._port,
            "config": str(self._config_file),
        }
