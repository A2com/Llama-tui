from collections import deque
from src.server_manager import ServerStatus


class StatusModel:
    def __init__(self):
        self.status = ServerStatus.STOPPED
        self.pid: int | None = None
        self.uptime_seconds: int = 0

    def update(self, status: ServerStatus, pid: int | None = None, uptime_seconds: int = 0):
        self.status = status
        self.pid = pid
        self.uptime_seconds = uptime_seconds

    @property
    def is_running(self) -> bool:
        return self.status == ServerStatus.RUNNING

    @property
    def uptime_str(self) -> str:
        if not self.is_running:
            return "--"
        h = self.uptime_seconds // 3600
        m = (self.uptime_seconds % 3600) // 60
        s = self.uptime_seconds % 60
        return f"{h}h {m:02d}m {s:02d}s"


class LogBuffer:
    def __init__(self, max_lines: int = 500):
        self._buffer: deque[str] = deque(maxlen=max_lines)

    def add(self, line: str) -> None:
        self._buffer.append(line)

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def lines(self) -> list[str]:
        return list(self._buffer)
