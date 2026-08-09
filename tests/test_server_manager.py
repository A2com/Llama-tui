import json
import time
import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.server_manager import ServerManager, ServerStatus


PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def config_data():
    return {
        "model": str(PROJECT_ROOT / "models" / "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"),
        "host": "0.0.0.0",
        "port": 8080,
        "n_gpu_layers": 99,
        "ctx_size": 32768,
        "batch_size": 2048,
        "ubatch_size": 512,
        "threads": 12,
        "flash_attn": True,
        "parallel": 4,
        "cont_batching": True,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
    }


@pytest.fixture
def config_file(tmp_path, config_data):
    cfg = tmp_path / "server.json"
    cfg.write_text(json.dumps(config_data))
    return cfg


@pytest.fixture(autouse=True)
def mock_port_methods(monkeypatch):
    monkeypatch.setattr(ServerManager, "_port_in_use", lambda self: False)
    monkeypatch.setattr(ServerManager, "_pid_on_port", lambda self: None)


@pytest.fixture
def manager(tmp_path, config_file):
    return ServerManager(config_file, pid_file=tmp_path / "llama.pid", log_dir=tmp_path)


def test_initial_status_is_stopped(manager):
    assert manager.status() == ServerStatus.STOPPED


def test_start_creates_pidfile(manager):
    fake_proc = MagicMock()
    fake_proc.pid = 12345
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc):
        manager.start()
    assert manager.pid_file.exists()
    assert manager.pid_file.read_text().strip() == "12345"


def test_start_returns_pid(manager):
    fake_proc = MagicMock()
    fake_proc.pid = 12345
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc):
        pid = manager.start()
    assert pid == 12345


def test_start_uses_correct_binary(manager):
    fake_proc = MagicMock()
    fake_proc.pid = 1
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc) as mock_popen:
        manager.start()
    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "llama-server"


def test_start_uses_env_binary(manager, monkeypatch):
    monkeypatch.setenv("LLAMA_TUI_SERVER_BIN", "/custom/llama-server")
    fake_proc = MagicMock()
    fake_proc.pid = 1
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc) as mock_popen:
        manager.start()
    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "/custom/llama-server"


def test_status_is_running_after_start(manager):
    fake_proc = MagicMock()
    fake_proc.pid = 99
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc):
        manager.start()
    assert manager.status() == ServerStatus.RUNNING


def test_stop_removes_pidfile(manager):
    fake_proc = MagicMock()
    fake_proc.pid = 99
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc):
        manager.start()
    manager.stop()
    assert not manager.pid_file.exists()


def test_status_is_stopped_after_stop(manager):
    fake_proc = MagicMock()
    fake_proc.pid = 99
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc):
        manager.start()
    manager.stop()
    assert manager.status() == ServerStatus.STOPPED


def test_start_raises_if_already_running(manager):
    fake_proc = MagicMock()
    fake_proc.pid = 99
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc):
        manager.start()
    with pytest.raises(RuntimeError, match="already running"):
        manager.start()


def test_stop_is_noop_when_not_running(manager):
    manager.stop()  # should not raise


def test_get_info_returns_expected_keys(manager):
    info = manager.get_info()
    assert "status" in info
    assert "pid" in info
    assert "port" in info
    assert "host" in info
    assert "model" in info
    assert "ctx_size" in info
    assert "n_gpu_layers" in info


def test_get_info_pid_is_none_when_stopped(manager):
    info = manager.get_info()
    assert info["pid"] is None


def test_health_check_false_when_stopped(manager):
    assert manager.health_check() is False


def test_health_check_uses_correct_port(manager):
    fake_proc = MagicMock()
    fake_proc.pid = 99
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc):
        manager.start()
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        result = manager.health_check()
    mock_get.assert_called_once()
    call_url = mock_get.call_args[0][0]
    assert "8080" in call_url
    assert result is True


def test_health_check_false_on_http_error(manager):
    fake_proc = MagicMock()
    fake_proc.pid = 99
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc):
        manager.start()
    with patch("httpx.get", side_effect=Exception("connection refused")):
        assert manager.health_check() is False


def test_health_check_false_on_500(manager):
    fake_proc = MagicMock()
    fake_proc.pid = 99
    fake_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake_proc):
        manager.start()
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 500
        assert manager.health_check() is False
