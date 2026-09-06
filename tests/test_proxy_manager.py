import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.proxy_manager import ProxyManager, ProxyStatus


PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def no_real_network():
    """Isole tous les tests du réseau reel — _port_responding() retourne False par defaut."""
    with patch("httpx.get", side_effect=Exception("no network in tests")):
        yield


@pytest.fixture
def manager(tmp_path):
    config = PROJECT_ROOT / "config" / "litellm.yaml"
    return ProxyManager(
        config_file=config,
        port=8001,
        pid_file=tmp_path / "litellm.pid",
        log_dir=tmp_path,
    )


def test_initial_status_is_stopped(manager):
    assert manager.status() == ProxyStatus.STOPPED


def test_start_returns_pid(manager):
    fake = MagicMock()
    fake.pid = 4242
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake):
        pid = manager.start()
    assert pid == 4242


def test_start_creates_pidfile(manager):
    fake = MagicMock()
    fake.pid = 4242
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake):
        manager.start()
    assert manager.pid_file.exists()
    assert manager.pid_file.read_text().strip() == "4242"


def test_start_uses_litellm_binary(manager):
    fake = MagicMock()
    fake.pid = 1
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake) as mock_popen:
        manager.start()
    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "litellm"


def test_start_passes_config_and_port(manager):
    fake = MagicMock()
    fake.pid = 1
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake) as mock_popen:
        manager.start()
    cmd = mock_popen.call_args[0][0]
    assert "--config" in cmd
    assert "--port" in cmd
    assert "8001" in cmd


# ── Backend Bun (fast_proxy.ts) ──────────────────────────────────────────────

def test_start_uses_bun_binary_when_backend_bun(tmp_path):
    pm = ProxyManager(
        config_file=PROJECT_ROOT / "config" / "litellm.yaml",
        port=8001, backend="bun", server_port=8082,
        pid_file=tmp_path / "bun.pid", log_dir=tmp_path,
    )
    fake = MagicMock()
    fake.pid = 7
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake) as mock_popen:
        pm.start()
    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "bun"
    assert "run" in cmd
    assert str(pm._proxy_script) in " ".join(cmd)


def test_start_bun_passes_env_upstream_port(tmp_path):
    pm = ProxyManager(
        config_file=PROJECT_ROOT / "config" / "litellm.yaml",
        port=8001, backend="bun", server_port=8082,
        pid_file=tmp_path / "bun.pid", log_dir=tmp_path,
    )
    fake = MagicMock()
    fake.pid = 7
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake) as mock_popen:
        pm.start()
    env = mock_popen.call_args.kwargs.get("env") or mock_popen.call_args[1].get("env")
    assert env is not None, "bun backend doit passer un env avec UPSTREAM/PORT"
    assert env["UPSTREAM"] == "http://127.0.0.1:8082/v1"
    assert env["PORT"] == "8001"


def test_start_bun_thinking_true_sets_enable_thinking_1(tmp_path):
    pm = ProxyManager(
        config_file=PROJECT_ROOT / "config" / "litellm.yaml",
        port=8001, backend="bun", server_port=8082, thinking=True,
        pid_file=tmp_path / "bun.pid", log_dir=tmp_path,
    )
    fake = MagicMock()
    fake.pid = 7
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake) as mock_popen:
        pm.start()
    env = mock_popen.call_args.kwargs.get("env") or mock_popen.call_args[1].get("env")
    assert env["ENABLE_THINKING"] == "1"


def test_start_bun_thinking_false_by_default(tmp_path):
    pm = ProxyManager(
        config_file=PROJECT_ROOT / "config" / "litellm.yaml",
        port=8001, backend="bun", server_port=8082,
        pid_file=tmp_path / "bun.pid", log_dir=tmp_path,
    )
    fake = MagicMock()
    fake.pid = 7
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake) as mock_popen:
        pm.start()
    env = mock_popen.call_args.kwargs.get("env") or mock_popen.call_args[1].get("env")
    assert env["ENABLE_THINKING"] == "0"


def test_backend_default_is_litellm(tmp_path):
    pm = ProxyManager(
        config_file=PROJECT_ROOT / "config" / "litellm.yaml",
        port=8001,
        pid_file=tmp_path / "p.pid", log_dir=tmp_path,
    )
    assert pm._backend == "litellm"


def test_status_running_after_start(manager):
    fake = MagicMock()
    fake.pid = 1
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake):
        manager.start()
    assert manager.status() == ProxyStatus.RUNNING


def test_stop_removes_pidfile(manager):
    fake = MagicMock()
    fake.pid = 1
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake):
        manager.start()
    manager.stop()
    assert not manager.pid_file.exists()


def test_status_stopped_after_stop(manager):
    fake = MagicMock()
    fake.pid = 1
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake):
        manager.start()
    manager.stop()
    assert manager.status() == ProxyStatus.STOPPED


def test_start_raises_if_already_running(manager):
    fake = MagicMock()
    fake.pid = 1
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake):
        manager.start()
    with pytest.raises(RuntimeError, match="already running"):
        manager.start()


def test_stop_noop_when_not_running(manager):
    manager.stop()


def test_health_check_false_when_stopped(manager):
    assert manager.health_check() is False


def test_health_check_true_when_running_and_ok(manager):
    fake = MagicMock()
    fake.pid = 1
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake):
        manager.start()
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        assert manager.health_check() is True


def test_health_check_false_on_error(manager):
    fake = MagicMock()
    fake.pid = 1
    fake.poll.return_value = None
    with patch("subprocess.Popen", return_value=fake):
        manager.start()
    with patch("httpx.get", side_effect=Exception("refused")):
        assert manager.health_check() is False


def test_get_info_keys(manager):
    info = manager.get_info()
    assert "status" in info
    assert "pid" in info
    assert "port" in info
    assert "config" in info


# ── Bug : port occupe par un processus externe (start-proxy.sh) ──────────────

def test_status_running_when_external_process_holds_port(manager):
    """status() doit retourner RUNNING si le port repond, meme sans self._process."""
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        assert manager.status() == ProxyStatus.RUNNING


def test_start_raises_when_external_process_holds_port(manager):
    """start() ne doit pas lancer un second LiteLLM si le port est deja pris."""
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        with pytest.raises(RuntimeError, match="already running"):
            manager.start()


# ── Bug : localhost IPv6 ambiguïté ───────────────────────────────────────────

def test_port_responding_uses_127_not_localhost(manager):
    """_port_responding() doit utiliser 127.0.0.1, pas localhost (IPv6 ambiguïté macOS)."""
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        manager._port_responding()
    url = mock_get.call_args[0][0]
    assert "127.0.0.1" in url, f"URL utilise localhost au lieu de 127.0.0.1 : {url}"


# ── Bug : status() ne doit pas effacer le pidfile d'un process externe ───────

def test_status_does_not_delete_pidfile_for_external_process(manager):
    """Si le proxy tourne (port repond) mais _process est None, ne pas effacer le pidfile."""
    manager.pid_file.write_text("12345")
    with patch("httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        manager.status()
    assert manager.pid_file.exists()  # NE DOIT PAS etre supprime
