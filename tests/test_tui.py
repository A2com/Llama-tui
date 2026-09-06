import re
from pathlib import Path

import pytest
from datetime import datetime, timedelta
from src.tui_model import StatusModel, LogBuffer, ServerStatus


PROJECT_ROOT = Path(__file__).parent.parent


def test_status_model_initial_state():
    model = StatusModel()
    assert model.status == ServerStatus.STOPPED
    assert model.pid is None
    assert model.uptime_seconds == 0


def test_status_model_update_to_running():
    model = StatusModel()
    model.update(status=ServerStatus.RUNNING, pid=1234)
    assert model.status == ServerStatus.RUNNING
    assert model.pid == 1234


def test_status_model_update_to_stopped():
    model = StatusModel()
    model.update(status=ServerStatus.RUNNING, pid=1234)
    model.update(status=ServerStatus.STOPPED, pid=None)
    assert model.status == ServerStatus.STOPPED
    assert model.pid is None


def test_status_model_formats_uptime_seconds():
    model = StatusModel()
    model.update(status=ServerStatus.RUNNING, pid=1, uptime_seconds=45)
    assert model.uptime_str == "0h 00m 45s"


def test_status_model_formats_uptime_minutes():
    model = StatusModel()
    model.update(status=ServerStatus.RUNNING, pid=1, uptime_seconds=3725)
    assert model.uptime_str == "1h 02m 05s"


def test_status_model_uptime_zero_when_stopped():
    model = StatusModel()
    assert model.uptime_str == "--"


def test_status_model_is_running_property():
    model = StatusModel()
    assert model.is_running is False
    model.update(status=ServerStatus.RUNNING, pid=1)
    assert model.is_running is True


def test_log_buffer_starts_empty():
    buf = LogBuffer(max_lines=100)
    assert len(buf.lines) == 0


def test_log_buffer_adds_line():
    buf = LogBuffer(max_lines=100)
    buf.add("hello world")
    assert len(buf.lines) == 1
    assert buf.lines[0] == "hello world"


def test_log_buffer_respects_max_lines():
    buf = LogBuffer(max_lines=5)
    for i in range(10):
        buf.add(f"line {i}")
    assert len(buf.lines) == 5
    assert buf.lines[-1] == "line 9"


def test_log_buffer_drops_oldest_lines():
    buf = LogBuffer(max_lines=3)
    buf.add("first")
    buf.add("second")
    buf.add("third")
    buf.add("fourth")
    assert buf.lines[0] == "second"


def test_log_buffer_clear():
    buf = LogBuffer(max_lines=100)
    buf.add("a")
    buf.add("b")
    buf.clear()
    assert len(buf.lines) == 0


def test_tui_source_no_fast_proxy_label():
    """Le TUI ne doit plus mentionner 'fast proxy' — remplace par litellm."""
    tui_src = (PROJECT_ROOT / "src" / "tui.py").read_text()
    matches = re.findall(r"fast proxy", tui_src, re.IGNORECASE)
    assert matches == [], f"found 'fast proxy' in tui.py: {matches}"


def test_tui_theme_is_dark():
    """Le TUI doit utiliser un thème sombre, pas monokai flashy."""
    from src.tui import LlamaTUI
    app = LlamaTUI()
    assert app.theme in ("textual-dark", "css"), f"theme actuel : {app.theme}"


def test_tui_has_llama_monitor_action():
    """Le TUI doit exposer une action pour lancer llama-monitor."""
    from src.tui import LlamaTUI
    assert hasattr(LlamaTUI, "action_start_llama_monitor")


def test_tui_has_llama_monitor_binding():
    """Le TUI doit avoir un binding clavier pour llama-monitor."""
    from src.tui import LlamaTUI
    bindings = {b.key: b.action for b in LlamaTUI.BINDINGS}
    assert "m" in bindings
    assert bindings["m"] == "start_llama_monitor"


def test_llama_monitor_cmd_includes_models_dir():
    """La commande llama-monitor doit passer --models-dir pour la découverte des modèles."""
    from src.tui import LlamaTUI, MODELS_DIR
    app = LlamaTUI()
    cmd = app._build_llama_monitor_cmd("llama-server")
    assert "--models-dir" in cmd
    assert str(MODELS_DIR) in cmd


def test_tui_has_monitor_status_label():
    """Sidebar doit afficher un label de statut llama-monitor (port 7778)."""
    tui_src = (PROJECT_ROOT / "src" / "tui.py").read_text()
    assert "lbl-monitor-status" in tui_src, "lbl-monitor-status manquant"


def test_tui_has_llama_version_label():
    """Sidebar doit afficher la version de llama.cpp active."""
    tui_src = (PROJECT_ROOT / "src" / "tui.py").read_text()
    assert "lbl-llama-version" in tui_src, "lbl-llama-version manquant"


def test_tui_exposes_llama_version_attribute():
    """La version llama.cpp est calculée une fois à l'init (pas de subprocess dans la boucle de poll)."""
    from src.tui import LlamaTUI
    app = LlamaTUI()
    assert hasattr(app, "_llama_version")


def test_tui_reads_thinking_flag_for_proxy():
    """server.json['thinking'] doit être lu et transmis à ProxyManager (profil qualite)."""
    tui_src = (PROJECT_ROOT / "src" / "tui.py").read_text()
    assert '.get("thinking"' in tui_src, "lecture de la clé 'thinking' manquante"
    assert "thinking=" in tui_src, "thinking= non transmis à ProxyManager"


def test_tui_load_and_restart_also_restarts_running_proxy():
    """Changer de profil doit relancer le proxy s'il tournait (sinon ENABLE_THINKING reste celui du profil précédent)."""
    import inspect
    from src.tui import LlamaTUI
    src = inspect.getsource(LlamaTUI._load_and_restart)
    assert "_proxy" in src, "_load_and_restart ne touche pas au proxy"


def test_tui_has_download_screen():
    """DownloadScreen (ModalScreen) doit exister pour télécharger des modèles."""
    from textual.screen import ModalScreen
    from src.tui import DownloadScreen
    assert issubclass(DownloadScreen, ModalScreen), "DownloadScreen n'est pas un ModalScreen"


def test_tui_has_download_binding():
    """Touche n → action download_model."""
    from src.tui import LlamaTUI
    bindings = {b.key: b.action for b in LlamaTUI.BINDINGS}
    assert bindings.get("n") == "download_model"


def test_tui_has_download_action():
    from src.tui import LlamaTUI
    assert hasattr(LlamaTUI, "action_download_model")


def test_tui_has_proxy_backend_binding():
    """Touche b → switch_proxy_backend."""
    from src.tui import LlamaTUI
    bindings = {b.key: b.action for b in LlamaTUI.BINDINGS}
    assert bindings.get("b") == "switch_proxy_backend"


def test_tui_has_switch_proxy_backend_action():
    from src.tui import LlamaTUI
    assert hasattr(LlamaTUI, "action_switch_proxy_backend")


def test_tui_has_proxy_backend_label():
    """Sidebar doit afficher le backend proxy actif."""
    tui_src = (PROJECT_ROOT / "src" / "tui.py").read_text()
    assert "lbl-proxy-backend" in tui_src, "lbl-proxy-backend manquant"


def test_tui_right_panel_uses_tabbedcontent():
    """Le panneau droit doit utiliser TabbedContent (Stats/Slots/Logs)."""
    tui_src = (PROJECT_ROOT / "src" / "tui.py").read_text()
    assert "TabbedContent" in tui_src, "TabbedContent manquant"
    assert "TabPane" in tui_src, "TabPane manquant"


def test_tui_has_slots_tab():
    tui_src = (PROJECT_ROOT / "src" / "tui.py").read_text()
    assert "tab-slots" in tui_src, "onglet Slots manquant"
    assert "lbl-slots" in tui_src, "lbl-slots manquant"


def test_tui_has_cache_sparkline():
    tui_src = (PROJECT_ROOT / "src" / "tui.py").read_text()
    assert "sparkline-cache" in tui_src, "sparkline-cache manquant"


def test_stats_collector_has_cache_history():
    from src.stats_collector import StatsCollector
    sc = StatsCollector(port=8082)
    assert hasattr(sc, "cache_history"), "cache_history manquant dans StatsCollector"


def test_tui_has_task_profile_binding():
    """Touche t → action task_profile."""
    from src.tui import LlamaTUI
    bindings = {b.key: b.action for b in LlamaTUI.BINDINGS}
    assert bindings.get("t") == "task_profile"


def test_tui_has_task_profile_action():
    from src.tui import LlamaTUI
    assert hasattr(LlamaTUI, "action_task_profile")


def test_tui_has_task_profile_screen():
    from textual.screen import ModalScreen
    from src.tui import TaskProfileScreen
    assert issubclass(TaskProfileScreen, ModalScreen), "TaskProfileScreen n'est pas un ModalScreen"


def test_tui_task_profiles_file_constant():
    from src.tui import TASK_PROFILES_FILE, PROJECT_ROOT
    assert TASK_PROFILES_FILE == PROJECT_ROOT / "config" / "task-profiles.json"
