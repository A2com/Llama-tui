"""TUI de gestion llama-server + litellm proxy avec sélection de modèles."""

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, ListItem, ListView, Log, Sparkline, Static, Input, TabbedContent, TabPane, Digits, ProgressBar

from src.model_manager import ModelManager
from src.proxy_manager import ProxyManager, ProxyStatus
from src.server_manager import ServerManager, ServerStatus
from src.cline_config import sync_cline_models
from src.stats_collector import StatsCollector
from src.task_profiles import TaskProfileError, TaskProfileManager
from src.tui_model import LogBuffer, StatusModel
from src.version import current_version_string, litellm_version_string

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE  = Path(os.environ.get("LLAMA_TUI_CONFIG", PROJECT_ROOT / "config" / "server.json"))
PROXY_ROOT = Path(os.environ.get("LLAMA_TUI_LITELLM_CONFIG", PROJECT_ROOT / "config" / "litellm.yaml"))
MODELS_DIR   = Path(os.environ.get("LLAMA_TUI_MODELS_DIR", PROJECT_ROOT / "models"))
TASK_PROFILES_FILE = Path(os.environ.get("LLAMA_TUI_TASK_PROFILES", PROJECT_ROOT / "config" / "task-profiles.json"))
CLINE_SETTINGS_DIR = Path(os.environ.get("LLAMA_TUI_CLINE_DIR", Path.home() / ".cline" / "data" / "settings"))


DEFAULT_MODEL = "Qwen3.6-35B-A3B-MTP-UD-Q6_K_XL.gguf"

# Catppuccin Mocha — styles Rich inline (tags [running]/[badge-ok] inexistants côté Rich)
_CLR_OK = "#a6e3a1"
_CLR_KO = "#f38ba8"


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _ensure_config() -> None:
    """Crée config/server.json par défaut s'il manque, avec modèle MTP par défaut."""
    if CONFIG_FILE.exists():
        return
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH = MODELS_DIR / DEFAULT_MODEL
    default_model = f"models/{DEFAULT_MODEL}" if CONFIG_PATH.exists() else "models/default.gguf"
    CONFIG_FILE.write_text(json.dumps({
        "model": default_model,
        "host": "0.0.0.0",
        "port": 8082,
        "n_gpu_layers": 99,
        "ctx_size": 131072,
        "batch_size": 2048,
        "ubatch_size": 2048,
        "threads": 12,
        "flash_attn": True,
        "parallel": 1,
        "cont_batching": True,
        "cache_type_k": "q4_0",
        "cache_type_v": "q4_0",
        "jinja": True,
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 2,
        "proxy_backend": "bun",
    }, indent=2))


class HelpScreen(ModalScreen):
    """Overlay d'aide — touche ?."""

    BINDINGS = [Binding("escape,question_mark,q", "dismiss", "Fermer", show=True)]

    CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-card {
        width: 72;
        max-height: 80%;
        border: solid $primary;
        background: $surface;
        padding: 1 2;
    }
    #help-card .title { color: $primary; text-style: bold; }
    #help-card .row   { color: $text; }
    #help-card .key   { color: $warning; text-style: bold; width: 8; }
    #help-card .desc  { color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        rows = [
            ("s", "Démarre llama-server"),
            ("q", "Arrête llama-server"),
            ("r", "Redémarre llama-server"),
            ("l", "Charge le modèle sélectionné"),
            ("p", "Démarre litellm proxy"),
            ("o", "Arrête litellm proxy"),
            ("a", "Tout démarrer (server + proxy)"),
            ("z", "Tout arrêter"),
            ("c", "Vide les logs"),
            ("d", "Ouvre Llama WebUI"),
            ("n", "Télécharge un modèle (HF)"),
            ("b", "Bascule backend proxy (bun↔litellm)"),
            ("t", "Profil de tâche (vitesse/qualité/uncensored)"),
            ("Tab", "Focus liste modèles"),
            ("?", "Affiche cette aide"),
            ("Ctrl+Q", "Quitte"),
        ]
        with VerticalScroll(id="help-card"):
            yield Static("▪ Raccourcis clavier", classes="title")
            yield Static("─" * 40, classes="divider")
            for key, desc in rows:
                yield Static(f"[bold yellow]{key:<8}[/bold yellow] [dim]{desc}[/dim]")
            yield Static("─" * 40, classes="divider")
            yield Static("[dim]Esc / ? / q pour fermer[/dim]")


class DownloadScreen(ModalScreen):
    """Overlay de téléchargement de modèle depuis HuggingFace."""

    BINDINGS = [Binding("escape", "dismiss", "Fermer", show=True)]

    CSS = """
    DownloadScreen { align: center middle; }
    #download-card {
        width: 72; max-height: 80%;
        border: solid $primary; background: $surface; padding: 1 2;
    }
    #download-card .title { color: $primary; text-style: bold; }
    #download-card .field   { color: $text-muted; margin-top: 1; }
    #download-card Input { margin: 0 0 1 0; border: solid $surface-lighten-1; }
    #download-card Input:focus { border: solid $primary; }
    #dl-submit { width: 20; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="download-card"):
            yield Static("▪ Télécharger un modèle", classes="title")
            yield Static("─" * 40, classes="divider")
            yield Static("repo_id  (ex: Qwen/Qwen3-0.6B-GGUF)", classes="field")
            yield Input(placeholder="user/repo", id="dl-repo")
            yield Static("filename  (ex: qwen3-0.6b-q8_0.gguf)", classes="field")
            yield Input(placeholder="fichier.gguf", id="dl-filename")
            yield Static("local_filename  (optionnel)", classes="field")
            yield Input(placeholder="(= filename par défaut)", id="dl-local")
            yield Button("⬇ Télécharger", id="dl-submit", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "dl-submit":
            return
        repo = self.query_one("#dl-repo", Input).value.strip()
        fn = self.query_one("#dl-filename", Input).value.strip()
        local = self.query_one("#dl-local", Input).value.strip() or None
        if not repo or not fn:
            return
        self.dismiss((repo, fn, local))


class TaskProfileScreen(ModalScreen):
    """Overlay de sélection de profil de tâche — touche t."""

    BINDINGS = [Binding("escape", "dismiss", "Fermer", show=True)]

    CSS = """
    TaskProfileScreen { align: center middle; }
    #profile-card {
        width: 60; max-height: 80%;
        border: solid $primary; background: $surface; padding: 1 2;
    }
    #profile-card .title { color: $primary; text-style: bold; }
    ListItem.active-profile { color: $success; text-style: bold; }
    """

    def __init__(self, profiles, active_path: Path | None):
        super().__init__()
        self._profiles = profiles
        self._active_path = active_path

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="profile-card"):
            yield Static("▪ Profil de tâche", classes="title")
            yield Static("─" * 30, classes="divider")
            yield ListView(id="profile-list")
            yield Static("[dim]Entrée pour activer · Esc pour fermer[/dim]")

    def on_mount(self) -> None:
        lv = self.query_one("#profile-list", ListView)
        for profile in self._profiles:
            is_active = profile.model_path == self._active_path
            prefix = "● " if is_active else "  "
            item = ListItem(Label(f"{prefix}{profile.name}"))
            item.profile_name = profile.name
            if is_active:
                item.add_class("active-profile")
            lv.append(item)
        lv.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.profile_name)


class LlamaTUI(App):
    CSS = """
    Screen { layout: vertical; background: #1e1e2e; }

    /* ── Catppuccin Mocha ── */
    /* fond #1e1e2e · surface #313244 · vert #a6e3a1 · rouge #f38ba8
       orange #fab387 · bleu #89b4fa · texte #cdd6f4 · muted #9399b2 */

    /* ── Barre de statut globale ── */
    #status-bar {
        height: 1;
        background: #181825;
        color: #cdd6f4;
        padding: 0 1;
        text-style: bold;
    }
    #status-bar .sep         { color: #45475a; }

    #main  { height: 1fr; layout: horizontal; }

    /* ── Panneau modèles ── */
    #model-panel {
        width: 38;
        background: #181825;
        border: round #45475a;
        padding: 1 1;
    }
    #model-panel .section  { color: #89b4fa; text-style: bold; }
    #model-panel .divider  { color: #45475a; }
    #model-list { height: 1fr; }
    ListView { background: #181825; border: none; }
    ListItem { padding: 0 1; }
    ListItem.active-model { color: #a6e3a1; text-style: bold; }
    ListItem:focus { background: #313244; }
    ListView:focus > ListItem.--highlight { background: #45475a; }
    .model-quant { color: #fab387; }
    .model-meta  { color: #9399b2; }

    /* ── Sidebar statut ── */
    #sidebar {
        width: 34;
        background: #181825;
        border: round #45475a;
        padding: 1 2;
    }
    #sidebar Label { margin-bottom: 1; }

    /* ── Log ── */
    #log-panel {
        border: round #45475a;
        height: 1fr;
        padding: 0 1;
    }

    /* ── Action bar unique ── */
    #action-bar {
        height: auto;
        layout: horizontal;
        padding: 0 1;
        background: #181825;
    }
    Button {
        margin: 0 1;
        width: 18;
        height: 3;
        background: #313244;
        color: #cdd6f4;
    }
    Button:hover { background: #45475a; color: #ffffff; }
    Button:disabled { background: #181825; color: #585b70; }
    .Button--success   { color: #a6e3a1; }
    .Button--success:hover   { color: #c9f7bc; }
    .Button--success:disabled { color: #3e4a3a; }
    .Button--error     { color: #f38ba8; }
    .Button--error:hover   { color: #ff9dbb; }
    .Button--error:disabled  { color: #5a3a4a; }
    .Button--warning   { color: #fab387; }
    .Button--warning:hover   { color: #ffd0b3; }
    .Button--warning:disabled { color: #5a463a; }
    .Button--primary   { color: #89b4fa; }
    .Button--primary:hover   { color: #b3ccff; }
    .Button--primary:disabled { color: #3a446a; }

    .section  { color: #89b4fa; text-style: bold; }
    .divider  { color: #45475a; }
    .hint     { color: #9399b2; text-style: italic; }

    /* ── Panneau droit (onglets Stats/Slots/Logs) ── */
    #right-tabs { height: 1fr; }

    /* ── Stats enrichies ── */
    #stats-panel {
        border: round #45475a;
        height: auto;
        padding: 0 1;
    }
    #stats-grid { height: auto; }
    #stats-grid Label { margin-bottom: 0; }
    #lbl-tps { color: #fab387; text-style: bold; }
    #lbl-tps-secondary { color: #9399b2; }
    #sparkline-tps { height: 4; }
    #sparkline-tps > .sparkline--max-color { color: #89b4fa; }
    #sparkline-tps > .sparkline--min-color { color: #6c7086; }
    #sparkline-cache { height: 3; }
    #sparkline-cache > .sparkline--max-color { color: #a6e3a1; }
    #sparkline-cache > .sparkline--min-color { color: #6c7086; }
    #digits-tps {
        color: #a6e3a1;
        background: #181825;
        width: auto;
        text-style: bold;
    }
    #progress-ctx { color: #89b4fa; margin-top: 1; }
    #lbl-spark-tps { color: #89b4fa; text-style: bold; margin-top: 1; }
    #lbl-spark-cache { color: #a6e3a1; text-style: bold; margin-top: 1; }
    #tab-slots { padding: 0 1; }
    #lbl-slots { color: #cdd6f4; }
    """

    BINDINGS = [
        Binding("s",      "start_server",   "LLM start"),
        Binding("q",      "stop_server",    "LLM stop"),
        Binding("r",      "restart_server", "LLM restart"),
        Binding("l",      "load_model",     "Charger modèle"),
        Binding("p",      "start_proxy",    "Proxy start"),
        Binding("o",      "stop_proxy",     "Proxy stop"),
        Binding("a",      "start_all",      "Tout démarrer"),
        Binding("z",      "stop_all",       "Tout arrêter"),
        Binding("c",      "clear_logs",     "Vider logs"),
        Binding("d",      "open_activity",  "Llama WebUI"),
        Binding("n",      "download_model", "Télécharger"),
        Binding("b",      "switch_proxy_backend", "Proxy backend"),
        Binding("t",      "task_profile",   "Profil tâche"),
        Binding("?",      "help",           "Aide", show=True),
        Binding("tab",    "focus_models",   "Focus modèles", show=False),
        Binding("ctrl+q", "quit",           "Quitter"),
    ]

    def __init__(self):
        super().__init__()
        self.theme = "textual-dark"
        _ensure_config()
        self._manager       = ServerManager(CONFIG_FILE)
        try:
            _cfg = json.loads(CONFIG_FILE.read_text())
            self._proxy_backend = _cfg.get("proxy_backend", "bun")
            self._thinking = _cfg.get("thinking", False)
        except Exception:
            self._proxy_backend = "bun"
            self._thinking = False
        self._proxy         = ProxyManager(
            PROXY_ROOT, port=8001,
            backend=self._proxy_backend,
            server_port=self._manager._config.port,
            thinking=self._thinking,
        )
        self._model_mgr     = ModelManager(models_dir=MODELS_DIR, config_file=CONFIG_FILE)
        self._task_profile_mgr = TaskProfileManager(profiles_file=TASK_PROFILES_FILE, project_root=PROJECT_ROOT)
        self._status_model  = StatusModel()
        self._logs          = LogBuffer(max_lines=500)
        self._stats         = StatsCollector(port=self._manager._config.port)
        self._llama_version = current_version_string()
        self._litellm_version: str | None = None
        threading.Thread(target=self._load_litellm_version, daemon=True).start()
        self._server_start_time: float | None = None
        self._poll_thread: threading.Thread | None = None
        self._running       = True

    def _load_litellm_version(self) -> None:
        # litellm --version = ~2.3 s (imports) → thread, jamais dans la boucle de poll
        v = litellm_version_string()
        if v:
            self._litellm_version = v

    # ── Layout ────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Static("", id="status-bar")
        with Horizontal(id="main"):
            with Vertical(id="model-panel"):
                yield Static("▪ Modèles", classes="section")
                yield Static("─" * 30, classes="divider")
                yield ListView(id="model-list")
            with Vertical(id="sidebar"):
                yield Static("▪ llama-server", classes="section")
                yield Static("─" * 26, classes="divider")
                yield Label("", id="lbl-status")
                yield Label("", id="lbl-pid")
                yield Label("", id="lbl-uptime")
                yield Label("", id="lbl-host")
                yield Label("", id="lbl-model")
                yield Label("", id="lbl-ctx")
                yield Label("", id="lbl-gpu")
                yield Label("", id="lbl-health")
                yield Label("", id="lbl-llama-version")
                yield Static(" ", classes="divider")
                yield Static("▪ litellm proxy", classes="section")
                yield Static("─" * 26, classes="divider")
                yield Label("", id="lbl-proxy-status")
                yield Label("", id="lbl-proxy-pid")
                yield Label("", id="lbl-proxy-port")
                yield Label("", id="lbl-proxy-health")
                yield Label("", id="lbl-proxy-backend")
                yield Label("", id="lbl-litellm-version")
            with TabbedContent(id="right-tabs"):
                with TabPane("Stats", id="tab-stats"):
                    with Vertical(id="stats-panel"):
                        yield Static("▪ Performance", classes="section")
                        yield Digits("  —", id="digits-tps")
                        yield Label("", id="lbl-tps-secondary")
                        yield Static("⚡ tokens/s", classes="section", id="lbl-spark-tps")
                        yield Sparkline([], id="sparkline-tps", summary_function=max)
                        yield Static("Cache %", classes="section", id="lbl-spark-cache")
                        yield Sparkline([], id="sparkline-cache", summary_function=max)
                        yield ProgressBar(total=100.0, show_eta=False, id="progress-ctx")
                with TabPane("Slots", id="tab-slots"):
                    yield Static("▪ Slots", classes="section")
                    yield Static("─" * 40, classes="divider")
                    yield Label("", id="lbl-slots")
                with TabPane("Logs", id="tab-logs"):
                    yield Log(id="log-panel", highlight=True)
        with Horizontal(id="action-bar"):
            yield Button("▶ LLM [s]",    id="btn-start",     variant="success")
            yield Button("■ LLM [q]",    id="btn-stop",      variant="error")
            yield Button("↺ LLM [r]",    id="btn-restart",   variant="warning")
            yield Button("⏏ Load [l]",   id="btn-load",      variant="primary")
            yield Button("▶▶ Tout [a]",  id="btn-all-start", variant="success")
            yield Button("■■ Tout [z]",  id="btn-all-stop",  variant="error")
            yield Button("▶ Proxy [p]",  id="btn-proxy-start", variant="success")
            yield Button("■ Proxy [o]",  id="btn-proxy-stop",  variant="error")
            yield Button("✕ Logs [c]",   id="btn-clear")
            yield Button("🌐 WebUI [d]", id="btn-activity", variant="warning")
        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._populate_model_list()
        self._sync_cline()
        self._refresh_ui()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def on_unmount(self) -> None:
        self._running = False

    # ── Poll loop ─────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            status = self._manager.status()
            uptime = 0
            if status == ServerStatus.RUNNING and self._server_start_time:
                uptime = int(time.monotonic() - self._server_start_time)
            self._status_model.update(
                status=status,
                pid=self._manager.get_info()["pid"],
                uptime_seconds=uptime,
            )
            if status == ServerStatus.RUNNING:
                self._stats.poll()
            self.call_from_thread(self._refresh_ui)
            time.sleep(0.5)

    # ── Model list ────────────────────────────────────────────────────────

    def _populate_model_list(self) -> None:
        lv = self.query_one("#model-list", ListView)
        lv.clear()
        active = self._model_mgr.active_model
        models = self._model_mgr.scan()
        for info in models:
            is_active = info.path == active
            size_str = f"{info.size_gb:.1f}GB" if info.size_gb >= 0.1 else f"{info.size_gb*1024:.0f}MB"
            quant = f"  [bold yellow]{info.quant}[/bold yellow]" if info.quant else ""
            ctx = f"  [dim]ctx {info.ctx_train}[/dim]" if info.ctx_train else ""
            prefix = "● " if is_active else "  "
            item = ListItem(Label(f"{prefix}{info.name}{quant}\n  [{size_str}]  [dim]{info.mtime_date}[/dim]{ctx}"))
            if is_active:
                item.add_class("active-model")
            lv.append(item)
        if not models:
            lv.append(ListItem(Label("  Aucun modèle trouvé")))

    def _selected_model_path(self) -> Path | None:
        lv      = self.query_one("#model-list", ListView)
        idx     = lv.index
        models  = self._model_mgr.scan()
        if idx is None or idx >= len(models):
            return None
        return models[idx].path

    # ── UI refresh ────────────────────────────────────────────────────────

    def _refresh_ui(self) -> None:
        self._refresh_status_bar()
        self._refresh_server()
        self._refresh_proxy()
        self._refresh_stats()

    def _refresh_status_bar(self) -> None:
        info = self._manager.get_info()
        pinfo = self._proxy.get_info()
        srv_up = self._status_model.is_running
        proxy_up = pinfo["status"] == "running"
        model_name = Path(info["model"]).stem if info["model"] else "—"
        s = self._stats.last_stats
        tps = f"⚡ {s.gen_tps:.1f} t/s" if (s.is_generating and s.gen_tps) else "idle"
        badge = f"[bold {_CLR_OK}]● RUNNING[/]" if srv_up else f"[bold {_CLR_KO}]● STOPPED[/]"
        pbadge = f"[bold {_CLR_OK}]●[/]" if proxy_up else f"[bold {_CLR_KO}]○[/]"
        bar = (
            f"{badge}  [sep]│[/sep]  {model_name}  [sep]│[/sep]  {tps}  "
            f"[sep]│[/sep]  up {self._status_model.uptime_str}  [sep]│[/sep]  proxy {pbadge}"
        )
        self.query_one("#status-bar", Static).update(bar)

    def _refresh_server(self) -> None:
        info       = self._manager.get_info()
        is_running = self._status_model.is_running
        lbl        = f"[bold {_CLR_OK}]● RUNNING[/]" if is_running else f"[bold {_CLR_KO}]● STOPPED[/]"
        model_name = Path(info["model"]).stem if info["model"] else "—"
        health     = self._manager.health_check()

        self.query_one("#lbl-status", Label).update(
            f"[bold]Status:[/bold] {lbl}")
        self.query_one("#lbl-pid",    Label).update(
            f"[bold]PID:[/bold]    {info['pid'] or '—'}")
        self.query_one("#lbl-uptime", Label).update(
            f"[bold]Uptime:[/bold] {self._status_model.uptime_str}")
        self.query_one("#lbl-host",   Label).update(
            f"[bold]Écoute:[/bold] {info['host']}:{info['port']}")
        self.query_one("#lbl-model",  Label).update(
            f"[bold]Modèle:[/bold]\n  {model_name}")
        self.query_one("#lbl-ctx",    Label).update(
            f"[bold]Ctx:[/bold]    {info['ctx_size']:,} tokens")
        self.query_one("#lbl-gpu",    Label).update(
            f"[bold]GPU:[/bold]    {info['n_gpu_layers']} layers (Metal)")
        self.query_one("#lbl-health", Label).update(
            f"[bold]Health:[/bold] "
            f"[bold {_CLR_OK}]✓ OK[/]" if health else f"[bold]Health:[/bold] [bold {_CLR_KO}]✗ N/A[/]")
        self.query_one("#lbl-llama-version", Label).update(
            f"[bold]llama.cpp:[/bold] {self._llama_version or '—'}")

        self.query_one("#btn-start",   Button).disabled = is_running
        self.query_one("#btn-stop",    Button).disabled = not is_running
        self.query_one("#btn-restart", Button).disabled = not is_running

    def _refresh_proxy(self) -> None:
        info       = self._proxy.get_info()
        is_running = info["status"] == "running"
        lbl        = f"[bold {_CLR_OK}]● RUNNING[/]" if is_running else f"[bold {_CLR_KO}]● STOPPED[/]"
        health     = self._proxy.health_check()

        self.query_one("#lbl-proxy-status", Label).update(
            f"[bold]Status:[/bold] {lbl}")
        self.query_one("#lbl-proxy-pid",    Label).update(
            f"[bold]PID:[/bold]    {info['pid'] or '—'}")
        self.query_one("#lbl-proxy-port",   Label).update(
            f"[bold]Port:[/bold]   :{info['port']} → Claude Code")
        self.query_one("#lbl-proxy-health", Label).update(
            f"[bold]Health:[/bold] "
            f"[bold {_CLR_OK}]✓ OK[/]" if health else f"[bold]Health:[/bold] [bold {_CLR_KO}]✗ N/A[/]")
        self.query_one("#lbl-proxy-backend", Label).update(
            f"[bold]Backend:[/bold] {self._proxy_backend}")
        self.query_one("#lbl-litellm-version", Label).update(
            f"[bold]Version:[/bold] {self._litellm_version or '—'}")

        self.query_one("#btn-proxy-start", Button).disabled = is_running
        self.query_one("#btn-proxy-stop",  Button).disabled = not is_running

    def _refresh_stats(self) -> None:
        s = self._stats.last_stats
        history = list(self._stats.history)
        if s.is_generating and s.gen_tps is not None:
            tps_val = s.gen_tps
        elif history:
            tps_val = history[-1]
        else:
            tps_val = None
        digits = self.query_one("#digits-tps", Digits)
        digits.update(f"{tps_val:.1f}" if tps_val else " —")

        avg = f"{s.avg_tps:.1f}" if s.avg_tps else "—"
        peak = f"{s.peak_tps:.1f}" if s.peak_tps else "—"
        cache = f"{s.cache_hit_ratio*100:.0f}%" if s.cache_hit_ratio is not None else "—"
        ctxpct = f"{s.ctx_used_pct:.1f}%" if s.ctx_used_pct is not None else "—"
        secondary = (
            f"avg [bold]{avg}[/bold]  peak [bold]{peak}[/bold]  "
            f"cache [bold]{cache}[/bold]  ctx [bold]{ctxpct}[/bold]  "
            f"total [bold]{s.total_generated:,}[/bold]"
        )
        self.query_one("#lbl-tps-secondary", Label).update(secondary)
        self.query_one("#sparkline-tps", Sparkline).data = history or [0.0]
        cache_hist = list(self._stats.cache_history)
        self.query_one("#sparkline-cache", Sparkline).data = cache_hist or [0.0]

        progress = self.query_one("#progress-ctx", ProgressBar)
        if s.ctx_used_pct is not None:
            progress.update(total=100.0, progress=min(max(s.ctx_used_pct, 0.0), 100.0))
        else:
            progress.update(progress=0.0)

        proc = "● processing" if s.is_generating else "○ idle"
        cache_pct = f"{s.cache_hit_ratio*100:.0f}%" if s.cache_hit_ratio is not None else "—"
        self.query_one("#lbl-slots", Label).update(
            f"[bold]État:[/bold] {proc}\n"
            f"[bold]Prompt:[/bold] {s.prompt_tokens:,}  [bold]Cache:[/bold] {s.prompt_cached:,} ({cache_pct})\n"
            f"[bold]Ctx:[/bold] {s.n_ctx:,}  [bold]Généré total:[/bold] {s.total_generated:,}")

    # ── Log helper ────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self._logs.add(msg)
        self.query_one("#log-panel", Log).write_line(msg)

    # ── Actions ───────────────────────────────────────────────────────────

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_load_model(self) -> None:
        path = self._selected_model_path()
        if path is None:
            self._log(f"[{self._ts()}] ⚠ Aucun modèle sélectionné")
            return
        if path == self._model_mgr.active_model:
            self._log(f"[{self._ts()}] ⚠ Modèle déjà actif")
            return
        self._load_and_restart(path)

    def _load_and_restart(self, path: Path) -> None:
        was_running = self._manager.status() == ServerStatus.RUNNING
        proxy_was_running = self._proxy.status() == ProxyStatus.RUNNING
        self._log(f"[{self._ts()}] ⏏ Chargement → {path.stem}")

        if was_running:
            self._manager.stop()
            self._server_start_time = None
            self._log(f"[{self._ts()}] ■ Serveur arrêté pour changement de modèle")
        if proxy_was_running:
            self._proxy.stop()

        try:
            self._model_mgr.load(path)
        except Exception as e:
            self._log(f"[{self._ts()}] ✗ Chargement impossible : {e}")
            self._log(f"[{self._ts()}] ⤴ Ancienne config conservée — serveur NON relancé")
            self._refresh_ui()
            return
        self._manager = ServerManager(CONFIG_FILE)
        self._stats = StatsCollector(port=self._manager._config.port)
        try:
            self._thinking = json.loads(CONFIG_FILE.read_text()).get("thinking", False)
        except Exception:
            self._thinking = False
        self._proxy = ProxyManager(
            PROXY_ROOT, port=8001,
            backend=self._proxy_backend,
            server_port=self._manager._config.port,
            thinking=self._thinking,
        )
        self._populate_model_list()
        self._sync_cline()
        self._log(f"[{self._ts()}] ✓ Modèle sélectionné : {path.stem}")

        if was_running:
            self.action_start_server()
        if proxy_was_running:
            self.action_start_proxy()

        self._refresh_ui()

    def _sync_cline(self) -> None:
        try:
            sync_cline_models(
                TASK_PROFILES_FILE, PROJECT_ROOT, CLINE_SETTINGS_DIR,
                active_model=self._model_mgr.active_model,
                base_url=f"http://localhost:{self._manager._config.port}/v1",
            )
        except Exception as e:
            self._log(f"[{self._ts()}] ⚠ Sync Cline : {e}")

    def action_task_profile(self) -> None:
        try:
            profiles = self._task_profile_mgr.list_profiles()
        except TaskProfileError as e:
            self._log(f"[{self._ts()}] ✗ Profils de tâche : {e}")
            return
        self.push_screen(
            TaskProfileScreen(profiles, active_path=self._model_mgr.active_model),
            self._on_task_profile_result,
        )

    def _on_task_profile_result(self, name: str | None) -> None:
        if name is None:
            return
        try:
            profiles = self._task_profile_mgr.list_profiles()
            path = next(p.model_path for p in profiles if p.name == name)
        except (TaskProfileError, StopIteration) as e:
            self._log(f"[{self._ts()}] ✗ Profil '{name}' : {e}")
            return
        if path == self._model_mgr.active_model:
            self._log(f"[{self._ts()}] ⚠ Profil '{name}' déjà actif")
            return
        self._load_and_restart(path)

    def action_focus_models(self) -> None:
        self.query_one("#model-list", ListView).focus()

    def action_start_server(self) -> None:
        try:
            pid = self._manager.start()
            self._server_start_time = time.monotonic()
            self._llama_version = current_version_string()
            self._log(f"[{self._ts()}] ▶ llama-server démarré (PID {pid})")
            self._refresh_ui()
        except RuntimeError as e:
            self._log(f"[{self._ts()}] ⚠ {e}")
        except Exception as e:
            self._log(f"[{self._ts()}] ✗ Erreur: {e}")

    def action_stop_server(self) -> None:
        self._manager.stop()
        self._server_start_time = None
        self._log(f"[{self._ts()}] ■ llama-server arrêté")
        self._refresh_ui()

    def action_restart_server(self) -> None:
        self.action_stop_server()
        time.sleep(1)
        self.action_start_server()

    def action_start_proxy(self) -> None:
        try:
            pid = self._proxy.start()
            self._log(f"[{self._ts()}] ▶ litellm proxy démarré (PID {pid})")
            self._refresh_ui()
        except RuntimeError as e:
            self._log(f"[{self._ts()}] ⚠ {e}")
        except Exception as e:
            self._log(f"[{self._ts()}] ✗ Erreur proxy: {e}")

    def action_stop_proxy(self) -> None:
        self._proxy.stop()
        self._log(f"[{self._ts()}] ■ litellm proxy arrêté")
        self._refresh_ui()

    def action_switch_proxy_backend(self) -> None:
        was_running = self._proxy.status() == ProxyStatus.RUNNING
        new_backend = "litellm" if self._proxy_backend == "bun" else "bun"
        if was_running:
            self._proxy.stop()
            self._log(f"[{self._ts()}] ■ proxy arrêté pour switch backend")
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            cfg["proxy_backend"] = new_backend
            CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
        except Exception as e:
            self._log(f"[{self._ts()}] ⚠ persist backend: {e}")
        self._proxy_backend = new_backend
        self._proxy = ProxyManager(
            PROXY_ROOT, port=8001,
            backend=new_backend, server_port=self._manager._config.port,
            thinking=self._thinking)
        self._log(f"[{self._ts()}] ⤢ Proxy backend → {new_backend}")
        if was_running:
            self.action_start_proxy()
        self._refresh_ui()

    def action_start_all(self) -> None:
        self.action_start_server()
        self.action_start_proxy()

    def action_stop_all(self) -> None:
        self.action_stop_server()
        self.action_stop_proxy()

    def action_clear_logs(self) -> None:
        self._logs.clear()
        self.query_one("#log-panel", Log).clear()

    def action_open_activity(self) -> None:
        subprocess.Popen(["open", f"http://localhost:{self._manager._config.port}"])
        self._log(f"[{self._ts()}] 📊 Llama WebUI → http://localhost:{self._manager._config.port}")

    def action_download_model(self) -> None:
        self.push_screen(DownloadScreen(), self._on_download_result)

    def _on_download_result(self, result) -> None:
        if result is None:
            return
        repo, fn, local = result
        threading.Thread(
            target=self._do_download, args=(repo, fn, local), daemon=True
        ).start()

    def _do_download(self, repo: str, fn: str, local: str | None) -> None:
        self.call_from_thread(self._log, f"[{self._ts()}] ⬇ Téléchargement {repo}/{fn}")
        try:
            info = self._model_mgr.download(repo, fn, local)
            self.call_from_thread(
                self._log, f"[{self._ts()}] ✓ {info.name} téléchargé ({info.size_gb:.1f}GB)")
            self.call_from_thread(self._populate_model_list)
        except Exception as e:
            self.call_from_thread(self._log, f"[{self._ts()}] ✗ Download: {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "btn-start":       self.action_start_server,
            "btn-stop":        self.action_stop_server,
            "btn-restart":     self.action_restart_server,
            "btn-load":        self.action_load_model,
            "btn-all-start":   self.action_start_all,
            "btn-all-stop":    self.action_stop_all,
            "btn-proxy-start": self.action_start_proxy,
            "btn-proxy-stop":  self.action_stop_proxy,
            "btn-clear":       self.action_clear_logs,
            "btn-activity":    self.action_open_activity,
        }
        if event.button.id in actions:
            actions[event.button.id]()

    @staticmethod
    def _ts() -> str:
        return time.strftime("%H:%M:%S")