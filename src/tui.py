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
from textual.widgets import Button, Footer, Label, ListItem, ListView, Log, Sparkline, Static, Input

from src.model_manager import ModelManager
from src.proxy_manager import ProxyManager, ProxyStatus
from src.server_manager import ServerManager, ServerStatus
from src.stats_collector import StatsCollector
from src.tui_model import LogBuffer, StatusModel

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE  = Path(os.environ.get("LLAMA_TUI_CONFIG", PROJECT_ROOT / "config" / "server.json"))
PROXY_ROOT = Path(os.environ.get("LLAMA_TUI_LITELLM_CONFIG", PROJECT_ROOT / "config" / "litellm.yaml"))
MODELS_DIR   = Path(os.environ.get("LLAMA_TUI_MODELS_DIR", PROJECT_ROOT / "models"))
LLAMA_MONITOR_BIN = Path(os.environ.get("LLAMA_TUI_MONITOR_BIN", Path.home() / "llama-monitor" / "target" / "release" / "llama-monitor"))
LLAMA_MONITOR_PORT = int(os.environ.get("LLAMA_TUI_MONITOR_PORT", 7778))
LLAMA_MONITOR_PRESETS = Path(os.environ.get("LLAMA_TUI_MONITOR_PRESETS", PROJECT_ROOT / "config" / "llama-monitor-presets.json"))


DEFAULT_MODEL = "Qwen3.6-35B-A3B-MTP-UD-Q6_K_XL.gguf"


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
            ("m", "Démarre llama-monitor"),
            ("n", "Télécharge un modèle (HF)"),
            ("/", "Filtre les modèles par nom"),
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


class LlamaTUI(App):
    CSS = """
    Screen { layout: vertical; }

    /* ── Barre de statut globale ── */
    #status-bar {
        height: 1;
        background: $surface-darken-2;
        color: $text;
        padding: 0 1;
        text-style: bold;
    }
    #status-bar .badge-ok    { color: $success; }
    #status-bar .badge-ko    { color: $error; }
    #status-bar .sep         { color: $surface-lighten-1; }

    #main  { height: 1fr; layout: horizontal; }

    /* ── Panneau modèles ── */
    #model-panel {
        width: 38;
        border: solid $surface-lighten-1;
        padding: 1 1;
    }
    #model-panel .section  { color: $text; text-style: bold; }
    #model-panel .divider  { color: $surface-lighten-1; }
    #model-filter {
        height: 1;
        margin: 0 0 1 0;
        border: solid $surface-lighten-1;
        padding: 0 1;
        color: $text-muted;
    }
    #model-filter:focus { border: solid $primary; color: $text; }
    #model-list { height: 1fr; }
    ListView { background: $surface; border: none; }
    ListItem { padding: 0 1; }
    ListItem.active-model { color: $success; text-style: bold; }
    ListItem:focus { background: $surface-lighten-2 30%; }
    ListView:focus > ListItem.--highlight { background: $surface-lighten-2 40%; }
    .model-quant { color: $warning; }
    .model-meta  { color: $text-muted; }

    /* ── Sidebar statut ── */
    #sidebar {
        width: 34;
        border: solid $surface-lighten-1;
        padding: 1 2;
    }
    #sidebar Label { margin-bottom: 1; }

    /* ── Log ── */
    #log-panel {
        border: solid $surface-lighten-1;
        height: 1fr;
        padding: 0 1;
    }

    /* ── Barres de boutons ── */
    #controls, #proxy-controls {
        height: auto;
        layout: horizontal;
        padding: 0 1;
    }
    #controls { background: $surface; }
    #proxy-controls { background: $surface-darken-1; }
    Button {
        margin: 0 1;
        width: 18;
        height: 3;
        background: #2f3c42;
        color: #d0d8dc;
    }
    Button:hover { background: #3a4a52; color: #ffffff; }
    Button:disabled { background: #1a2226; color: #5a6a70; }
    .Button--success   { color: #7ec883; }
    .Button--success:hover   { color: #a0e8a3; }
    .Button--success:disabled { color: #3a4a3a; }
    .Button--error     { color: #e06c75; }
    .Button--error:hover     { color: #ff8a91; }
    .Button--error:disabled  { color: #4a2a2e; }
    .Button--warning   { color: #e5c07b; }
    .Button--warning:hover   { color: #ffe08a; }
    .Button--warning:disabled { color: #4a3e2e; }
    .Button--primary   { color: #61afef; }
    .Button--primary:hover   { color: #8ac4ff; }
    .Button--primary:disabled { color: #2e3a4a; }

    .running  { color: $success; }
    .stopped  { color: $error; }
    .section  { color: $text; text-style: bold; }
    .divider  { color: $surface-lighten-1; }
    .hint     { color: $text-muted; text-style: italic; }

    /* ── Panneau droit (stats + log) ── */
    #right-panel { height: 1fr; }

    /* ── Stats enrichies ── */
    #stats-panel {
        border: solid $surface-lighten-1;
        height: 7;
        padding: 0 1;
    }
    #stats-grid { height: auto; }
    #stats-grid Label { margin-bottom: 0; }
    #lbl-tps { color: $warning; text-style: bold; }
    #lbl-tps-secondary { color: $text-muted; }
    #sparkline-tps { height: 2; }
    #sparkline-tps > .sparkline--max-color { color: $warning; }
    #sparkline-tps > .sparkline--min-color { color: $warning-darken-3; }
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
        Binding("m",      "start_llama_monitor", "Monitor"),
        Binding("n",      "download_model", "Télécharger"),
        Binding("?",      "help",           "Aide", show=True),
        Binding("slash",  "focus_filter",   "Filtre", show=True),
        Binding("tab",    "focus_models",   "Focus modèles", show=False),
        Binding("ctrl+q", "quit",           "Quitter"),
    ]

    def __init__(self):
        super().__init__()
        self.theme = "textual-dark"
        _ensure_config()
        self._manager       = ServerManager(CONFIG_FILE)
        self._proxy         = ProxyManager(PROXY_ROOT, port=8001)
        self._model_mgr     = ModelManager(models_dir=MODELS_DIR, config_file=CONFIG_FILE)
        self._status_model  = StatusModel()
        self._logs          = LogBuffer(max_lines=500)
        self._stats         = StatsCollector(port=self._manager._config.port)
        self._server_start_time: float | None = None
        self._poll_thread: threading.Thread | None = None
        self._running       = True
        self._filter        = ""

    # ── Layout ────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Static("", id="status-bar")
        with Horizontal(id="main"):
            with Vertical(id="model-panel"):
                yield Static("▪ Modèles", classes="section")
                yield Static("─" * 30, classes="divider")
                yield Input(placeholder="/ filtrer par nom…", id="model-filter")
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
                yield Static(" ", classes="divider")
                yield Static("▪ litellm proxy", classes="section")
                yield Static("─" * 26, classes="divider")
                yield Label("", id="lbl-proxy-status")
                yield Label("", id="lbl-proxy-pid")
                yield Label("", id="lbl-proxy-port")
                yield Label("", id="lbl-proxy-health")
                yield Static(" ", classes="divider")
                yield Static("▪ llama-monitor", classes="section")
                yield Static("─" * 26, classes="divider")
                yield Label("", id="lbl-monitor-status")
            with Vertical(id="right-panel"):
                with Vertical(id="stats-panel"):
                    yield Static("▪ Performance", classes="section")
                    yield Label("", id="lbl-tps-secondary")
                    yield Label("— t/s", id="lbl-tps")
                    yield Sparkline([], id="sparkline-tps", summary_function=max)
                yield Log(id="log-panel", highlight=True)
        with Horizontal(id="controls"):
            yield Button("▶ LLM [s]",    id="btn-start",     variant="success")
            yield Button("■ LLM [q]",    id="btn-stop",      variant="error")
            yield Button("↺ LLM [r]",    id="btn-restart",   variant="warning")
            yield Button("⏏ Load [l]",   id="btn-load",      variant="primary")
            yield Button("▶▶ Tout [a]",  id="btn-all-start", variant="success")
            yield Button("■■ Tout [z]",  id="btn-all-stop",  variant="error")
        with Horizontal(id="proxy-controls"):
            yield Button("▶ Proxy [p]",  id="btn-proxy-start", variant="success")
            yield Button("■ Proxy [o]",  id="btn-proxy-stop",  variant="error")
            yield Button("✕ Logs [c]",   id="btn-clear")
            yield Button("🌐 Llama WebUI [d]", id="btn-activity", variant="warning")
            yield Button("📈 Monitor [m]", id="btn-monitor")
        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._populate_model_list()
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
        flt = self._filter.lower()
        if flt:
            models = [m for m in models if flt in m.name.lower()]
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
            lv.append(ListItem(Label("  Aucun modèle" + (" correspondant" if flt else " trouvé"))))

    def _selected_model_path(self) -> Path | None:
        lv      = self.query_one("#model-list", ListView)
        idx     = lv.index
        models  = self._model_mgr.scan()
        flt = self._filter.lower()
        if flt:
            models = [m for m in models if flt in m.name.lower()]
        if idx is None or idx >= len(models):
            return None
        return models[idx].path

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "model-filter":
            self._filter = event.value.strip()
            self._populate_model_list()
            self.query_one("#model-list", ListView).focus()

    # ── UI refresh ────────────────────────────────────────────────────────

    def _refresh_ui(self) -> None:
        self._refresh_status_bar()
        self._refresh_server()
        self._refresh_proxy()
        self._refresh_monitor()
        self._refresh_stats()

    def _refresh_status_bar(self) -> None:
        info = self._manager.get_info()
        pinfo = self._proxy.get_info()
        srv_up = self._status_model.is_running
        proxy_up = pinfo["status"] == "running"
        model_name = Path(info["model"]).stem if info["model"] else "—"
        s = self._stats.last_stats
        tps = f"⚡ {s.gen_tps:.1f} t/s" if (s.is_generating and s.gen_tps) else "idle"
        badge = "[badge-ok]● RUNNING[/badge-ok]" if srv_up else "[badge-ko]● STOPPED[/badge-ko]"
        pbadge = "[badge-ok]●[/badge-ok]" if proxy_up else "[badge-ko]○[/badge-ko]"
        bar = (
            f"{badge}  [sep]│[/sep]  {model_name}  [sep]│[/sep]  {tps}  "
            f"[sep]│[/sep]  up {self._status_model.uptime_str}  [sep]│[/sep]  proxy {pbadge}"
        )
        self.query_one("#status-bar", Static).update(bar)

    def _refresh_server(self) -> None:
        info       = self._manager.get_info()
        is_running = self._status_model.is_running
        cls        = "running" if is_running else "stopped"
        lbl        = "● RUNNING" if is_running else "● STOPPED"
        model_name = Path(info["model"]).stem if info["model"] else "—"
        health     = self._manager.health_check()

        self.query_one("#lbl-status", Label).update(
            f"[bold]Status:[/bold] [{cls}]{lbl}[/{cls}]")
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
            f"[bold]Health:[/bold] [{'running' if health else 'stopped'}]"
            f"{'✓ OK' if health else '✗ N/A'}[/{'running' if health else 'stopped'}]")

        self.query_one("#btn-start",   Button).disabled = is_running
        self.query_one("#btn-stop",    Button).disabled = not is_running
        self.query_one("#btn-restart", Button).disabled = not is_running

    def _refresh_proxy(self) -> None:
        info       = self._proxy.get_info()
        is_running = info["status"] == "running"
        cls        = "running" if is_running else "stopped"
        lbl        = "● RUNNING" if is_running else "● STOPPED"
        health     = self._proxy.health_check()

        self.query_one("#lbl-proxy-status", Label).update(
            f"[bold]Status:[/bold] [{cls}]{lbl}[/{cls}]")
        self.query_one("#lbl-proxy-pid",    Label).update(
            f"[bold]PID:[/bold]    {info['pid'] or '—'}")
        self.query_one("#lbl-proxy-port",   Label).update(
            f"[bold]Port:[/bold]   :{info['port']} → Claude Code")
        self.query_one("#lbl-proxy-health", Label).update(
            f"[bold]Health:[/bold] [{'running' if health else 'stopped'}]"
            f"{'✓ OK' if health else '✗ N/A'}[/{'running' if health else 'stopped'}]")

        self.query_one("#btn-proxy-start", Button).disabled = is_running
        self.query_one("#btn-proxy-stop",  Button).disabled = not is_running

    def _refresh_monitor(self) -> None:
        up = _port_open(LLAMA_MONITOR_PORT)
        cls = "running" if up else "stopped"
        sym = "●" if up else "○"
        self.query_one("#lbl-monitor-status", Label).update(
            f"[bold]Monitor:[/bold] [{cls}]{sym} :{LLAMA_MONITOR_PORT}[/{cls}]")

    def _refresh_stats(self) -> None:
        s = self._stats.last_stats
        history = list(self._stats.history)
        if s.is_generating and s.gen_tps is not None:
            primary = f"⚡ {s.gen_tps:.1f} t/s"
        elif history:
            primary = f"[dim]dernier : {history[-1]:.1f} t/s[/dim]"
        else:
            primary = "— t/s"
        self.query_one("#lbl-tps", Label).update(primary)

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

    # ── Log helper ────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        self._logs.add(msg)
        self.query_one("#log-panel", Log).write_line(msg)

    # ── Actions ───────────────────────────────────────────────────────────

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_focus_filter(self) -> None:
        self.query_one("#model-filter", Input).focus()

    def action_load_model(self) -> None:
        path = self._selected_model_path()
        if path is None:
            self._log(f"[{self._ts()}] ⚠ Aucun modèle sélectionné")
            return
        if path == self._model_mgr.active_model:
            self._log(f"[{self._ts()}] ⚠ Modèle déjà actif")
            return

        was_running = self._manager.status() == ServerStatus.RUNNING
        self._log(f"[{self._ts()}] ⏏ Chargement → {path.stem}")

        if was_running:
            self._manager.stop()
            self._server_start_time = None
            self._log(f"[{self._ts()}] ■ Serveur arrêté pour changement de modèle")

        self._model_mgr.load(path)
        self._manager = ServerManager(CONFIG_FILE)
        self._stats = StatsCollector(port=self._manager._config.port)
        self._populate_model_list()
        self._log(f"[{self._ts()}] ✓ Modèle sélectionné : {path.stem}")

        if was_running:
            self.action_start_server()

        self._refresh_ui()

    def action_focus_models(self) -> None:
        self.query_one("#model-list", ListView).focus()

    def action_start_server(self) -> None:
        try:
            pid = self._manager.start()
            self._server_start_time = time.monotonic()
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

    def _build_llama_monitor_cmd(self, llama_server_bin: str) -> list[str]:
        from src.config import ServerConfig
        server_port = ServerConfig.from_file(CONFIG_FILE).port
        return [
            str(LLAMA_MONITOR_BIN),
            "--llama-server-path", llama_server_bin,
            "--llama-server-cwd", str(PROJECT_ROOT),
            "--port", str(LLAMA_MONITOR_PORT),
            "--models-dir", str(MODELS_DIR),
            "--presets-file", str(LLAMA_MONITOR_PRESETS),
            "--monitor-port", str(server_port),
        ]

    def action_start_llama_monitor(self) -> None:
        if not LLAMA_MONITOR_BIN.exists():
            self._log(f"[{self._ts()}] ⚠ llama-monitor non trouvé : {LLAMA_MONITOR_BIN}")
            self._log(f"[{self._ts()}]   → cd ~/llama-monitor && cargo build --release")
            return
        import shutil
        llama_server_bin = shutil.which("llama-server") or "llama-server"
        try:
            subprocess.Popen(
                self._build_llama_monitor_cmd(llama_server_bin),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                cwd=str(PROJECT_ROOT),
            )
            time.sleep(0.8)
            subprocess.Popen(["open", f"http://localhost:{LLAMA_MONITOR_PORT}"])
            self._log(f"[{self._ts()}] ▶ llama-monitor lancé → http://localhost:{LLAMA_MONITOR_PORT}")
        except Exception as e:
            self._log(f"[{self._ts()}] ✗ Erreur llama-monitor: {e}")

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
            "btn-monitor":     self.action_start_llama_monitor,
        }
        if event.button.id in actions:
            actions[event.button.id]()

    @staticmethod
    def _ts() -> str:
        return time.strftime("%H:%M:%S")