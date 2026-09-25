# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Local LLM inference stack with a TUI manager. Two-layer architecture:

1. **llama-server** (port 8082) — llama.cpp OpenAI-compatible inference server, runs GGUF models via Metal GPU
2. **litellm proxy** (port 8001) — translates Anthropic API → llama-server OpenAI API, so Claude Code can talk to local models using Claude model names

The TUI (`llama-tui`) manages both processes: start/stop/restart, model switching, real-time t/s stats.

## Commands

```bash
# Run the TUI
python llama-tui

# Run tests
pytest

# Run a single test file
pytest tests/test_server_manager.py

# Run a single test
pytest tests/test_server_manager.py::test_start_returns_pid
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLAMA_TUI_MODELS_DIR` | `PROJECT_ROOT/models` | Directory containing `.gguf` files |
| `LLAMA_TUI_CONFIG` | `PROJECT_ROOT/config/server.json` | llama-server config file |
| `LLAMA_TUI_SERVER_BIN` | `llama-server` | llama-server binary path (override for side-by-side version testing) |
| `LLAMA_TUI_LITELLM_CONFIG` | `PROJECT_ROOT/config/litellm.yaml` | litellm proxy config |
| `LLAMA_TUI_TASK_PROFILES` | `PROJECT_ROOT/config/task-profiles.json` | named model-switch profiles (e.g. vitesse/qualite) |
| `LLAMA_TUI_CLINE_DIR` | `~/.cline/data/settings` | Cline settings dir synced by the TUI (`models.json`, `providers.json`) |

## Architecture

```
src/
  config.py          # ServerConfig: loads config/server.json, validates, generates llama-server CLI args
  server_manager.py  # ServerManager: subprocess lifecycle for llama-server (start/stop/status/health)
  proxy_manager.py   # ProxyManager: subprocess lifecycle for litellm / Bun backend (start/stop/status/health)
  model_manager.py   # ModelManager: scans models/, rewrites config/server.json to switch models, hf download
  stats_collector.py # StatsCollector: polls llama-server /slots for live tokens/sec and cache history
  cline_config.py    # sync_cline_models(): one Cline model entry per task profile (vision/reasoning/ctx from sidecar), active model written to providers.json
  tui_model.py       # StatusModel + LogBuffer: pure state containers (no I/O)
  tui.py             # LlamaTUI(App): Textual app, composes everything, 0.5s poll loop via thread
  version.py         # current_version_string()/check_version(): parses `llama-server --version` (build# or semver)
  fast_proxy.ts      # Bun proxy backend: Anthropic → OpenAI translation, /health passthrough
```

### Config files

- `config/server.json` — llama-server parameters (model path, ctx size, GPU layers, etc.). `host` must be `0.0.0.0`. `ModelManager.load()` rewrites the `model` field here to switch models. Relative paths are resolved from the project root. A model's `.json` sidecar (same stem) can set `spec_type`/`spec_draft_n_max` explicitly — sidecar wins over the filename-based MTP heuristic (`"MTP"` in name).
- `config/litellm.yaml` — litellm model list: maps Claude model names to `openai/qwen3` pointing at `http://127.0.0.1:8082/v1`.
- `config/task-profiles.json` — named task profiles (`[{"name": ..., "model": "models/xxx.gguf"}, ...]`), switchable from the TUI (`t` key) without touching `server.json` by hand. `TaskProfileManager` (`src/task_profiles.py`) resolves a profile to a model path and delegates to `ModelManager.load()`.

### Key design constraints

- `ServerConfig.host` must be `"0.0.0.0"` (LAN access requirement enforced in validation).
- `StatsCollector` polls `/slots` endpoint; resets delta tracking when `n_decoded` goes backward (new generation).
- `ServerManager.status()` checks in priority order: in-memory process → pidfile → port scan. Allows detecting manually launched servers.
- Tests mock `_port_in_use` and `_pid_on_port` via `monkeypatch` — never hit real ports.

### TUI keybindings

| Key | Action |
|-----|--------|
| `s` | Start llama-server |
| `q` | Stop llama-server |
| `r` | Restart llama-server |
| `l` | Load selected model |
| `p` | Start proxy |
| `o` | Stop proxy |
| `b` | Switch proxy backend (bun ↔ litellm) |
| `t` | Switch task profile (vitesse/qualite/uncensored) |
| `a` | Start all |
| `z` | Stop all |
| `n` | Download model from HuggingFace |
| `d` | Open Llama WebUI |
| `c` | Clear logs |
| `/` | Filter models |
| `Tab` | Focus model list |
| `?` | Show help |
| `Ctrl+Q` | Quit |
