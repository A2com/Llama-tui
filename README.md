# Llama-tui

Local LLM inference stack with a terminal UI manager.

## Stack

- **llama-server** — llama.cpp OpenAI-compatible server (port 8082)
- **litellm proxy** — Anthropic API → OpenAI bridge for Claude Code integration (port 8001)
- **TUI** — model switching, process control, live tokens/sec stats

Requires `llama-server` b10310+ (tested up to b10330).

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Put GGUF models in models/
# Start the TUI
python llama-tui
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLAMA_TUI_MODELS_DIR` | `PROJECT_ROOT/models` | Directory containing `.gguf` files |
| `LLAMA_TUI_CONFIG` | `PROJECT_ROOT/config/server.json` | llama-server config |
| `LLAMA_TUI_SERVER_BIN` | `llama-server` | llama-server binary path (override for side-by-side testing) |
| `LLAMA_TUI_LITELLM_CONFIG` | `PROJECT_ROOT/config/litellm.yaml` | litellm proxy config |
| `LLAMA_TUI_MONITOR_BIN` | `~/llama-monitor/target/release/llama-monitor` | Optional monitor binary |
| `LLAMA_TUI_MONITOR_PORT` | `7778` | Monitor web port |
| `LLAMA_TUI_MONITOR_PRESETS` | `PROJECT_ROOT/config/llama-monitor-presets.json` | Monitor presets |

## Tests

```bash
pytest
```

## License

MIT
