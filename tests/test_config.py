import json
import pytest
from pathlib import Path
from src.config import ServerConfig, ConfigError


CONFIG_PATH = Path(__file__).parent.parent / "config" / "server.json"


def test_config_loads_from_json(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({
        "model": "/some/model.gguf",
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
    }))
    config = ServerConfig.from_file(cfg_file)
    assert config.model == "/some/model.gguf"
    assert config.port == 8080


def test_config_raises_on_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        ServerConfig.from_file(Path("/nonexistent/server.json"))


def test_config_raises_on_missing_required_field(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({"host": "0.0.0.0", "port": 8080}))
    with pytest.raises(ConfigError, match="model"):
        ServerConfig.from_file(cfg_file)


def test_config_host_allows_lan_access(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({
        "model": "/some/model.gguf",
        "host": "127.0.0.1",
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
    }))
    with pytest.raises(ConfigError, match="LAN"):
        ServerConfig.from_file(cfg_file)


def test_config_port_must_be_valid(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({
        "model": "/some/model.gguf",
        "host": "0.0.0.0",
        "port": 80,
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
    }))
    with pytest.raises(ConfigError, match="port"):
        ServerConfig.from_file(cfg_file)


def test_config_model_must_be_gguf(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({
        "model": "/some/model.bin",
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
    }))
    with pytest.raises(ConfigError, match="gguf"):
        ServerConfig.from_file(cfg_file)


def test_config_to_cli_args(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({
        "model": "/some/model.gguf",
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
    }))
    config = ServerConfig.from_file(cfg_file)
    args = config.to_cli_args()
    assert "--model" in args
    assert "/some/model.gguf" in args
    assert "--host" in args
    assert "0.0.0.0" in args
    assert "--port" in args
    assert "8080" in args
    assert "--n-gpu-layers" in args
    assert "99" in args
    assert "--flash-attn" in args
    assert "--ctx-size" in args
    assert "--parallel" in args


def test_cli_args_flash_attn_value_form(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({
        "model": "/some/model.gguf", "host": "0.0.0.0", "port": 8080,
        "n_gpu_layers": 99, "ctx_size": 32768, "batch_size": 2048,
        "ubatch_size": 512, "threads": 12, "flash_attn": True,
        "parallel": 4, "cont_batching": True,
        "cache_type_k": "q8_0", "cache_type_v": "q8_0",
    }))
    args = ServerConfig.from_file(cfg_file).to_cli_args()
    i = args.index("--flash-attn")
    assert args[i + 1] == "on"


def test_cli_args_flash_attn_off_when_false(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({
        "model": "/some/model.gguf", "host": "0.0.0.0", "port": 8080,
        "n_gpu_layers": 99, "ctx_size": 32768, "batch_size": 2048,
        "ubatch_size": 512, "threads": 12, "flash_attn": False,
        "parallel": 4, "cont_batching": True,
        "cache_type_k": "q8_0", "cache_type_v": "q8_0",
    }))
    args = ServerConfig.from_file(cfg_file).to_cli_args()
    i = args.index("--flash-attn")
    assert args[i + 1] == "off"


def test_config_full_gpu_offload(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({
        "model": "/some/model.gguf",
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
    }))
    config = ServerConfig.from_file(cfg_file)
    assert config.n_gpu_layers >= 99


# ── MTP (Multi-Token Prediction) ──────────────────────────────────────────────

BASE_CFG = {
    "model": "/some/model.gguf",
    "host": "0.0.0.0",
    "port": 8080,
    "n_gpu_layers": 99,
    "ctx_size": 32768,
    "batch_size": 2048,
    "ubatch_size": 512,
    "threads": 12,
    "flash_attn": True,
    "parallel": 1,
    "cont_batching": True,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
}


def test_mtp_fields_default_to_none_when_absent(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps(BASE_CFG))
    config = ServerConfig.from_file(cfg_file)
    assert config.spec_type is None
    assert config.spec_draft_n_max is None


def test_mtp_fields_load_from_config(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({**BASE_CFG, "spec_type": "draft-mtp", "spec_draft_n_max": 2}))
    config = ServerConfig.from_file(cfg_file)
    assert config.spec_type == "draft-mtp"
    assert config.spec_draft_n_max == 2


def test_mtp_cli_args_emitted_when_set(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({**BASE_CFG, "spec_type": "draft-mtp", "spec_draft_n_max": 2}))
    args = ServerConfig.from_file(cfg_file).to_cli_args()
    assert "--spec-type" in args
    assert "draft-mtp" in args
    assert "--spec-draft-n-max" in args
    assert "2" in args


def test_no_mtp_cli_args_when_spec_type_absent(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps(BASE_CFG))
    args = ServerConfig.from_file(cfg_file).to_cli_args()
    assert "--spec-type" not in args
    assert "--spec-draft-n-max" not in args


def test_mtp_requires_parallel_1(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({**BASE_CFG, "parallel": 4, "spec_type": "draft-mtp", "spec_draft_n_max": 2}))
    with pytest.raises(ConfigError, match="parallel"):
        ServerConfig.from_file(cfg_file)


# ── Jinja (chat_template_kwargs support) ─────────────────────────────────────

def test_jinja_default_true_emits_flag(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps(BASE_CFG))
    args = ServerConfig.from_file(cfg_file).to_cli_args()
    assert "--jinja" in args


def test_jinja_false_omits_flag(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({**BASE_CFG, "jinja": False}))
    args = ServerConfig.from_file(cfg_file).to_cli_args()
    assert "--jinja" not in args


def test_jinja_loaded_from_config(tmp_path):
    cfg_file = tmp_path / "server.json"
    cfg_file.write_text(json.dumps({**BASE_CFG, "jinja": False}))
    config = ServerConfig.from_file(cfg_file)
    assert config.jinja is False
