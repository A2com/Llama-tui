"""
Tests for litellm.yaml correctness.

Key invariant: all api_base values must use 127.0.0.1 (not localhost).
On macOS, `localhost` resolves to ::1 (IPv6) which hits any Docker container
bound on [::]:8081 (e.g. phpmyadmin) instead of llama-server on IPv4 0.0.0.0:8081.
"""
import yaml
import pytest
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent / "config" / "litellm.yaml"


@pytest.fixture
def litellm_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_config_file_exists():
    assert CONFIG_PATH.exists(), f"litellm.yaml not found at {CONFIG_PATH}"


def test_model_list_not_empty(litellm_config):
    assert litellm_config.get("model_list"), "model_list must not be empty"


def test_wildcard_catch_all_present(litellm_config):
    """A '*' catch-all entry routes any future Anthropic model id (claude-sonnet-5,
    claude-sonnet-6, …) to qwen, so Claude Code updates never break the proxy."""
    names = [m["model_name"] for m in litellm_config["model_list"]]
    assert "*" in names, "wildcard model_name '*' must be present for update-proof routing"


def test_enable_thinking_disabled(litellm_config):
    """qwen3 is a hybrid-thinking model; without enable_thinking=false it emits
    empty content (all tokens go to reasoning_content). Must be disabled via
    chat_template_kwargs in extra_body."""
    for entry in litellm_config["model_list"]:
        extra = entry.get("litellm_params", {}).get("extra_body", {})
        kwargs = extra.get("chat_template_kwargs", {})
        assert kwargs.get("enable_thinking") is False, (
            f"Model '{entry['model_name']}' must set extra_body.chat_template_kwargs.enable_thinking=false"
        )


def test_no_api_base_uses_localhost(litellm_config):
    """
    All api_base must use 127.0.0.1, not localhost.
    localhost resolves to IPv6 ::1 on macOS, hitting Docker containers
    that shadow port 8081 (e.g. phpmyadmin) instead of llama-server on IPv4.
    """
    for entry in litellm_config["model_list"]:
        api_base = entry.get("litellm_params", {}).get("api_base", "")
        assert "localhost" not in api_base, (
            f"Model '{entry['model_name']}' uses localhost in api_base='{api_base}'. "
            "Use 127.0.0.1 to force IPv4 and avoid Docker IPv6 shadowing."
        )


def test_all_api_bases_use_ipv4(litellm_config):
    """Positive assertion: api_base must explicitly use 127.0.0.1."""
    for entry in litellm_config["model_list"]:
        api_base = entry.get("litellm_params", {}).get("api_base", "")
        assert "127.0.0.1" in api_base, (
            f"Model '{entry['model_name']}' api_base='{api_base}' must use 127.0.0.1"
        )


def test_drop_params_enabled(litellm_config):
    """drop_params must be True to absorb Claude-specific params llama.cpp ignores."""
    settings = litellm_config.get("litellm_settings", {})
    assert settings.get("drop_params") is True, "litellm_settings.drop_params must be true"


def test_request_timeout_set(litellm_config):
    settings = litellm_config.get("litellm_settings", {})
    timeout = settings.get("request_timeout", 0)
    assert timeout >= 120, f"request_timeout={timeout} too low for local inference (min 120s)"
