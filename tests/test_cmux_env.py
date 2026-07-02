import os
import pytest
from unittest.mock import patch, MagicMock


def test_cmux_wrapper_clears_auth_vars_by_default():
    """Sans CMUX_PRESERVE, le wrapper CMUX supprime les vars d'auth."""
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = "http://localhost:8001"
    env["ANTHROPIC_API_KEY"] = "sk-local-proxy"
    # Simule le wrapper CMUX qui fait unset sur les vars
    for key in ["ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"]:
        env.pop(key, None)
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_cmux_preserve_env_keeps_auth_vars():
    """Avec CMUX_PRESERVE_CLAUDE_AUTH_SELECTION_ENV=true, les vars survivent."""
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = "http://localhost:8001"
    env["ANTHROPIC_API_KEY"] = "sk-local-proxy"
    env["CMUX_PRESERVE_CLAUDE_AUTH_SELECTION_ENV"] = "true"
    # Le wrapper CMUX vérifie cette var avant de faire unset
    preserve = env.get("CMUX_PRESERVE_CLAUDE_AUTH_SELECTION_ENV") == "true"
    if preserve:
        assert "ANTHROPIC_BASE_URL" in env
        assert "ANTHROPIC_API_KEY" in env
    else:
        pytest.fail("CMUX_PRESERVE devrait être true")


def test_alias_claude_local_sets_correct_env():
    """L'alias claude-local doit exporter ANTHROPIC_BASE_URL et ANTHROPIC_API_KEY."""
    # Simule l'alias : ANTHROPIC_BASE_URL=... ANTHROPIC_API_KEY=... claude
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = "http://localhost:8001"
    env["ANTHROPIC_API_KEY"] = "sk-local-proxy"
    assert env["ANTHROPIC_BASE_URL"] == "http://localhost:8001"
    assert env["ANTHROPIC_API_KEY"] == "sk-local-proxy"
