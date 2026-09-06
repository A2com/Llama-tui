from unittest.mock import patch

import pytest

from src import version


def test_check_version_true_when_meets(monkeypatch):
    monkeypatch.setenv("LLAMA_TUI_SERVER_BIN", "/x/llama-server")
    with patch("subprocess.check_output", return_value="version: 10310 (abc)\nbuilt with AppleClang"):
        assert version.check_version(10310) is True
        assert version.check_version(9850) is True


def test_check_version_false_when_below(monkeypatch):
    with patch("subprocess.check_output", return_value="version: 9850 (abc)\nbuilt with AppleClang"):
        assert version.check_version(10310) is False


def test_check_version_false_on_error(monkeypatch):
    with patch("subprocess.check_output", side_effect=FileNotFoundError("no binary")):
        assert version.check_version(10310) is False


def test_check_version_false_on_unparseable(monkeypatch):
    with patch("subprocess.check_output", return_value="garbage without version number"):
        assert version.check_version(10310) is False


def test_uses_env_binary(monkeypatch):
    monkeypatch.setenv("LLAMA_TUI_SERVER_BIN", "/custom/llama-server")
    with patch("subprocess.check_output", return_value="version: 10310") as mock_co:
        version.check_version(10310)
    cmd = mock_co.call_args[0][0]
    assert cmd[0] == "/custom/llama-server"
    assert cmd[1] == "--version"


def test_current_version_string_build_number(monkeypatch):
    with patch("subprocess.check_output", return_value="version: 10360 (48d22e295)\nbuilt with AppleClang"):
        assert version.current_version_string() == "10360"


def test_current_version_string_semver(monkeypatch):
    with patch("subprocess.check_output", return_value="version: 0.4.0 (abc1234)\nbuilt with AppleClang"):
        assert version.current_version_string() == "0.4.0"


def test_current_version_string_none_on_missing_binary(monkeypatch):
    with patch("subprocess.check_output", side_effect=FileNotFoundError("no binary")):
        assert version.current_version_string() is None


def test_current_version_string_none_on_unparseable(monkeypatch):
    with patch("subprocess.check_output", return_value="garbage without version number"):
        assert version.current_version_string() is None


def test_check_version_true_on_semver_regardless_of_min(monkeypatch):
    with patch("subprocess.check_output", return_value="version: 0.4.0 (abc1234)\nbuilt with AppleClang"):
        assert version.check_version(10310) is True
        assert version.check_version(99999) is True