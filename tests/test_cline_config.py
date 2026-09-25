import json
from pathlib import Path

import pytest

from src.cline_config import sync_cline_models

BASE_URL = "http://localhost:8082/v1"


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "proj"
    models = root / "models"
    models.mkdir(parents=True)
    (models / "vitesse.gguf").write_bytes(b"x")
    (models / "vitesse.json").write_text(json.dumps({"mmproj": "models/mm.gguf"}))
    (models / "qualite.gguf").write_bytes(b"x")
    (models / "qualite.json").write_text(json.dumps({
        "mmproj": "models/mm.gguf", "thinking": True, "ctx_size": 262144,
    }))
    (models / "texte.gguf").write_bytes(b"x")
    profiles = root / "profiles.json"
    profiles.write_text(json.dumps([
        {"name": "vitesse", "model": "models/vitesse.gguf"},
        {"name": "qualite", "model": "models/qualite.gguf"},
        {"name": "texte", "model": "models/texte.gguf"},
    ]))
    cline = tmp_path / "cline_settings"
    cline.mkdir()
    return root, profiles, cline


def _sync(env, active="vitesse"):
    root, profiles, cline = env
    sync_cline_models(profiles, root, cline, active_model=root / "models" / f"{active}.gguf", base_url=BASE_URL)
    return json.loads((cline / "models.json").read_text())


def test_one_entry_per_profile(env):
    data = _sync(env)
    assert set(data["providers"]["llama-cpp"]["models"]) == {"vitesse", "qualite", "texte"}


def test_vision_follows_sidecar_mmproj(env):
    models = _sync(env)["providers"]["llama-cpp"]["models"]
    assert models["vitesse"]["supportsVision"] is True
    assert models["vitesse"]["supportsAttachments"] is True
    assert models["texte"]["supportsVision"] is False


def test_reasoning_follows_sidecar_thinking(env):
    models = _sync(env)["providers"]["llama-cpp"]["models"]
    assert models["qualite"]["supportsReasoning"] is True
    assert models["vitesse"]["supportsReasoning"] is False


def test_context_window_from_sidecar_or_default(env):
    models = _sync(env)["providers"]["llama-cpp"]["models"]
    assert models["qualite"]["contextWindow"] == 262144
    assert models["vitesse"]["contextWindow"] == 131072


def test_provider_default_is_active_model(env):
    provider = _sync(env, active="qualite")["providers"]["llama-cpp"]["provider"]
    assert provider["defaultModelId"] == "qualite"
    assert provider["baseUrl"] == BASE_URL


def test_idempotent(env):
    _, _, cline = env
    _sync(env)
    first = (cline / "models.json").read_text()
    _sync(env)
    assert (cline / "models.json").read_text() == first


def test_preserves_other_providers_and_unknown_keys(env):
    _, _, cline = env
    (cline / "models.json").write_text(json.dumps({
        "version": 1,
        "extra": {"keep": True},
        "providers": {"autre": {"provider": {"name": "x"}, "models": {}}},
    }))
    data = _sync(env)
    assert data["extra"] == {"keep": True}
    assert data["providers"]["autre"] == {"provider": {"name": "x"}, "models": {}}


def test_replaces_stale_model_entries(env):
    _, _, cline = env
    (cline / "models.json").write_text(json.dumps({
        "version": 1,
        "providers": {"llama-cpp": {"provider": {}, "models": {"/old/path.gguf": {"id": "/old/path.gguf"}}}},
    }))
    assert "/old/path.gguf" not in _sync(env)["providers"]["llama-cpp"]["models"]


def test_providers_json_llama_cpp_model_follows_active(env):
    _, _, cline = env
    (cline / "providers.json").write_text(json.dumps({
        "version": 1,
        "providers": {"llama-cpp": {"settings": {"provider": "llama-cpp", "model": "anthropic/claude-sonnet-4.6"}}},
    }))
    _sync(env, active="qualite")
    saved = json.loads((cline / "providers.json").read_text())
    assert saved["providers"]["llama-cpp"]["settings"]["model"] == "qualite"


def test_providers_json_openai_compatible_synced_only_when_same_base_url(env):
    _, _, cline = env
    (cline / "providers.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-compatible": {"settings": {"baseUrl": BASE_URL, "model": "old", "apiKey": "k"}},
            "other": {"settings": {"baseUrl": "http://elsewhere/v1", "model": "keep"}},
        },
    }))
    _sync(env, active="qualite")
    saved = json.loads((cline / "providers.json").read_text())["providers"]
    assert saved["openai-compatible"]["settings"]["model"] == "qualite"
    assert saved["openai-compatible"]["settings"]["apiKey"] == "k"
    assert saved["other"]["settings"]["model"] == "keep"


def test_missing_providers_json_is_not_created(env):
    _, _, cline = env
    _sync(env)
    assert not (cline / "providers.json").exists()


def test_missing_cline_dir_is_noop(env, tmp_path):
    root, profiles, _ = env
    ghost = tmp_path / "absent"
    sync_cline_models(profiles, root, ghost, active_model=root / "models" / "vitesse.gguf", base_url=BASE_URL)
    assert not ghost.exists()


def test_active_model_outside_profiles_leaves_default_untouched(env):
    _, _, cline = env
    (cline / "models.json").write_text(json.dumps({
        "version": 1,
        "providers": {"llama-cpp": {"provider": {"defaultModelId": "keep"}, "models": {}}},
    }))
    root, profiles, _ = env
    sync_cline_models(profiles, root, cline, active_model=root / "models" / "inconnu.gguf", base_url=BASE_URL)
    data = json.loads((cline / "models.json").read_text())
    assert data["providers"]["llama-cpp"]["provider"]["defaultModelId"] == "keep"
