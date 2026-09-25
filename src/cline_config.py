import json
from pathlib import Path

from src.task_profiles import TaskProfileManager

PROVIDER_ID = "llama-cpp"
DEFAULT_CTX = 131072


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _model_entry(name: str, model_path: Path) -> dict:
    sidecar = model_path.with_suffix(".json")
    data = _read_json(sidecar) if sidecar.exists() else {}
    vision = bool(data.get("mmproj"))
    return {
        "id": name,
        "name": name,
        "contextWindow": data.get("ctx_size", DEFAULT_CTX),
        "supportsVision": vision,
        "supportsAttachments": vision,
        "supportsReasoning": bool(data.get("thinking")),
    }


def _sync_models_json(path: Path, entries: dict, base_url: str, active_id: str | None) -> None:
    data = _read_json(path) if path.exists() else {"version": 1}
    provider = data.setdefault("providers", {}).setdefault(PROVIDER_ID, {})
    meta = provider.setdefault("provider", {})
    meta.setdefault("name", "llama.cpp (local)")
    meta["baseUrl"] = base_url
    meta.setdefault("capabilities", ["streaming", "tools"])
    if active_id is not None:
        meta["defaultModelId"] = active_id
    provider["models"] = entries
    path.write_text(json.dumps(data, indent=2))


def _sync_providers_json(path: Path, base_url: str, active_id: str) -> None:
    if not path.exists():
        return
    data = _read_json(path)
    for pid, entry in data.get("providers", {}).items():
        settings = entry.get("settings", {})
        if pid == PROVIDER_ID or settings.get("baseUrl") == base_url:
            settings["model"] = active_id
    path.write_text(json.dumps(data, indent=2))


def sync_cline_models(
    profiles_file: Path,
    project_root: Path,
    cline_settings_dir: Path,
    active_model: Path,
    base_url: str,
) -> None:
    if not cline_settings_dir.is_dir():
        return
    profiles = TaskProfileManager(profiles_file, project_root).list_profiles()
    entries = {p.name: _model_entry(p.name, p.model_path) for p in profiles}
    active_id = next((p.name for p in profiles if p.model_path == active_model), None)
    _sync_models_json(cline_settings_dir / "models.json", entries, base_url, active_id)
    if active_id is not None:
        _sync_providers_json(cline_settings_dir / "providers.json", base_url, active_id)
