import json
import pytest
from pathlib import Path
from src.model_manager import ModelManager
from src.task_profiles import TaskProfile, TaskProfileError, TaskProfileManager


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    (d / "vitesse.gguf").write_bytes(b"x" * 1024)
    (d / "qualite.gguf").write_bytes(b"x" * 2048)
    return d


@pytest.fixture
def config_file(tmp_path, models_dir):
    cfg = tmp_path / "server.json"
    cfg.write_text(json.dumps({
        "model": str(models_dir / "vitesse.gguf"),
        "host": "0.0.0.0", "port": 8082, "n_gpu_layers": 99,
        "ctx_size": 131072, "batch_size": 2048, "ubatch_size": 2048,
        "threads": 12, "flash_attn": True, "parallel": 1,
        "cont_batching": True, "cache_type_k": "q4_0", "cache_type_v": "q4_0",
    }))
    return cfg


@pytest.fixture
def model_mgr(models_dir, config_file):
    return ModelManager(models_dir=models_dir, config_file=config_file)


@pytest.fixture
def profiles_file(tmp_path):
    pf = tmp_path / "task-profiles.json"
    pf.write_text(json.dumps([
        {"name": "vitesse", "model": "models/vitesse.gguf"},
        {"name": "qualite", "model": "models/qualite.gguf"},
    ]))
    return pf


@pytest.fixture
def profile_mgr(profiles_file, tmp_path):
    return TaskProfileManager(profiles_file=profiles_file, project_root=tmp_path)


def test_list_profiles_reads_json(profile_mgr, tmp_path):
    profiles = profile_mgr.list_profiles()
    assert {p.name for p in profiles} == {"vitesse", "qualite"}
    by_name = {p.name: p for p in profiles}
    assert by_name["vitesse"].model_path == tmp_path / "models" / "vitesse.gguf"


def test_list_profiles_raises_on_missing_file(tmp_path):
    mgr = TaskProfileManager(profiles_file=tmp_path / "ghost.json", project_root=tmp_path)
    with pytest.raises(TaskProfileError, match="not found"):
        mgr.list_profiles()


def test_list_profiles_raises_on_malformed_json_not_a_list(tmp_path):
    pf = tmp_path / "task-profiles.json"
    pf.write_text(json.dumps({"name": "vitesse", "model": "models/vitesse.gguf"}))
    mgr = TaskProfileManager(profiles_file=pf, project_root=tmp_path)
    with pytest.raises(TaskProfileError, match="list"):
        mgr.list_profiles()


def test_list_profiles_raises_on_missing_keys(tmp_path):
    pf = tmp_path / "task-profiles.json"
    pf.write_text(json.dumps([{"name": "vitesse"}]))
    mgr = TaskProfileManager(profiles_file=pf, project_root=tmp_path)
    with pytest.raises(TaskProfileError, match="model"):
        mgr.list_profiles()


def test_activate_calls_model_manager_load(profile_mgr, model_mgr, config_file, models_dir):
    profile_mgr.activate("qualite", model_mgr)
    saved = json.loads(config_file.read_text())
    assert Path(saved["model"]).name == "qualite.gguf"


def test_activate_raises_on_unknown_profile_name(profile_mgr, model_mgr):
    with pytest.raises(TaskProfileError, match="inconnu"):
        profile_mgr.activate("inconnu", model_mgr)


def test_activate_propagates_missing_model_file(tmp_path, model_mgr):
    pf = tmp_path / "task-profiles.json"
    pf.write_text(json.dumps([{"name": "ghost", "model": "models/ghost.gguf"}]))
    mgr = TaskProfileManager(profiles_file=pf, project_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        mgr.activate("ghost", model_mgr)


# ── Repo réel : config/task-profiles.json ──────────────────────────────────

def test_real_profiles_file_has_qualite_profile():
    """Le profil 'qualite' doit pointer vers un .gguf existant dans models/."""
    from src.tui import PROJECT_ROOT, TASK_PROFILES_FILE
    mgr = TaskProfileManager(profiles_file=TASK_PROFILES_FILE, project_root=PROJECT_ROOT)
    profiles = {p.name: p for p in mgr.list_profiles()}
    assert "qualite" in profiles, "profil qualite manquant dans config/task-profiles.json"
    assert profiles["qualite"].model_path.exists(), profiles["qualite"].model_path
    assert profiles["qualite"].model_path.name == "Qwen3.8-27B-UD-Q4_K_M.gguf"


def test_real_profiles_file_has_uncensored_profile():
    """Le profil 'uncensored' doit pointer vers un .gguf existant, avec sidecar vision sans MTP."""
    import json
    from src.tui import PROJECT_ROOT, TASK_PROFILES_FILE
    mgr = TaskProfileManager(profiles_file=TASK_PROFILES_FILE, project_root=PROJECT_ROOT)
    profiles = {p.name: p for p in mgr.list_profiles()}
    assert "uncensored" in profiles, "profil uncensored manquant dans config/task-profiles.json"
    model = profiles["uncensored"].model_path
    assert model.exists(), model
    assert model.name == "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
    sidecar = json.loads(model.with_suffix(".json").read_text())
    assert (PROJECT_ROOT / sidecar["mmproj"]).exists()
    assert "spec_type" not in sidecar
