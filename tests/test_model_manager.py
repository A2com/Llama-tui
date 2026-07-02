import json
import pytest
from pathlib import Path
from src.model_manager import ModelInfo, ModelManager


@pytest.fixture
def models_dir(tmp_path):
    (tmp_path / "alpha.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "beta.gguf").write_bytes(b"x" * 2048)
    (tmp_path / "ignored.txt").write_text("not a model")
    return tmp_path


@pytest.fixture
def config_file(tmp_path, models_dir):
    cfg = tmp_path / "server.json"
    cfg.write_text(json.dumps({
        "model": str(models_dir / "alpha.gguf"),
        "host": "0.0.0.0", "port": 11434, "n_gpu_layers": 99,
        "ctx_size": 131072, "batch_size": 2048, "ubatch_size": 2048,
        "threads": 12, "flash_attn": True, "parallel": 2,
        "cont_batching": True, "cache_type_k": "q4_0", "cache_type_v": "q4_0",
    }))
    return cfg


@pytest.fixture
def manager(models_dir, config_file):
    return ModelManager(models_dir=models_dir, config_file=config_file)


def test_scan_returns_only_gguf_files(manager):
    models = manager.scan()
    assert all(m.path.suffix == ".gguf" for m in models)


def test_scan_finds_all_gguf_files(manager):
    names = {m.name for m in manager.scan()}
    assert names == {"alpha", "beta"}


def test_scan_returns_model_info_with_size(manager):
    models = {m.name: m for m in manager.scan()}
    assert models["alpha"].size_gb > 0
    assert models["beta"].size_gb > models["alpha"].size_gb


def test_active_model_matches_config(manager, models_dir):
    assert manager.active_model == models_dir / "alpha.gguf"


def test_load_updates_active_model(manager, models_dir):
    manager.load(models_dir / "beta.gguf")
    assert manager.active_model == models_dir / "beta.gguf"


def test_load_persists_to_config(manager, models_dir, config_file):
    manager.load(models_dir / "beta.gguf")
    saved = json.loads(config_file.read_text())
    assert Path(saved["model"]).name == "beta.gguf"
    assert (config_file.parent.parent / saved["model"]).resolve() == (models_dir / "beta.gguf").resolve()


def test_load_raises_if_file_not_gguf(manager, tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="gguf"):
        manager.load(bad)


def test_load_raises_if_file_missing(manager, tmp_path):
    with pytest.raises(FileNotFoundError):
        manager.load(tmp_path / "ghost.gguf")


def test_scan_sorted_alphabetically(manager):
    names = [m.name for m in manager.scan()]
    assert names == sorted(names)


def test_model_info_has_path(manager, models_dir):
    models = {m.name: m for m in manager.scan()}
    assert models["alpha"].path == models_dir / "alpha.gguf"


# ── Champs étendus (quant / date / ctx / tri active-first) ───────────────────

def test_quant_extracted_from_name(tmp_path, config_file):
    (tmp_path / "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf").write_bytes(b"x" * 1024)
    mgr = ModelManager(models_dir=tmp_path, config_file=config_file)
    models = {m.name: m for m in mgr.scan()}
    assert models["Qwen3.6-35B-A3B-UD-Q6_K_XL"].quant == "Q6_K_XL"


def test_quant_empty_when_no_match(tmp_path, config_file):
    (tmp_path / "plain-model.gguf").write_bytes(b"x" * 1024)
    mgr = ModelManager(models_dir=tmp_path, config_file=config_file)
    models = {m.name: m for m in mgr.scan()}
    assert models["plain-model"].quant == ""


def test_mtime_date_formatted(manager):
    models = {m.name: m for m in manager.scan()}
    assert len(models["alpha"].mtime_date) == 10  # YYYY-MM-DD


def test_ctx_train_read_from_sidecar(tmp_path, config_file):
    (tmp_path / "model.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "model.json").write_text(json.dumps({"ctx_train": "262144"}))
    mgr = ModelManager(models_dir=tmp_path, config_file=config_file)
    models = {m.name: m for m in mgr.scan()}
    assert models["model"].ctx_train == "262144"


def test_scan_active_model_first(config_file, tmp_path):
    # active = zeta.gguf, autres alpha/beta — zeta doit sortir en premier
    cfg = json.loads(config_file.read_text())
    cfg["model"] = str(tmp_path / "zeta.gguf")
    config_file.write_text(json.dumps(cfg))
    (tmp_path / "zeta.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "alpha.gguf").write_bytes(b"x" * 1024)
    (tmp_path / "beta.gguf").write_bytes(b"x" * 1024)
    mgr = ModelManager(models_dir=tmp_path, config_file=config_file)
    names = [m.name for m in mgr.scan()]
    assert names[0] == "zeta"
    assert names[1:] == sorted(["alpha", "beta"])


# ── ModelManager.download() ───────────────────────────────────────────────────

def test_download_calls_hf_hub_with_correct_args(manager, models_dir, monkeypatch):
    calls = []

    def fake_download(repo_id, filename, local_dir):
        calls.append({"repo_id": repo_id, "filename": filename, "local_dir": local_dir})
        (local_dir / filename).write_bytes(b"fake")
        return str(local_dir / filename)

    monkeypatch.setattr("src.model_manager.hf_hub_download", fake_download)
    manager.download("unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "Qwen3.6-35B-A3B-MTP-UD-Q6_K_XL.gguf")

    assert len(calls) == 1
    assert calls[0]["repo_id"] == "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
    assert calls[0]["filename"] == "Qwen3.6-35B-A3B-MTP-UD-Q6_K_XL.gguf"
    assert calls[0]["local_dir"] == models_dir


def test_download_returns_model_info(manager, models_dir, monkeypatch):
    def fake_download(repo_id, filename, local_dir):
        p = local_dir / filename
        p.write_bytes(b"x" * 1024)
        return str(p)

    monkeypatch.setattr("src.model_manager.hf_hub_download", fake_download)
    info = manager.download("unsloth/repo", "new.gguf")

    assert info.name == "new"
    assert info.path == models_dir / "new.gguf"
    assert info.size_gb > 0


def test_download_raises_if_file_already_exists(manager, models_dir):
    with pytest.raises(FileExistsError):
        manager.download("unsloth/repo", "alpha.gguf")


def test_download_saves_under_local_filename_when_provided(manager, models_dir, monkeypatch):
    def fake_download(repo_id, filename, local_dir):
        p = local_dir / filename
        p.write_bytes(b"x" * 512)
        return str(p)

    monkeypatch.setattr("src.model_manager.hf_hub_download", fake_download)
    info = manager.download("unsloth/repo", "remote.gguf", local_filename="local-alias.gguf")

    assert info.name == "local-alias"
    assert info.path == models_dir / "local-alias.gguf"
    assert info.path.exists()
    assert not (models_dir / "remote.gguf").exists()


def test_download_raises_if_local_filename_already_exists(manager, models_dir):
    with pytest.raises(FileExistsError):
        manager.download("unsloth/repo", "remote.gguf", local_filename="alpha.gguf")


# ── Sidecar config ────────────────────────────────────────────────────────────

def test_load_applies_sidecar_overrides_when_present(manager, models_dir, config_file):
    (models_dir / "beta.json").write_text(json.dumps({"spec_type": "draft-mtp", "spec_draft_n_max": 2}))
    manager.load(models_dir / "beta.gguf")
    saved = json.loads(config_file.read_text())
    assert saved["spec_type"] == "draft-mtp"
    assert saved["spec_draft_n_max"] == 2


def test_load_clears_mtp_fields_when_no_sidecar(manager, models_dir, config_file):
    # Pre-set MTP fields in config
    cfg = json.loads(config_file.read_text())
    cfg["spec_type"] = "draft-mtp"
    cfg["spec_draft_n_max"] = 2
    config_file.write_text(json.dumps(cfg))

    manager.load(models_dir / "alpha.gguf")
    saved = json.loads(config_file.read_text())
    assert "spec_type" not in saved
    assert "spec_draft_n_max" not in saved


def test_load_replaces_sidecar_when_switching_models(manager, models_dir, config_file):
    (models_dir / "beta.json").write_text(json.dumps({"spec_type": "draft-mtp", "spec_draft_n_max": 2}))
    manager.load(models_dir / "beta.gguf")

    # Switch to alpha (no sidecar) — MTP fields must disappear
    manager.load(models_dir / "alpha.gguf")
    saved = json.loads(config_file.read_text())
    assert "spec_type" not in saved
    assert "spec_draft_n_max" not in saved
