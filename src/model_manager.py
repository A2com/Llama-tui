import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.config import ConfigError

_QUANT_RE = re.compile(r"(Q[0-9]_[A-Z0-9_]+|IQ[0-9]?_[A-Z0-9_]+|F[0-9]+|BF16|FP16|F16)")

# Champs model-specific : reset à la baseline (défauts _ensure_config) ou purge
# au switch de modèle — jamais de fusion sur la config du modèle précédent.
_MODEL_SPECIFIC_RESET = {"ctx_size": 131072, "cache_type_k": "q4_0", "cache_type_v": "q4_0"}
_MODEL_SPECIFIC_POP = ("spec_type", "spec_draft_n_max", "mmproj", "thinking")

# Clés que le sidecar n'a pas le droit d'écraser (intégrité config serveur).
_PROTECTED_KEYS = ("model", "host", "port", "n_gpu_layers", "proxy_backend", "threads")


@dataclass
class ModelInfo:
    name: str
    path: Path
    size_gb: float
    mtime: float = 0.0
    quant: str = ""
    ctx_train: str = ""
    sidecar: dict = field(default_factory=dict)

    @property
    def mtime_date(self) -> str:
        if not self.mtime:
            return "—"
        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d")


def _read_sidecar(path: Path) -> dict:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_sidecar_ctx(path: Path) -> str:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return ""
    try:
        data = json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return ""
    ctx = data.get("ctx_size") or data.get("ctx_train") or data.get("context_size")
    return str(ctx) if ctx else ""


class ModelManager:
    def __init__(self, models_dir: Path, config_file: Path):
        self._models_dir = models_dir
        self._config_file = config_file
        self._project_root = config_file.parent.parent

    def _resolve(self, raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        return self._project_root / p

    def scan(self) -> list[ModelInfo]:
        files = sorted(f for f in self._models_dir.glob("*.gguf") if not f.name.startswith("mmproj"))
        infos = [
            ModelInfo(
                name=f.stem,
                path=f,
                size_gb=f.stat().st_size / 1_073_741_824,
                mtime=f.stat().st_mtime,
                quant=_extract_quant(f.stem),
                ctx_train=_read_sidecar_ctx(f),
                sidecar=_read_sidecar(f),
            )
            for f in files
        ]
        active = self.active_model
        infos.sort(key=lambda m: (m.path != active, m.name.lower()))
        return infos

    @property
    def active_model(self) -> Path:
        cfg = json.loads(self._config_file.read_text())
        return self._resolve(cfg["model"])

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        if path.suffix != ".gguf":
            raise ValueError(f"Not a gguf file: {path}")
        cfg = json.loads(self._config_file.read_text())
        try:
            rel = path.relative_to(self._project_root)
            cfg["model"] = str(rel)
        except ValueError:
            cfg["model"] = str(path)
        sidecar = path.with_suffix(".json")
        sidecar_sets_spec = False
        if sidecar.exists():
            try:
                sidecar_data = json.loads(sidecar.read_text())
            except json.JSONDecodeError as e:
                raise ConfigError(f"Sidecar illisible pour {path.name}: {e}") from e
            if not isinstance(sidecar_data, dict):
                raise ConfigError(f"Sidecar invalide pour {path.name}: JSON non-object")
            sidecar_sets_spec = "spec_type" in sidecar_data

        for key in _MODEL_SPECIFIC_POP:
            cfg.pop(key, None)
        cfg.update(_MODEL_SPECIFIC_RESET)
        if sidecar.exists():
            for key in _PROTECTED_KEYS:
                sidecar_data.pop(key, None)
            cfg.update(sidecar_data)

        if not sidecar_sets_spec:
            if "MTP" in path.name.upper():
                cfg["spec_type"] = "draft-mtp"
                cfg["spec_draft_n_max"] = 2
            else:
                cfg.pop("spec_type", None)
                cfg.pop("spec_draft_n_max", None)
        self._config_file.write_text(json.dumps(cfg, indent=2))

    def download(self, repo_id: str, filename: str, local_filename: str = None) -> ModelInfo:
        dest = self._models_dir / (local_filename or filename)
        if dest.exists():
            raise FileExistsError(f"Already exists: {dest}")
        if local_filename:
            import tempfile, shutil
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=tmp_path)
                shutil.move(str(tmp_path / filename), str(dest))
        else:
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=self._models_dir)
        return ModelInfo(
            name=dest.stem,
            path=dest,
            size_gb=dest.stat().st_size / 1_073_741_824,
            mtime=dest.stat().st_mtime,
            quant=_extract_quant(dest.stem),
            ctx_train=_read_sidecar_ctx(dest),
        )


def _extract_quant(stem: str) -> str:
    m = _QUANT_RE.search(stem)
    return m.group(1) if m else ""
