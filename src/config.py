import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


class ConfigError(Exception):
    pass


@dataclass
class ServerConfig:
    model: str
    host: str
    port: int
    n_gpu_layers: int
    ctx_size: int
    batch_size: int
    ubatch_size: int
    threads: int
    flash_attn: bool
    parallel: int
    cont_batching: bool
    cache_type_k: str
    cache_type_v: str
    jinja: bool = True
    spec_type: Optional[str] = None
    spec_draft_n_max: Optional[int] = None
    mmproj: Optional[str] = None
    server_bin: Optional[str] = None  # binaire llama-server (persisté, indépendant de l'env)
    _project_root: Path = field(default=Path("."), repr=False)

    @classmethod
    def from_file(cls, path: Path) -> "ServerConfig":
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON: {e}")

        required = ["model", "host", "port", "n_gpu_layers", "ctx_size",
                    "batch_size", "ubatch_size", "threads", "flash_attn",
                    "parallel", "cont_batching", "cache_type_k", "cache_type_v"]
        for key in required:
            if key not in data:
                raise ConfigError(f"Missing required field: {key}")

        if not data["model"].endswith(".gguf"):
            raise ConfigError("model must point to a .gguf file")

        if data["host"] != "0.0.0.0":
            raise ConfigError("host must be 0.0.0.0 for LAN access")

        if not (1024 <= data["port"] <= 65535):
            raise ConfigError(f"port {data['port']} is not in valid range 1024-65535")

        spec_type = data.get("spec_type")
        spec_draft_n_max = data.get("spec_draft_n_max")
        jinja = data.get("jinja", True)

        if spec_type is not None and data["parallel"] != 1:
            raise ConfigError("MTP (spec_type) requires parallel=1")

        project_root = path.parent.parent
        return cls(
            **{k: data[k] for k in required},
            jinja=jinja,
            spec_type=spec_type,
            spec_draft_n_max=spec_draft_n_max,
            mmproj=data.get("mmproj"),
            server_bin=data.get("server_bin"),
            _project_root=project_root,
        )

    def _resolve(self, raw_path: str) -> Path:
        raw = Path(raw_path)
        if raw.is_absolute():
            return raw
        return self._project_root / raw

    @property
    def model_path(self) -> Path:
        return self._resolve(self.model)

    @property
    def mmproj_path(self) -> Optional[Path]:
        return self._resolve(self.mmproj) if self.mmproj is not None else None

    def to_cli_args(self) -> List[str]:
        args = [
            "--model", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--ctx-size", str(self.ctx_size),
            "--batch-size", str(self.batch_size),
            "--ubatch-size", str(self.ubatch_size),
            "--threads", str(self.threads),
            "--parallel", str(self.parallel),
            "--cache-type-k", self.cache_type_k,
            "--cache-type-v", self.cache_type_v,
        ]
        args.extend(["--flash-attn", "on" if self.flash_attn else "off"])
        if self.jinja:
            args.append("--jinja")
        if self.cont_batching:
            args.append("--cont-batching")
        if self.spec_type is not None:
            args.extend(["--spec-type", self.spec_type])
        if self.spec_draft_n_max is not None:
            args.extend(["--spec-draft-n-max", str(self.spec_draft_n_max)])
        if self.mmproj_path is not None:
            args.extend(["--mmproj", str(self.mmproj_path)])
        return args
