import json
from dataclasses import dataclass
from pathlib import Path

from src.model_manager import ModelManager


class TaskProfileError(Exception):
    pass


@dataclass
class TaskProfile:
    name: str
    model_path: Path


class TaskProfileManager:
    def __init__(self, profiles_file: Path, project_root: Path):
        self._profiles_file = profiles_file
        self._project_root = project_root

    def _resolve(self, raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        return self._project_root / p

    def list_profiles(self) -> list[TaskProfile]:
        if not self._profiles_file.exists():
            raise TaskProfileError(f"Task profiles file not found: {self._profiles_file}")

        try:
            data = json.loads(self._profiles_file.read_text())
        except json.JSONDecodeError as e:
            raise TaskProfileError(f"Invalid JSON: {e}")

        if not isinstance(data, list):
            raise TaskProfileError("task-profiles.json must contain a list")

        profiles = []
        for entry in data:
            for key in ("name", "model"):
                if key not in entry:
                    raise TaskProfileError(f"Task profile entry missing required field: {key}")
            profiles.append(TaskProfile(name=entry["name"], model_path=self._resolve(entry["model"])))
        return profiles

    def activate(self, name: str, model_mgr: ModelManager) -> Path:
        profiles = self.list_profiles()
        by_name = {p.name: p for p in profiles}
        if name not in by_name:
            raise TaskProfileError(f"Profil de tâche inconnu : {name}")

        path = by_name[name].model_path
        model_mgr.load(path)
        return path
