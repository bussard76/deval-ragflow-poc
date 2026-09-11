"""Central, secret-free catalog of chat models exposed by the web UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelOption:
    """A model/provider pair safe to expose to the browser."""

    id: str
    name: str
    provider: str
    ragflow_model: str
    description: str = ""

    def public_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "description": self.description,
        }


class ModelCatalog:
    """Load enabled model entries from a central JSON file on demand."""

    def __init__(self, path: str | Path, fallback_model: str = ""):
        self.path = Path(path).expanduser()
        self.fallback_model = fallback_model.strip()

    def _raw_entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not load model config: {exc}") from exc
        entries = payload.get("models", []) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise TypeError("model config must contain a models list")
        return [entry for entry in entries if isinstance(entry, dict)]

    def options(self) -> list[ModelOption]:
        result: list[ModelOption] = []
        seen: set[str] = set()
        for entry in self._raw_entries():
            if not entry.get("enabled", True):
                continue
            model_id = str(entry.get("id", "")).strip()
            name = str(entry.get("name", "")).strip()
            provider = str(entry.get("provider", "")).strip()
            ragflow_model = str(
                entry.get("ragflow_model", entry.get("model", model_id))
            ).strip()
            if not model_id or not name or not provider or not ragflow_model:
                raise ValueError(
                    "enabled model entries require id, name, provider, and ragflow_model"
                )
            if model_id in seen:
                raise ValueError(f"duplicate model id: {model_id}")
            seen.add(model_id)
            result.append(
                ModelOption(
                    id=model_id,
                    name=name,
                    provider=provider,
                    ragflow_model=ragflow_model,
                    description=str(entry.get("description", "")).strip(),
                )
            )
        if result or self.path.is_file() or not self.fallback_model:
            return result
        return [
            ModelOption(
                id="default",
                name=self.fallback_model,
                provider="RAGFlow",
                ragflow_model=self.fallback_model,
                description="Aktuelles RAGFlow-Chatmodell",
            )
        ]

    def get(self, model_id: str) -> ModelOption | None:
        return next((item for item in self.options() if item.id == model_id), None)
