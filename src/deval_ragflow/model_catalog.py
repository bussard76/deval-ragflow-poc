"""Convert RAGFlow's chat-model response into browser-safe options."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


def _text(remote: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = remote.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def remote_model_options(
    remote_models: Sequence[Mapping[str, Any]],
) -> list[ModelOption]:
    """Build options directly from RAGFlow's allowed chat models."""

    result: list[ModelOption] = []
    seen_references: set[str] = set()
    seen_ids: set[str] = set()
    for remote in remote_models:
        name = _text(remote, "name", "model_name", "model")
        model_id = _text(remote, "model_id", "id")
        if not name and not model_id:
            continue
        instance = _text(remote, "instance_name", "instance")
        provider = _text(remote, "provider_name", "provider") or "RAGFlow"
        reference = (
            f"{name}@{instance}@{provider}"
            if name and instance and provider
            else name or model_id
        )
        if not reference or reference in seen_references:
            continue
        option_id = name or reference
        if option_id in seen_ids:
            option_id = reference
        if option_id in seen_ids:
            continue
        seen_references.add(reference)
        seen_ids.add(option_id)
        provider_label = f"{provider} · {instance}" if instance else provider
        result.append(
            ModelOption(
                id=option_id,
                name=name or model_id,
                provider=provider_label,
                ragflow_model=reference,
                description=_text(remote, "description"),
            )
        )
    return result
