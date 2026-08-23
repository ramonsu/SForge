"""Immutable human-facing Persona configuration for Agent instances."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


DEFAULT_PERSONA_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "persona.json"
)

# These fields describe runtime authority rather than presentation. Unknown
# fields are otherwise treated as presentation metadata and remain unable to
# enter the operational context.
EXECUTION_FIELDS = {
    "role",
    "roles",
    "permission",
    "permissions",
    "capability",
    "capabilities",
    "skill",
    "skills",
    "tool",
    "tools",
    "workflow",
    "workflows",
    "memory_scope",
    "memory_scopes",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Persona:
    """Static presentation metadata; never a role or permission principal."""

    persona_id: str
    version: str
    name: str
    description: str
    traits: tuple[str, ...]
    communication_style: str
    presentation_metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def reference(self) -> str:
        return f"{self.persona_id}@{self.version}"

    def as_context(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "name": self.name,
            "description": self.description,
            "traits": list(self.traits),
            "communication_style": self.communication_style,
            "presentation": _thaw(self.presentation_metadata),
            "boundary": (
                "仅影响面向用户的表达，不得影响 Workflow、权限、Skill、Tool、"
                "Memory 访问或生命周期。"
            ),
        }


def load_persona(path: str | Path | None = None) -> Persona:
    """Load and validate one Persona file into an immutable reference object."""

    source = Path(path) if path is not None else DEFAULT_PERSONA_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Persona 配置不存在: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Persona 配置不是有效 JSON: {source}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Persona 配置必须是 JSON 对象")

    forbidden = sorted(
        key for key in raw if str(key).casefold() in EXECUTION_FIELDS
    )
    if forbidden:
        raise ValueError(
            f"Persona 不能声明运行控制字段: {', '.join(forbidden)}"
        )

    required = (
        "id",
        "version",
        "name",
        "description",
        "traits",
        "communication_style",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Persona 缺少字段: {', '.join(missing)}")
    for key in ("id", "version", "name", "description", "communication_style"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"Persona 字段 {key} 必须是非空字符串")
    if not isinstance(raw["traits"], list) or any(
        not isinstance(item, str) or not item.strip() for item in raw["traits"]
    ):
        raise ValueError("Persona traits 必须是字符串数组")

    presentation = raw.get("presentation", {})
    if not isinstance(presentation, dict):
        raise ValueError("Persona presentation 必须是对象")
    base_fields = set(required) | {"presentation"}
    metadata = {
        **presentation,
        **{key: value for key, value in raw.items() if key not in base_fields},
    }
    return Persona(
        persona_id=raw["id"].strip(),
        version=raw["version"].strip(),
        name=raw["name"].strip(),
        description=raw["description"].strip(),
        traits=tuple(item.strip() for item in raw["traits"]),
        communication_style=raw["communication_style"].strip(),
        presentation_metadata=_freeze(metadata),
    )
