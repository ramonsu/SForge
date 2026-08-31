"""Professional resource configurations independent of roles and grants."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Profession:
    id: str
    description: str
    memory_tags: tuple[str, ...]
    knowledge_references: tuple[str, ...]
    preferred_skills: tuple[str, ...]
    methods: tuple[str, ...]
    evaluation_criteria: tuple[str, ...]

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "memory": {"tags": list(self.memory_tags)},
            "knowledge_references": list(self.knowledge_references),
            "skills": {"preferred": list(self.preferred_skills)},
            "methods": list(self.methods),
            "evaluation_criteria": list(self.evaluation_criteria),
            "boundary": "Profession 提供专业资源，不授予 Capability。",
        }

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "description": self.description}


class ProfessionRegistry:
    def __init__(self, professions: tuple[Profession, ...]):
        self._professions = {item.id: item for item in professions}
        if len(self._professions) != len(professions):
            raise ValueError("Profession id 不允许重复")

    def get(self, profession_id: str) -> Profession:
        try:
            return self._professions[profession_id]
        except KeyError as exc:
            raise KeyError(f"Profession 不存在: {profession_id}") from exc

    def available(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self._professions[key].summary()
            for key in sorted(self._professions)
        )


def load_professions(path: str | Path | None = None) -> ProfessionRegistry:
    source = Path(path) if path else (
        Path(__file__).resolve().parent.parent / "config" / "professions.json"
    )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 Profession 配置: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("professions"), list):
        raise ValueError("professions 必须是数组")
    professions = tuple(_parse_profession(value) for value in raw["professions"])
    return ProfessionRegistry(professions)


def _parse_profession(raw: Any) -> Profession:
    if not isinstance(raw, dict):
        raise ValueError("Profession 必须是对象")
    forbidden = sorted(
        set(raw).intersection(
            {
                "allowed_capabilities",
                "capabilities",
                "grants",
                "permissions",
                "role",
                "workflow",
                "workspace",
            }
        )
    )
    if forbidden:
        raise ValueError(
            "Profession 不能声明运行控制字段: " + ", ".join(forbidden)
        )
    profession_id = raw.get("id")
    description = raw.get("description")
    if not isinstance(profession_id, str) or not profession_id.strip():
        raise ValueError("Profession id 无效")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Profession description 无效")
    return Profession(
        profession_id.strip(),
        description.strip(),
        _strings(raw, "memory_tags"),
        _strings(raw, "knowledge_references"),
        _strings(raw, "preferred_skills"),
        _strings(raw, "methods"),
        _strings(raw, "evaluation_criteria"),
    )


def _strings(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"Profession {key} 必须是字符串数组")
    normalized = tuple(item.strip() for item in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Profession {key} 不允许重复项")
    return normalized
