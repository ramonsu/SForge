"""Reusable procedural knowledge; Skills never authorize execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Skill:
    id: str
    description: str
    instructions: tuple[str, ...]

    def as_context(self, *, sources: tuple[str, ...] = ()) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "instructions": list(self.instructions),
            "sources": list(sources),
            "boundary": "Skill 是方法知识，不是 Capability 或 Permission。",
        }


class SkillRegistry:
    def __init__(self, skills: tuple[Skill, ...]):
        self._skills = {skill.id: skill for skill in skills}
        if len(self._skills) != len(skills):
            raise ValueError("Skill id 不允许重复")

    def get(self, skill_id: str) -> Skill:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise KeyError(f"Skill 不存在: {skill_id}") from exc

    def available(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "id": key,
                "description": self._skills[key].description,
            }
            for key in sorted(self._skills)
        )


def load_skills(path: str | Path | None = None) -> SkillRegistry:
    source = Path(path) if path else (
        Path(__file__).resolve().parent.parent / "config" / "skills.json"
    )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 Skill 配置: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("skills"), list):
        raise ValueError("skills 必须是数组")
    skills = tuple(_parse_skill(value) for value in raw["skills"])
    return SkillRegistry(skills)


def _parse_skill(raw: Any) -> Skill:
    if not isinstance(raw, dict):
        raise ValueError("Skill 必须是对象")
    forbidden = sorted(
        set(raw).intersection(
            {"capabilities", "grants", "permissions", "workspace", "workflow"}
        )
    )
    if forbidden:
        raise ValueError("Skill 不能声明运行控制字段: " + ", ".join(forbidden))
    skill_id = raw.get("id")
    description = raw.get("description")
    instructions = raw.get("instructions")
    if not isinstance(skill_id, str) or not skill_id.strip():
        raise ValueError("Skill id 无效")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Skill description 无效")
    if not isinstance(instructions, list) or not all(
        isinstance(item, str) and item.strip() for item in instructions
    ):
        raise ValueError("Skill instructions 必须是字符串数组")
    return Skill(
        skill_id.strip(),
        description.strip(),
        tuple(item.strip() for item in instructions),
    )
