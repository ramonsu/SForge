"""Declarative Assignment responsibility with no independent lifecycle."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.errors import InvalidWorkRoleError, WorkRoleNotFoundError


_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_FORBIDDEN_FIELDS = {
    "allowed_capabilities",
    "capabilities",
    "capability_set",
    "grants",
    "identity",
    "memory_scope",
    "memory_scopes",
    "memory_write_scope",
    "permissions",
    "persona",
    "personality",
    "skills",
    "tools",
    "tool_ids",
    "workflow",
    "workflow_id",
    "workflow_state",
    "workspace",
    "workspace_id",
    "workspace_root",
    "lifecycle",
}


@dataclass(frozen=True)
class WorkRole:
    id: str
    description: str
    instructions: tuple[str, ...]
    evaluation_criteria: tuple[str, ...] = ()

    @property
    def context_instructions(self) -> tuple[str, ...]:
        return self.instructions

    @property
    def memory_hints(self) -> tuple[str, ...]:
        """Deprecated V1.5 alias; retrieval moved to Profession."""

        return ()

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "instructions": list(self.instructions),
            "context_instructions": list(self.instructions),
            "memory_hints": [],
            "evaluation_criteria": list(self.evaluation_criteria),
            "boundary": (
                "WorkRole 只描述当前 Workspace 内的临时职责；专业资源来自"
                " Profession，权限只来自 WorkAssignment grants。"
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "instructions": list(self.instructions),
        }


class WorkRoleRegistry:
    def __init__(self, roles: tuple[WorkRole, ...]):
        self._roles = {role.id: role for role in roles}
        if len(self._roles) != len(roles):
            raise InvalidWorkRoleError("WorkRole id 不允许重复")

    def get(self, role_id: str) -> WorkRole:
        try:
            return self._roles[role_id]
        except KeyError as exc:
            raise WorkRoleNotFoundError(f"WorkRole 不存在: {role_id}") from exc

    def available(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self._roles[role_id].summary()
            for role_id in sorted(self._roles)
        )


def load_work_roles(path: str | Path | None = None) -> WorkRoleRegistry:
    source = Path(path) if path else (
        Path(__file__).resolve().parent.parent / "config" / "work_roles.json"
    )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidWorkRoleError(f"无法读取 WorkRole 配置: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("roles"), list):
        raise InvalidWorkRoleError("roles 必须是数组")
    roles = tuple(_parse_role(value) for value in raw["roles"])
    if not roles:
        raise InvalidWorkRoleError("roles 不能为空")
    return WorkRoleRegistry(roles)


def _parse_role(raw: Any) -> WorkRole:
    if not isinstance(raw, dict):
        raise InvalidWorkRoleError("WorkRole 必须是对象")
    forbidden = sorted(_FORBIDDEN_FIELDS.intersection(raw))
    if forbidden:
        raise InvalidWorkRoleError(
            "WorkRole 不能声明运行控制字段: " + ", ".join(forbidden)
        )
    role_id = raw.get("id")
    if not isinstance(role_id, str) or not _IDENTIFIER.fullmatch(role_id):
        raise InvalidWorkRoleError("WorkRole id 无效")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise InvalidWorkRoleError("description 必须是字符串")
    instructions = _string_list(raw, "instructions", required=True)
    evaluation = _string_list(raw, "evaluation_criteria", required=False)
    return WorkRole(
        role_id, description.strip(), tuple(instructions), tuple(evaluation)
    )


def _string_list(raw: dict[str, Any], key: str, *, required: bool) -> list[str]:
    value = raw.get(key, [] if not required else None)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise InvalidWorkRoleError(f"{key} 必须是字符串数组")
    if required and not value:
        raise InvalidWorkRoleError(f"{key} 不能为空")
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise InvalidWorkRoleError(f"{key} 不允许重复项")
    return normalized
