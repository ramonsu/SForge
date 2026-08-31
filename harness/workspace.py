"""Declarative project environment metadata and archive retrieval config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Workspace:
    id: str
    description: str
    retrieval_sources: tuple[str, ...]
    retrieval_preferences: tuple[str, ...]
    knowledge_references: tuple[str, ...]
    local_skills: tuple[str, ...]

    @property
    def archive_scope(self) -> str:
        return f"workspace:{self.id}"

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "archive_scope": self.archive_scope,
            "retrieval": {
                "sources": list(self.retrieval_sources),
                "prefer": list(self.retrieval_preferences),
            },
            "local_knowledge": {
                "references": list(self.knowledge_references)
            },
            "local_skills": list(self.local_skills),
            "boundary": (
                "Workspace 提供项目资源与 Archive 检索配置；"
                "不代表 Identity Memory，也不自动授予 Capability。"
            ),
        }


def load_workspace(
    workspace_id: str, path: str | Path | None = None
) -> Workspace:
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        raise ValueError("Workspace id 不能为空")
    source = Path(path) if path else (
        Path(__file__).resolve().parent.parent / "config" / "workspace.json"
    )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 Workspace 配置: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Workspace 配置必须是对象")
    forbidden = sorted(
        set(raw).intersection({"capabilities", "grants", "permissions"})
    )
    if forbidden:
        raise ValueError(
            "Workspace 不能声明 Capability grants: " + ", ".join(forbidden)
        )
    description = raw.get("description", "Project workspace")
    retrieval = raw.get("retrieval", {})
    if not isinstance(description, str) or not isinstance(retrieval, dict):
        raise ValueError("Workspace description/retrieval 无效")
    return Workspace(
        workspace_id.strip(),
        description.strip(),
        _strings(retrieval, "sources"),
        _strings(retrieval, "prefer"),
        _strings(raw, "knowledge_references"),
        _strings(raw, "local_skills"),
    )


def _strings(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"Workspace {key} 必须是字符串数组")
    return tuple(dict.fromkeys(item.strip() for item in value))
