"""Persistent identity configuration, independent of process resources."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.errors import InvalidIdentityError


_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
_AUTHORITY_FIELDS = {
    "allowed_capabilities",
    "capabilities",
    "grants",
    "memory_scope",
    "memory_scopes",
    "permissions",
    "personality",
    "role",
    "roles",
    "skills",
    "tools",
    "workflow",
    "workflow_id",
    "workspace",
    "workspace_id",
}


@dataclass(frozen=True)
class Identity:
    """Long-lived identity reference mounted into disposable processes."""

    id: str
    display_name: str
    owner_binding: str
    created_at: datetime
    persona_reference: str
    default_cognitive_policy_id: str | None = None

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner_binding": self.owner_binding,
            "created_at": self.created_at.isoformat(),
            "default_cognitive_policy_id": self.default_cognitive_policy_id,
            "boundary": (
                "Identity 跨 AgentProcess 与 WorkAssignment 保持连续；"
                "不直接拥有 Workspace、Role、Workflow 或临时权限。"
            ),
        }


def load_identity(path: str | Path | None = None) -> Identity:
    source = Path(path) if path else (
        Path(__file__).resolve().parent.parent / "config" / "identity.json"
    )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidIdentityError(f"无法读取 Identity 配置: {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidIdentityError("Identity 配置必须是对象")
    forbidden = sorted(_AUTHORITY_FIELDS.intersection(raw))
    if forbidden:
        raise InvalidIdentityError(
            "Identity 不能声明运行控制字段: " + ", ".join(forbidden)
        )
    identity_id = _identifier(raw.get("id"), "Identity id")
    persona_reference = raw.get("persona_reference")
    if not isinstance(persona_reference, str) or not persona_reference.strip():
        raise InvalidIdentityError("persona_reference 必须是非空字符串")
    display_name = raw.get("display_name")
    owner_binding = raw.get("owner_binding")
    created_at = raw.get("created_at")
    if not isinstance(display_name, str) or not display_name.strip():
        raise InvalidIdentityError("display_name 必须是非空字符串")
    if not isinstance(owner_binding, str) or not owner_binding.strip():
        raise InvalidIdentityError("owner_binding 必须是非空字符串")
    if not isinstance(created_at, str):
        raise InvalidIdentityError("created_at 必须是 ISO 时间字符串")
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise InvalidIdentityError("created_at 不是有效 ISO 时间") from exc
    if created.tzinfo is None:
        raise InvalidIdentityError("created_at 必须包含时区")
    default_policy = raw.get("default_cognitive_policy_id")
    if default_policy is not None and (
        not isinstance(default_policy, str) or not default_policy.strip()
    ):
        raise InvalidIdentityError(
            "default_cognitive_policy_id 必须是字符串或 null"
        )
    return Identity(
        identity_id,
        display_name.strip(),
        owner_binding.strip(),
        created,
        persona_reference.strip(),
        default_policy.strip() if default_policy else None,
    )


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise InvalidIdentityError(f"{label} 无效")
    return value
