"""Serializable contracts exposed to SForge frontends."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from harness.models import TokenUsage


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressKind(str, Enum):
    RUN_QUEUED = "run_queued"
    RUN_STARTED = "run_started"
    AGENT_CREATED = "agent_created"
    WORKFLOW_REQUESTED = "workflow_requested"
    WORKFLOW_ADMISSION_COMPLETED = "workflow_admission_completed"
    WORK_ASSIGNMENT_REQUESTED = "work_assignment_requested"
    WORK_ASSIGNMENT_ADMISSION_COMPLETED = (
        "work_assignment_admission_completed"
    )
    WORK_ASSIGNMENT_ENDED = "work_assignment_ended"
    RESOURCE_BINDING_REQUESTED = "resource_binding_requested"
    RESOURCE_BINDING_COMPLETED = "resource_binding_completed"
    CONTEXT_READY = "context_ready"
    REASONING_STARTED = "reasoning_started"
    REASONING_COMPLETED = "reasoning_completed"
    CAPABILITY_REQUESTED = "capability_requested"
    CAPABILITY_COMPLETED = "capability_completed"
    ACTION_COMPLETED = "action_completed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"ProgressEvent data 不支持类型: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    sequence: int
    kind: ProgressKind
    message: str
    timestamp: datetime = field(default_factory=_utc_now)
    data: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.run_id.strip() or self.sequence < 1:
            raise ValueError("ProgressEvent run_id 和 sequence 无效")
        if self.timestamp.tzinfo is None:
            raise ValueError("ProgressEvent timestamp 必须包含时区")
        object.__setattr__(
            self, "timestamp", self.timestamp.astimezone(timezone.utc)
        )
        object.__setattr__(
            self,
            "data",
            _freeze_json(dict(self.data)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "data": _thaw_json(self.data),
        }


@dataclass(frozen=True)
class RunSnapshot:
    id: str
    status: RunStatus
    request: str
    workflow_id: str | None
    workflow_state_id: str | None
    assignment_id: str | None
    work_role_id: str | None
    workspace_id: str | None
    cognitive_policy_id: str | None
    profession_ids: tuple[str, ...]
    agent_id: str | None
    stage: str
    current_capability: str | None
    token_usage: TokenUsage
    answer: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "request": self.request,
            "workflow_id": self.workflow_id,
            "workflow_state_id": self.workflow_state_id,
            "assignment_id": self.assignment_id,
            "work_role_id": self.work_role_id,
            "workspace_id": self.workspace_id,
            "cognitive_policy_id": self.cognitive_policy_id,
            "profession_ids": list(self.profession_ids),
            "agent_id": self.agent_id,
            "stage": self.stage,
            "current_capability": self.current_capability,
            "token_usage": self.token_usage.as_dict(),
            "answer": self.answer,
            "error": self.error,
        }
