"""Small, typed contracts shared by the SForge V1 runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypeAlias
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


TERMINAL_AGENT_STATUSES = {
    AgentStatus.COMPLETED,
    AgentStatus.FAILED,
    AgentStatus.TERMINATED,
}


@dataclass(frozen=True)
class TaskSpec:
    request: str
    id: str = field(default_factory=lambda: uuid4().hex)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentProcess:
    runtime_state_id: str
    id: str = field(default_factory=lambda: uuid4().hex)
    status: AgentStatus = AgentStatus.CREATED
    created_at: datetime = field(default_factory=utc_now)
    model_ref: str | None = None
    host_process_id: str | None = None

    def snapshot(self) -> "AgentProcess":
        return AgentProcess(
            id=self.id,
            runtime_state_id=self.runtime_state_id,
            status=self.status,
            created_at=self.created_at,
            model_ref=self.model_ref,
            host_process_id=self.host_process_id,
        )


@dataclass
class RuntimeState:
    agent_id: str
    task_id: str
    mode: Literal["direct", "workflow"]
    allowed_capabilities: frozenset[str]
    memory_scope: str
    id: str = field(default_factory=lambda: uuid4().hex)
    workflow_id: str | None = None
    workflow_state_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def snapshot(self) -> "RuntimeState":
        return RuntimeState(
            id=self.id,
            agent_id=self.agent_id,
            task_id=self.task_id,
            mode=self.mode,
            workflow_id=self.workflow_id,
            workflow_state_id=self.workflow_state_id,
            allowed_capabilities=frozenset(self.allowed_capabilities),
            memory_scope=self.memory_scope,
            metadata=deepcopy(self.metadata),
            version=self.version,
        )

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "workflow_id": self.workflow_id,
            "workflow_state_id": self.workflow_state_id,
            "allowed_capabilities": sorted(self.allowed_capabilities),
            "memory_scope": self.memory_scope,
            "metadata": deepcopy(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class CapabilityDescriptor:
    id: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    side_effects: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "input_schema": deepcopy(self.input_schema),
            "output_schema": deepcopy(self.output_schema),
            "side_effects": self.side_effects,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ActionRequest:
    capability_id: str
    arguments: dict[str, Any]
    request_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:
    request_id: str
    capability_id: str
    status: Literal["success", "rejected", "failed"]
    output: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "capability_id": self.capability_id,
            "status": self.status,
            "output": deepcopy(self.output),
            "error": self.error,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class FinalAnswer:
    content: str


AgentDecision: TypeAlias = ActionRequest | FinalAnswer


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    reason: str | None = None


@dataclass(frozen=True)
class MemoryRecord:
    scope: str
    kind: str
    content: str
    id: str = field(default_factory=lambda: uuid4().hex)
    importance: float | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowStateDefinition:
    id: str
    allowed_capabilities: frozenset[str]
    memory_scope: str
    context_sources: tuple[str, ...] = ()

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "allowed_capabilities": sorted(self.allowed_capabilities),
            "memory_scope": self.memory_scope,
            "context_sources": list(self.context_sources),
        }


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    initial_state: str
    states: dict[str, WorkflowStateDefinition]
    instruction: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "initial_state": self.initial_state,
            "states": sorted(self.states),
        }


@dataclass(frozen=True)
class OperationalContext:
    system: dict[str, Any]
    task: dict[str, Any]
    runtime: dict[str, Any]
    memory: tuple[MemoryRecord, ...]
    capabilities: tuple[CapabilityDescriptor, ...]
    workflow: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": deepcopy(self.system),
            "task": deepcopy(self.task),
            "runtime": deepcopy(self.runtime),
            "memory": [
                {
                    "id": record.id,
                    "scope": record.scope,
                    "kind": record.kind,
                    "content": record.content,
                    "importance": record.importance,
                    "created_at": record.created_at.isoformat(),
                    "metadata": deepcopy(record.metadata),
                }
                for record in self.memory
            ],
            "capabilities": [item.as_context() for item in self.capabilities],
            "workflow": deepcopy(self.workflow),
        }
