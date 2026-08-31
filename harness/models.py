"""Small, typed contracts shared by the SForge runtime."""

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
    identity_id: str = "default"
    status: AgentStatus = AgentStatus.CREATED
    created_at: datetime = field(default_factory=utc_now)
    model_ref: str | None = None
    host_process_id: str | None = None

    def snapshot(self) -> "AgentProcess":
        return AgentProcess(
            id=self.id,
            runtime_state_id=self.runtime_state_id,
            identity_id=self.identity_id,
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
    identity_id: str = "default"
    cognitive_policy_id: str | None = None
    profession_ids: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: uuid4().hex)
    assignment_id: str | None = None
    workflow_id: str | None = None
    workflow_state_id: str | None = None
    memory_scopes: tuple[str, ...] = ("core",)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def snapshot(self) -> "RuntimeState":
        return RuntimeState(
            id=self.id,
            agent_id=self.agent_id,
            task_id=self.task_id,
            mode=self.mode,
            identity_id=self.identity_id,
            cognitive_policy_id=self.cognitive_policy_id,
            profession_ids=tuple(self.profession_ids),
            assignment_id=self.assignment_id,
            workflow_id=self.workflow_id,
            workflow_state_id=self.workflow_state_id,
            allowed_capabilities=frozenset(self.allowed_capabilities),
            memory_scope=self.memory_scope,
            memory_scopes=tuple(self.memory_scopes),
            metadata=deepcopy(self.metadata),
            version=self.version,
        )

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "identity_id": self.identity_id,
            "cognitive_policy_id": self.cognitive_policy_id,
            "profession_ids": list(self.profession_ids),
            "assignment_id": self.assignment_id,
            "workflow_id": self.workflow_id,
            "workflow_state_id": self.workflow_state_id,
            "allowed_capabilities": sorted(self.allowed_capabilities),
            "memory_scope": self.memory_scope,
            "memory_scopes": list(self.memory_scopes),
            "metadata": deepcopy(self.metadata),
            "version": self.version,
        }


@dataclass
class WorkAssignment:
    """One temporary relationship between a process and a Workspace."""

    agent_process_id: str
    identity_id: str
    workspace_id: str
    role_id: str
    task_id: str
    grants: frozenset[str]
    id: str = field(default_factory=lambda: uuid4().hex)
    workflow_id: str | None = None
    status: Literal["active", "ended"] = "active"
    created_at: datetime = field(default_factory=utc_now)
    ended_at: datetime | None = None

    def snapshot(self) -> "WorkAssignment":
        return WorkAssignment(
            id=self.id,
            agent_process_id=self.agent_process_id,
            identity_id=self.identity_id,
            workspace_id=self.workspace_id,
            role_id=self.role_id,
            task_id=self.task_id,
            workflow_id=self.workflow_id,
            grants=frozenset(self.grants),
            status=self.status,
            created_at=self.created_at,
            ended_at=self.ended_at,
        )

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_process_id": self.agent_process_id,
            "identity_id": self.identity_id,
            "workspace_id": self.workspace_id,
            "role_id": self.role_id,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "grants": sorted(self.grants),
            "status": self.status,
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
class WorkflowRequest:
    workflow_id: str
    target_state_id: str | None = None
    transition_condition: str | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class WorkflowAdmission:
    request_id: str
    status: Literal["success", "rejected"]
    workflow_id: str
    previous_state_id: str | None = None
    workflow_state_id: str | None = None
    memory_scope: str | None = None
    memory_scopes: tuple[str, ...] = ()
    allowed_capabilities: frozenset[str] = frozenset()
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "workflow_id": self.workflow_id,
            "previous_state_id": self.previous_state_id,
            "workflow_state_id": self.workflow_state_id,
            "memory_scope": self.memory_scope,
            "memory_scopes": list(self.memory_scopes),
            "allowed_capabilities": sorted(self.allowed_capabilities),
            "error": self.error,
        }


@dataclass(frozen=True)
class WorkAssignmentRequest:
    role_id: str
    workspace_id: str | None = None
    task_id: str | None = None
    workflow_id: str | None = None
    target_state_id: str | None = None
    requested_capabilities: tuple[str, ...] = ()
    request_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class WorkAssignmentAdmission:
    request_id: str
    status: Literal["success", "rejected"]
    role_id: str
    workspace_id: str
    task_id: str
    assignment_id: str | None = None
    previous_assignment_id: str | None = None
    workflow_id: str | None = None
    workflow_state_id: str | None = None
    memory_scope: str | None = None
    memory_scopes: tuple[str, ...] = ()
    grants: frozenset[str] = frozenset()
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "assignment_id": self.assignment_id,
            "previous_assignment_id": self.previous_assignment_id,
            "role_id": self.role_id,
            "workspace_id": self.workspace_id,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "workflow_state_id": self.workflow_state_id,
            "memory_scope": self.memory_scope,
            "memory_scopes": list(self.memory_scopes),
            "grants": sorted(self.grants),
            "error": self.error,
        }


@dataclass(frozen=True)
class ResourceBindingRequest:
    resource_type: Literal["cognitive_policy", "profession"]
    operation: Literal["activate", "deactivate"]
    resource_id: str | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class ResourceBindingAdmission:
    request_id: str
    status: Literal["success", "rejected"]
    resource_type: Literal["cognitive_policy", "profession"]
    operation: Literal["activate", "deactivate"]
    resource_id: str | None = None
    cognitive_policy_id: str | None = None
    profession_ids: tuple[str, ...] = ()
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "resource_type": self.resource_type,
            "operation": self.operation,
            "resource_id": self.resource_id,
            "cognitive_policy_id": self.cognitive_policy_id,
            "profession_ids": list(self.profession_ids),
            "error": self.error,
        }


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "TokenUsage":
        value = value or {}
        return cls(
            input_tokens=int(value.get("input_tokens", 0) or 0),
            output_tokens=int(value.get("output_tokens", 0) or 0),
        )


@dataclass(frozen=True)
class ReasoningResponse:
    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class FinalAnswer:
    content: str
    primary_evidence_id: str | None = None
    secondary_evidence_ids: tuple[str, ...] = ()
    final_choice: str | None = None


AgentDecision: TypeAlias = (
    ActionRequest
    | WorkflowRequest
    | WorkAssignmentRequest
    | ResourceBindingRequest
    | FinalAnswer
)
AgentObservation: TypeAlias = (
    ActionResult
    | WorkflowAdmission
    | WorkAssignmentAdmission
    | ResourceBindingAdmission
)


@dataclass(frozen=True)
class AgentTurn:
    decision: AgentDecision
    usage: TokenUsage = field(default_factory=TokenUsage)
    decision_protocol: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedResponse:
    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)


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
    memory_scopes: tuple[str, ...]
    memory_write_scope: str
    context: str = ""
    goal: str = ""
    context_sources: tuple[str, ...] = ()
    memory_hints: tuple[str, ...] = ()
    evaluation_criteria: tuple[str, ...] = ()

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "context": self.context,
            "goal": self.goal,
            "allowed_capabilities": sorted(self.allowed_capabilities),
            "memory_scopes": list(self.memory_scopes),
            "memory_write_scope": self.memory_write_scope,
            "context_sources": list(self.context_sources),
            "memory_hints": list(self.memory_hints),
            "evaluation_criteria": list(self.evaluation_criteria),
        }


@dataclass(frozen=True)
class WorkflowTransitionDefinition:
    condition: str
    target: str

    def as_context(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "target": self.target,
        }


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    initial_state: str
    states: dict[str, WorkflowStateDefinition]
    instruction: str = ""
    description: str = ""
    transitions: dict[str, tuple[WorkflowTransitionDefinition, ...]] = field(
        default_factory=dict
    )

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "initial_state": self.initial_state,
            "states": sorted(self.states),
            "transitions": {
                source: [edge.as_context() for edge in edges]
                for source, edges in self.transitions.items()
            },
        }

    def outgoing(
        self, state_id: str
    ) -> tuple[WorkflowTransitionDefinition, ...]:
        return self.transitions.get(state_id, ())


@dataclass(frozen=True)
class OperationalContext:
    system: dict[str, Any]
    task: dict[str, Any]
    runtime: dict[str, Any]
    memory: tuple[MemoryRecord, ...]
    capabilities: tuple[CapabilityDescriptor, ...]
    identity: dict[str, Any] | None = None
    cognitive_policy: dict[str, Any] | None = None
    professions: tuple[dict[str, Any], ...] = ()
    skills: tuple[dict[str, Any], ...] = ()
    workspace: dict[str, Any] | None = None
    work_assignment: dict[str, Any] | None = None
    work_role: dict[str, Any] | None = None
    workflow: dict[str, Any] | None = None
    model_projection: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": deepcopy(self.system),
            "task": deepcopy(self.task),
            "runtime": deepcopy(self.runtime),
            "identity": deepcopy(self.identity),
            "cognitive_policy": deepcopy(self.cognitive_policy),
            "professions": deepcopy(self.professions),
            "skills": deepcopy(self.skills),
            "workspace": deepcopy(self.workspace),
            "work_assignment": deepcopy(self.work_assignment),
            "work_role": deepcopy(self.work_role),
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

    def for_model(self) -> dict[str, Any]:
        """Return the explicit model-facing view, if one was constructed."""

        if self.model_projection is not None:
            return deepcopy(self.model_projection)
        return self.as_dict()
