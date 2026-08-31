"""Thin public interface for the SForge runtime."""

from __future__ import annotations

from typing import Any, Iterable

from harness.events import EventType, RuntimeEvent
from harness.inspector import RuntimeSnapshot
from harness.models import (
    ActionRequest,
    ActionResult,
    AgentProcess,
    FinalAnswer,
    MemoryRecord,
    OperationalContext,
    RuntimeState,
    ResourceBindingAdmission,
    ResourceBindingRequest,
    TaskSpec,
    WorkAssignment,
    WorkAssignmentAdmission,
    WorkAssignmentRequest,
    WorkflowAdmission,
    WorkflowRequest,
)
from harness.runtime_engine import RuntimeEngine


class Harness:
    """Expose runtime mechanisms without owning task reasoning or policy."""

    def __init__(self, runtime: RuntimeEngine):
        self._runtime = runtime

    def create_agent(
        self,
        task: TaskSpec | str,
        workflow_id: str | None = None,
    ) -> AgentProcess:
        return self._runtime.create_agent(task, workflow_id)

    def terminate_agent(
        self, agent_id: str, reason: str | None = None
    ) -> AgentProcess:
        return self._runtime.terminate_agent(agent_id, reason)

    def build_context(self, agent_id: str) -> OperationalContext:
        return self._runtime.build_context(agent_id)

    def retrieval_trace(self, agent_id: str) -> dict[str, Any]:
        return self._runtime.retrieval_trace(agent_id)

    def execute_action(
        self, agent_id: str, request: ActionRequest
    ) -> ActionResult:
        return self._runtime.execute_action(agent_id, request)

    def request_workflow(
        self, agent_id: str, request: WorkflowRequest
    ) -> WorkflowAdmission:
        return self._runtime.request_workflow(agent_id, request)

    def request_binding(
        self, agent_id: str, request: ResourceBindingRequest
    ) -> ResourceBindingAdmission:
        return self._runtime.request_binding(agent_id, request)

    def request_work_assignment(
        self, agent_id: str, request: WorkAssignmentRequest
    ) -> WorkAssignmentAdmission:
        return self._runtime.request_work_assignment(agent_id, request)

    def end_work_assignment(
        self, agent_id: str, reason: str | None = None
    ) -> WorkAssignment | None:
        return self._runtime.end_work_assignment(agent_id, reason)

    def step(
        self, agent_id: str
    ) -> (
        FinalAnswer
        | ActionResult
        | WorkflowAdmission
        | WorkAssignmentAdmission
        | ResourceBindingAdmission
    ):
        return self._runtime.step(agent_id)

    def run(self, agent_id: str) -> str:
        return self._runtime.run(agent_id)

    def process(self, agent_id: str) -> AgentProcess:
        return self._runtime.process(agent_id)

    def runtime_state(self, agent_id: str) -> RuntimeState:
        return self._runtime.runtime_state(agent_id)

    def available_workflows(self) -> tuple[dict[str, Any], ...]:
        return self._runtime.available_workflows()

    def available_workspaces(self) -> tuple[dict[str, str], ...]:
        return self._runtime.available_workspaces()

    def available_cognitive_policies(self) -> tuple[dict[str, Any], ...]:
        return self._runtime.available_cognitive_policies()

    def available_professions(self) -> tuple[dict[str, Any], ...]:
        return self._runtime.available_professions()

    def available_skills(self) -> tuple[dict[str, Any], ...]:
        return self._runtime.available_skills()

    def available_work_roles(self) -> tuple[dict[str, Any], ...]:
        return self._runtime.available_work_roles()

    def work_assignment(self, agent_id: str) -> WorkAssignment | None:
        return self._runtime.work_assignment(agent_id)

    def recent_events(
        self,
        limit: int = 20,
        *,
        agent_id: str | None = None,
        trace_id: str | None = None,
        event_types: Iterable[EventType] | None = None,
    ) -> tuple[RuntimeEvent, ...]:
        return self._runtime.recent_events(
            limit,
            agent_id=agent_id,
            trace_id=trace_id,
            event_types=event_types,
        )

    def inspect(
        self,
        agent_id: str | None = None,
        *,
        event_limit: int = 20,
        memory_limit: int = 20,
    ) -> RuntimeSnapshot:
        return self._runtime.inspect(
            agent_id,
            event_limit=event_limit,
            memory_limit=memory_limit,
        )

    def write_memory(
        self,
        agent_id: str,
        kind: str,
        content: str,
        *,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        return self._runtime.write_memory(
            agent_id,
            kind,
            content,
            importance=importance,
            metadata=metadata,
        )

    def record_work_experience(
        self,
        agent_id: str,
        lesson: str,
        *,
        objective_outcome: str | None = None,
        self_reflection: str | None = None,
        external_feedback: str | None = None,
        professional_tags: tuple[str, ...] = (),
        artifact_refs: tuple[str, ...] = (),
    ) -> MemoryRecord:
        return self._runtime.record_work_experience(
            agent_id,
            lesson,
            objective_outcome=objective_outcome,
            self_reflection=self_reflection,
            external_feedback=external_feedback,
            professional_tags=professional_tags,
            artifact_refs=artifact_refs,
        )

    def close(self) -> None:
        self._runtime.close()
