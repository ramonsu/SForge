"""SForge V1 Harness: stable runtime mechanism and execution boundary."""

from __future__ import annotations

import json
from typing import Any, Literal

from harness.agent_manager import AgentManager
from harness.capability import (
    AdmissionPolicy,
    CapabilityRegistry,
)
from harness.context_manager import ContextBundle, ContextManager
from harness.errors import (
    CapabilityNotFoundError,
    InvalidActionArgumentsError,
    InvalidAgentStateError,
)
from harness.memory_manager import MemoryProvider
from harness.models import (
    ActionRequest,
    ActionResult,
    AgentProcess,
    AgentStatus,
    FinalAnswer,
    MemoryRecord,
    OperationalContext,
    RuntimeState,
    TaskSpec,
)
from harness.persona import Persona
from harness.workflow_manager import WorkflowRegistry


class Harness:
    """Own the loop boundary; never decide how a domain task should be solved."""

    def __init__(
        self,
        agents: AgentManager,
        contexts: ContextManager,
        memory: MemoryProvider,
        capabilities: CapabilityRegistry,
        workflows: WorkflowRegistry,
        admission: AdmissionPolicy,
        persona: Persona,
        *,
        direct_capabilities: frozenset[str] = frozenset(),
        max_steps: int = 8,
    ):
        if max_steps < 1:
            raise ValueError("max_steps 必须大于零")
        for capability_id in direct_capabilities:
            capabilities.get(capability_id)
        self._agents = agents
        self._contexts = contexts
        self._memory = memory
        self._capabilities = capabilities
        self._workflows = workflows
        self._admission = admission
        self._persona = persona
        self._direct_capabilities = frozenset(direct_capabilities)
        self._max_steps = max_steps
        self._observations: dict[str, ActionResult] = {}

    def create_agent(
        self,
        task: TaskSpec | str,
        workflow_id: str | None = None,
    ) -> AgentProcess:
        task_spec = task if isinstance(task, TaskSpec) else TaskSpec(str(task))
        if not task_spec.request.strip():
            raise ValueError("Task request 不能为空")

        if workflow_id is None:
            mode = "direct"
            state_id = None
            allowed = self._direct_capabilities
            memory_scope = f"task:{task_spec.id}"
        else:
            definition = self._workflows.get(workflow_id)
            state = definition.states[definition.initial_state]
            mode = "workflow"
            state_id = state.id
            allowed = state.allowed_capabilities
            memory_scope = self._resolve_memory_scope(
                state.memory_scope, task_spec.id, workflow_id
            )

        for capability_id in allowed:
            self._capabilities.get(capability_id)
        return self._agents.create(
            task_spec,
            persona=self._persona,
            mode=mode,
            workflow_id=workflow_id,
            workflow_state_id=state_id,
            allowed_capabilities=allowed,
            memory_scope=memory_scope,
        )

    def terminate_agent(
        self, agent_id: str, reason: str | None = None
    ) -> AgentProcess:
        if reason:
            state = self._agents.runtime_state(agent_id)
            self._memory.write(
                MemoryRecord(
                    scope=state.memory_scope,
                    kind="runtime.termination",
                    content=reason,
                    metadata={"agent_id": agent_id},
                )
            )
        self._observations.pop(agent_id, None)
        return self._agents.terminate(agent_id)

    def build_context(self, agent_id: str) -> OperationalContext:
        return self._bundle(agent_id).operational

    def execute_action(
        self, agent_id: str, request: ActionRequest
    ) -> ActionResult:
        self._agents.require_active(agent_id)
        runtime = self._agents.runtime_state(agent_id)

        try:
            capability = self._capabilities.get(request.capability_id)
        except CapabilityNotFoundError as exc:
            return self._result(
                agent_id, request, "rejected", error=str(exc), stage="resolve"
            )

        try:
            self._capabilities.validate_input(request)
        except InvalidActionArgumentsError as exc:
            return self._result(
                agent_id,
                request,
                "rejected",
                error=str(exc),
                stage="validation",
            )

        decision = self._admission.authorize(
            runtime, request, capability.descriptor
        )
        if not decision.allowed:
            return self._result(
                agent_id,
                request,
                "rejected",
                error=decision.reason or "Action 被拒绝",
                stage="admission",
            )

        try:
            output = capability.invoke(request.arguments)
            self._capabilities.validate_output(request.capability_id, output)
        except Exception as exc:
            return self._result(
                agent_id,
                request,
                "failed",
                error=str(exc),
                stage="execution",
            )
        return self._result(
            agent_id, request, "success", output=output, stage="execution"
        )

    def step(self, agent_id: str) -> FinalAnswer | ActionResult:
        identity = self._agents.require_active(agent_id)
        if identity.status is AgentStatus.WAITING:
            self._agents.mark_running(agent_id)
        try:
            bundle = self._bundle(agent_id)
            agent = self._agents.agent(agent_id)
            decision = agent.run(
                bundle.operational,
                self._agents.task(agent_id).request,
                observation=self._observations.get(agent_id),
            )
        except Exception:
            self._agents.fail(agent_id)
            raise

        if isinstance(decision, FinalAnswer):
            self._write_runtime_memory(
                agent_id, "runtime.final_answer", decision.content
            )
            formatted = agent.format_response(
                bundle.presentation_for(decision.content)
            )
            self._agents.complete(agent_id)
            self._observations.pop(agent_id, None)
            return FinalAnswer(formatted)

        result = self.execute_action(agent_id, decision)
        self._observations[agent_id] = result
        self._agents.mark_waiting(agent_id)
        return result

    def run(self, agent_id: str) -> str:
        for _ in range(self._max_steps):
            outcome = self.step(agent_id)
            if isinstance(outcome, FinalAnswer):
                return outcome.content
        self._agents.fail(agent_id)
        raise InvalidAgentStateError("Agent 超过单次运行允许的最大步骤数")

    def process(self, agent_id: str) -> AgentProcess:
        return self._agents.process(agent_id)

    def runtime_state(self, agent_id: str) -> RuntimeState:
        return self._agents.runtime_state(agent_id)

    def write_memory(
        self,
        agent_id: str,
        kind: str,
        content: str,
        *,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        self._agents.require_active(agent_id)
        state = self._agents.runtime_state(agent_id)
        return self._memory.write(
            MemoryRecord(
                scope=state.memory_scope,
                kind=kind,
                content=content,
                importance=importance,
                metadata=dict(metadata or {}),
            )
        )

    def close(self) -> None:
        self._agents.close()
        self._memory.close()

    def _bundle(self, agent_id: str) -> ContextBundle:
        process = self._agents.require_active(agent_id)
        return self._contexts.build(
            process,
            self._agents.runtime_state(agent_id),
            self._agents.task(agent_id),
            self._agents.agent(agent_id).persona,
        )

    def _result(
        self,
        agent_id: str,
        request: ActionRequest,
        status: Literal["success", "rejected", "failed"],
        *,
        output: Any | None = None,
        error: str | None = None,
        stage: str,
    ) -> ActionResult:
        result = ActionResult(
            request_id=request.request_id,
            capability_id=request.capability_id,
            status=status,
            output=output,
            error=error,
            metadata={"stage": stage},
        )
        state = self._agents.runtime_state(agent_id)
        self._memory.write(
            MemoryRecord(
                scope=state.memory_scope,
                kind="runtime.action_result",
                content=json.dumps(result.as_dict(), ensure_ascii=False),
                metadata={"request_id": request.request_id},
            )
        )
        return result

    def _write_runtime_memory(
        self, agent_id: str, kind: str, content: str
    ) -> MemoryRecord:
        state = self._agents.runtime_state(agent_id)
        return self._memory.write(
            MemoryRecord(
                scope=state.memory_scope,
                kind=kind,
                content=content,
                metadata={"agent_id": agent_id},
            )
        )

    @staticmethod
    def _resolve_memory_scope(
        declared: str, task_id: str, workflow_id: str
    ) -> str:
        if declared == "core":
            return "core"
        if declared == "task":
            return f"task:{task_id}"
        if declared == "workflow":
            return f"workflow:{workflow_id}"
        return declared
