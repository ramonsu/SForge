"""Lifecycle manager for homogeneous, disposable Agent processes."""

from __future__ import annotations

from typing import Literal, Protocol
from uuid import uuid4

from agent import Agent
from harness.errors import AgentNotFoundError, InvalidAgentStateError
from harness.identity import Identity
from harness.models import (
    AgentProcess,
    AgentStatus,
    ReasoningResponse,
    RuntimeState,
    TERMINAL_AGENT_STATUSES,
    TaskSpec,
    WorkAssignment,
    utc_now,
)
from harness.persona import Persona


class ProcessSupervisor(Protocol):
    def spawn(self, run_id: str) -> str: ...
    def terminate(self, process_id: str | None) -> None: ...
    def reason(
        self, process_id: str, messages: list[dict]
    ) -> ReasoningResponse: ...
    def close(self) -> None: ...


class AgentManager:
    def __init__(self, processes: ProcessSupervisor):
        self.processes = processes
        self._processes: dict[str, AgentProcess] = {}
        self._runtime_states: dict[str, RuntimeState] = {}
        self._tasks: dict[str, TaskSpec] = {}
        self._agents: dict[str, Agent] = {}
        self._assignments: dict[str, WorkAssignment] = {}
        self._current_assignments: dict[str, str] = {}

    def create(
        self,
        task: TaskSpec,
        *,
        identity: Identity,
        persona: Persona,
        mode: Literal["direct", "workflow"],
        allowed_capabilities: frozenset[str],
        memory_scope: str,
        memory_scopes: tuple[str, ...] = ("core",),
        workflow_id: str | None = None,
        workflow_state_id: str | None = None,
        model_ref: str | None = None,
    ) -> AgentProcess:
        persistent_identity = identity
        agent_id = uuid4().hex
        runtime = RuntimeState(
            agent_id=agent_id,
            task_id=task.id,
            mode=mode,
            identity_id=persistent_identity.id,
            workflow_id=workflow_id,
            workflow_state_id=workflow_state_id,
            allowed_capabilities=allowed_capabilities,
            memory_scope=memory_scope,
            memory_scopes=memory_scopes,
        )
        process = AgentProcess(
            id=agent_id,
            runtime_state_id=runtime.id,
            identity_id=persistent_identity.id,
            model_ref=model_ref,
        )
        process.host_process_id = self.processes.spawn(runtime.id)
        process.status = AgentStatus.RUNNING
        self._processes[agent_id] = process
        self._runtime_states[agent_id] = runtime
        self._tasks[agent_id] = task
        self._agents[agent_id] = Agent(
            process, persistent_identity, persona, self.processes
        )
        return process.snapshot()

    def mount_resources(
        self,
        agent_id: str,
        *,
        cognitive_policy_id: str | None,
        profession_ids: tuple[str, ...],
        memory_scope: str,
        memory_scopes: tuple[str, ...],
    ) -> RuntimeState:
        """Atomically replace non-authoritative cognitive resource bindings."""

        self.require_active(agent_id)
        runtime = self._runtime_states[agent_id]
        runtime.cognitive_policy_id = cognitive_policy_id
        runtime.profession_ids = tuple(profession_ids)
        runtime.memory_scope = memory_scope
        runtime.memory_scopes = tuple(memory_scopes)
        runtime.version += 1
        return runtime.snapshot()

    def mount_workflow_state(
        self,
        agent_id: str,
        *,
        workflow_id: str,
        workflow_state_id: str,
        allowed_capabilities: frozenset[str],
        memory_scope: str,
        memory_scopes: tuple[str, ...],
    ) -> RuntimeState:
        """Atomically replace runtime visibility after admission."""

        self.require_active(agent_id)
        runtime = self._runtime_states[agent_id]
        runtime.mode = "workflow"
        runtime.workflow_id = workflow_id
        runtime.workflow_state_id = workflow_state_id
        runtime.allowed_capabilities = frozenset(allowed_capabilities)
        runtime.memory_scope = memory_scope
        runtime.memory_scopes = tuple(memory_scopes)
        assignment = self.current_assignment(agent_id)
        if assignment is not None:
            mutable = self._assignments[assignment.id]
            mutable.workflow_id = workflow_id
        runtime.version += 1
        return runtime.snapshot()

    def bind_work_assignment(
        self,
        agent_id: str,
        *,
        role_id: str,
        workspace_id: str,
        task_id: str,
        workflow_id: str | None,
        grants: frozenset[str],
        effective_capabilities: frozenset[str],
        workflow_state_id: str | None,
        memory_scope: str,
        memory_scopes: tuple[str, ...],
    ) -> WorkAssignment:
        """Atomically replace one temporary work relationship."""

        self.require_active(agent_id)
        runtime = self._runtime_states[agent_id]
        current = self.current_assignment(agent_id)
        if current is not None and (
            current.role_id == role_id
            and current.workspace_id == workspace_id
            and current.task_id == task_id
            and current.workflow_id == workflow_id
            and current.grants == grants
            and runtime.workflow_state_id == workflow_state_id
        ):
            return current
        if current is not None:
            mutable = self._assignments[current.id]
            mutable.status = "ended"
            mutable.ended_at = utc_now()
        assignment = WorkAssignment(
            agent_process_id=agent_id,
            identity_id=runtime.identity_id,
            workspace_id=workspace_id,
            role_id=role_id,
            task_id=task_id,
            workflow_id=workflow_id,
            grants=frozenset(grants),
        )
        self._assignments[assignment.id] = assignment
        self._current_assignments[agent_id] = assignment.id
        runtime.assignment_id = assignment.id
        runtime.mode = "workflow" if workflow_id else "direct"
        runtime.workflow_id = workflow_id
        runtime.workflow_state_id = workflow_state_id
        runtime.allowed_capabilities = frozenset(effective_capabilities)
        runtime.memory_scope = memory_scope
        runtime.memory_scopes = tuple(memory_scopes)
        runtime.version += 1
        return assignment.snapshot()

    def end_work_assignment(
        self,
        agent_id: str,
        *,
        basic_capabilities: frozenset[str],
        memory_scope: str,
        memory_scopes: tuple[str, ...],
    ) -> WorkAssignment | None:
        self.require_active(agent_id)
        current = self.current_assignment(agent_id)
        if current is None:
            return None
        mutable = self._assignments[current.id]
        mutable.status = "ended"
        mutable.ended_at = utc_now()
        self._current_assignments.pop(agent_id, None)
        runtime = self._runtime_states[agent_id]
        runtime.assignment_id = None
        runtime.mode = "direct"
        runtime.workflow_id = None
        runtime.workflow_state_id = None
        runtime.allowed_capabilities = frozenset(basic_capabilities)
        runtime.memory_scope = memory_scope
        runtime.memory_scopes = tuple(memory_scopes)
        runtime.version += 1
        return mutable.snapshot()

    def current_assignment(self, agent_id: str) -> WorkAssignment | None:
        self._mutable(agent_id)
        assignment_id = self._current_assignments.get(agent_id)
        if assignment_id is None:
            return None
        return self._assignments[assignment_id].snapshot()

    def assignments_snapshot(self) -> tuple[WorkAssignment, ...]:
        return tuple(item.snapshot() for item in self._assignments.values())

    def agent(self, agent_id: str) -> Agent:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise AgentNotFoundError(f"Agent 不存在: {agent_id}") from exc

    def process(self, agent_id: str) -> AgentProcess:
        return self._mutable(agent_id).snapshot()

    def processes_snapshot(self) -> tuple[AgentProcess, ...]:
        return tuple(identity.snapshot() for identity in self._processes.values())

    def runtime_state(self, agent_id: str) -> RuntimeState:
        self._mutable(agent_id)
        return self._runtime_states[agent_id].snapshot()

    def task(self, agent_id: str) -> TaskSpec:
        self._mutable(agent_id)
        return self._tasks[agent_id]

    def mark_running(self, agent_id: str) -> AgentProcess:
        return self._transition(agent_id, AgentStatus.RUNNING)

    def mark_waiting(self, agent_id: str) -> AgentProcess:
        return self._transition(agent_id, AgentStatus.WAITING)

    def complete(self, agent_id: str) -> AgentProcess:
        return self._transition(agent_id, AgentStatus.COMPLETED, stop=True)

    def fail(self, agent_id: str) -> AgentProcess:
        return self._transition(agent_id, AgentStatus.FAILED, stop=True)

    def terminate(self, agent_id: str) -> AgentProcess:
        identity = self._mutable(agent_id)
        if identity.status in TERMINAL_AGENT_STATUSES:
            return identity.snapshot()
        return self._transition(agent_id, AgentStatus.TERMINATED, stop=True)

    def close(self) -> None:
        for agent_id in list(self._processes):
            self.terminate(agent_id)
        self.processes.close()

    def require_active(self, agent_id: str) -> AgentProcess:
        identity = self._mutable(agent_id)
        if identity.status not in {AgentStatus.RUNNING, AgentStatus.WAITING}:
            raise InvalidAgentStateError(
                f"Agent 状态不允许执行: {identity.status.value}"
            )
        return identity.snapshot()

    def _mutable(self, agent_id: str) -> AgentProcess:
        try:
            return self._processes[agent_id]
        except KeyError as exc:
            raise AgentNotFoundError(f"Agent 不存在: {agent_id}") from exc

    def _transition(
        self, agent_id: str, target: AgentStatus, *, stop: bool = False
    ) -> AgentProcess:
        identity = self._mutable(agent_id)
        legal = {
            AgentStatus.RUNNING: {
                AgentStatus.WAITING,
                AgentStatus.COMPLETED,
                AgentStatus.FAILED,
                AgentStatus.TERMINATED,
            },
            AgentStatus.WAITING: {
                AgentStatus.RUNNING,
                AgentStatus.FAILED,
                AgentStatus.TERMINATED,
            },
        }
        if target not in legal.get(identity.status, set()):
            raise InvalidAgentStateError(
                f"非法 Agent 状态迁移: {identity.status.value} -> {target.value}"
            )
        if stop:
            self.processes.terminate(identity.host_process_id)
            identity.host_process_id = None
        identity.status = target
        return identity.snapshot()
