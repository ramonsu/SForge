"""Lifecycle manager for homogeneous, disposable Agent processes."""

from __future__ import annotations

from typing import Literal, Protocol
from uuid import uuid4

from agent import Agent
from harness.errors import AgentNotFoundError, InvalidAgentStateError
from harness.models import (
    AgentProcess,
    AgentStatus,
    RuntimeState,
    TERMINAL_AGENT_STATUSES,
    TaskSpec,
)
from harness.persona import Persona


class ProcessSupervisor(Protocol):
    def spawn(self, run_id: str) -> str: ...
    def terminate(self, process_id: str | None) -> None: ...
    def reason(self, process_id: str, messages: list[dict]) -> str: ...
    def close(self) -> None: ...


class AgentManager:
    def __init__(self, processes: ProcessSupervisor):
        self.processes = processes
        self._processes: dict[str, AgentProcess] = {}
        self._runtime_states: dict[str, RuntimeState] = {}
        self._tasks: dict[str, TaskSpec] = {}
        self._agents: dict[str, Agent] = {}

    def create(
        self,
        task: TaskSpec,
        *,
        persona: Persona,
        mode: Literal["direct", "workflow"],
        allowed_capabilities: frozenset[str],
        memory_scope: str,
        workflow_id: str | None = None,
        workflow_state_id: str | None = None,
        model_ref: str | None = None,
    ) -> AgentProcess:
        agent_id = uuid4().hex
        runtime = RuntimeState(
            agent_id=agent_id,
            task_id=task.id,
            mode=mode,
            workflow_id=workflow_id,
            workflow_state_id=workflow_state_id,
            allowed_capabilities=allowed_capabilities,
            memory_scope=memory_scope,
        )
        identity = AgentProcess(
            id=agent_id,
            runtime_state_id=runtime.id,
            model_ref=model_ref,
        )
        identity.host_process_id = self.processes.spawn(runtime.id)
        identity.status = AgentStatus.RUNNING
        self._processes[agent_id] = identity
        self._runtime_states[agent_id] = runtime
        self._tasks[agent_id] = task
        self._agents[agent_id] = Agent(identity, persona, self.processes)
        return identity.snapshot()

    def agent(self, agent_id: str) -> Agent:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise AgentNotFoundError(f"Agent 不存在: {agent_id}") from exc

    def process(self, agent_id: str) -> AgentProcess:
        return self._mutable(agent_id).snapshot()

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
