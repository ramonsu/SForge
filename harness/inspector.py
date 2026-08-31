"""Read-only runtime snapshots for local diagnostics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from harness.agent_manager import AgentManager
from harness.capability import CapabilityRegistry
from harness.events import EventLogger
from harness.memory_manager import MemoryProvider
from harness.models import AgentProcess, MemoryRecord, RuntimeState


@dataclass(frozen=True)
class RuntimeSnapshot:
    captured_at: datetime
    agent: dict[str, Any] | None
    runtime_state: dict[str, Any] | None
    work_assignment: dict[str, Any] | None
    workflow: dict[str, Any] | None
    loaded_memory: tuple[dict[str, Any], ...]
    available_capabilities: tuple[dict[str, Any], ...]
    recent_events: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at.isoformat(),
            "agent": deepcopy(self.agent),
            "runtime_state": deepcopy(self.runtime_state),
            "work_assignment": deepcopy(self.work_assignment),
            "workflow": deepcopy(self.workflow),
            "loaded_memory": deepcopy(list(self.loaded_memory)),
            "available_capabilities": deepcopy(
                list(self.available_capabilities)
            ),
            "recent_events": deepcopy(list(self.recent_events)),
        }


class RuntimeInspector:
    """Project runtime state into copies without executing runtime behavior."""

    def __init__(
        self,
        agents: AgentManager,
        memory: MemoryProvider,
        capabilities: CapabilityRegistry,
        events: EventLogger,
    ):
        self._agents = agents
        self._memory = memory
        self._capabilities = capabilities
        self._events = events

    def capture(
        self,
        agent_id: str | None = None,
        *,
        event_limit: int = 20,
        memory_limit: int = 20,
    ) -> RuntimeSnapshot:
        if memory_limit < 1:
            raise ValueError("Memory inspection limit 必须大于零")
        process = self._select_process(agent_id)
        state = self._agents.runtime_state(process.id) if process else None
        assignment = (
            self._agents.current_assignment(process.id) if process else None
        )
        memories = self._loaded_memory(state, memory_limit)
        capabilities = (
            self._capabilities.descriptors(state.allowed_capabilities)
            if state
            else ()
        )
        events = self._events.recent(
            event_limit,
            agent_id=process.id if process else None,
        )
        return RuntimeSnapshot(
            captured_at=datetime.now(timezone.utc),
            agent=self._process_context(process) if process else None,
            runtime_state=self._state_context(state) if state else None,
            work_assignment=(assignment.as_context() if assignment else None),
            workflow=self._workflow_context(state) if state else None,
            loaded_memory=tuple(
                self._memory_context(record) for record in memories
            ),
            available_capabilities=tuple(
                descriptor.as_context() for descriptor in capabilities
            ),
            recent_events=tuple(event.as_dict() for event in events),
        )

    def _select_process(self, agent_id: str | None) -> AgentProcess | None:
        if agent_id is not None:
            return self._agents.process(agent_id)
        processes = self._agents.processes_snapshot()
        return processes[-1] if processes else None

    def _loaded_memory(
        self, state: RuntimeState | None, limit: int
    ) -> list[MemoryRecord]:
        scopes = list(state.memory_scopes) if state else ["core"]
        records: list[MemoryRecord] = []
        for scope in scopes:
            records.extend(self._memory.retrieve(scope=scope, limit=limit))
        return records

    @staticmethod
    def _process_context(process: AgentProcess) -> dict[str, Any]:
        return {
            "id": process.id,
            "runtime_state_id": process.runtime_state_id,
            "status": process.status.value,
            "created_at": process.created_at.isoformat(),
            "model_ref": process.model_ref,
            "host_process_id": process.host_process_id,
        }

    @staticmethod
    def _state_context(state: RuntimeState) -> dict[str, Any]:
        return {
            **state.as_context(),
            "agent_id": state.agent_id,
            "task_id": state.task_id,
        }

    @staticmethod
    def _workflow_context(state: RuntimeState) -> dict[str, Any] | None:
        if state.workflow_id is None:
            return None
        return {
            "id": state.workflow_id,
            "state_id": state.workflow_state_id,
        }

    @staticmethod
    def _memory_context(record: MemoryRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "scope": record.scope,
            "kind": record.kind,
            "content": record.content,
            "importance": record.importance,
            "created_at": record.created_at.isoformat(),
            "metadata": deepcopy(record.metadata),
        }
